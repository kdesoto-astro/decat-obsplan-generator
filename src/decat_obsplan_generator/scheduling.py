# Where schedule and observing blocks are generated.
import os
import warnings
from pathlib import Path
from datetime import datetime

import astroplan
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

from decat_obsplan_generator.airmass import AirmassCalculator
from decat_obsplan_generator.helpers import (
    MANY_DASHES,
    OBS_PLAN_HEADER,
    calc_night_limits,
    ctio_location,
    get_all_jsons_from_directory,
    get_program_name,
    merge_json_same_night,
    nice_duration_formatting,
    nice_time_formatting,
    read_json,
    universal_to_local_time,
)
from decat_obsplan_generator.moon import MoonCalculator

warnings.filterwarnings("ignore")


def generate_nightly_schedule(directory, date=None, observer=None):
    """Generate all observing blocks
    from folders of jsons.
    """
    if observer is None:
        observer = ctio_location()

    if date is None:
        current_time = Time.now()
        current_datetime = universal_to_local_time(current_time, observer)
        date = current_datetime.strftime("%y%m%d")

    blocks = []
    night_start, night_end = calc_night_limits(date=date)
    airmass_calc = AirmassCalculator()
    airmass_calc.load_airmass_grid()

    all_ids = []
    for fn, prio in zip(*get_all_jsons_from_directory(directory, date)):
        names, jsons, propids, ras, decs, exptimes = read_json(fn)
        if jsons is None:
            print(f"Skipped: {fn}")
            continue
        ra, dec, dur = merge_json_same_night(ras, decs, exptimes)
        early_limit, late_limit = airmass_calc.observable_range(ra, dec, night_start, night_end)
        early_limit2, late_limit2 = airmass_calc.observable_range(
            ra, dec, night_start, night_end, max_airmass=2.0
        )

        if early_limit is not None:
            early_limit = max(early_limit, night_start)
            late_limit = min(late_limit, night_end)

        if early_limit2 is not None:
            early_limit2 = max(early_limit2, night_start)
            late_limit2 = min(late_limit2, night_end)

        name = names[0].strip().split()[0]
        while name in all_ids:
            name += "'"

        block = {
            "name": name,
            "json": jsons[0][:-5],
            "ra": ra,
            "dec": dec,
            "coord": SkyCoord(ra=ra, dec=dec),
            "dur": dur * u.min,
            "early_limit": early_limit2,
            "late_limit": late_limit2,
            "prio": prio,
            "propid": propids[0],
        }

        block["secz18_beg"] = universal_to_local_time(early_limit, observer).strftime("%H:%M")
        block["secz18_end"] = universal_to_local_time(late_limit, observer).strftime("%H:%M")
        block["secz2_beg"] = universal_to_local_time(early_limit2, observer).strftime("%H:%M")
        block["secz2_end"] = universal_to_local_time(late_limit2, observer).strftime("%H:%M")

        block["program"] = get_program_name(block["propid"])

        all_ids.append(name)
        blocks.append(block)

    schedule = DECATSchedule(date, night_start, night_end, blocks, observer=observer)
    return schedule


class DECATSchedule:
    """Modified schedule for DECAT, which includes total program
    allocated times.
    """

    def __init__(
            self, date, start_time, end_time,
            all_blocks, observer=None
        ):
        """Adds program time dictionary."""
        self.start_time = start_time
        self.end_time = end_time
        self.date = date

        if observer is None:
            self.observer = ctio_location()
        else:
            self.observer = observer

        program_time_fn = os.path.join(
            Path(__file__).parent.parent.parent.absolute(), "data/program_times.csv"
        )
        sched = pd.read_csv(program_time_fn)
        month = start_time.to_datetime().strftime("%b")
        self.program_times = sched.set_index("propid")[month].to_dict()
        self.program_times = {k: v * u.min for (k, v) in self.program_times.items()}

        self._df = pd.DataFrame.from_records(all_blocks)
        self._df.loc[
            :,
            [
                "t_start",
                "t_end",
                "secz_i",
                "secz_f",
                "slew_s",
                "order",
                "eff_prio",
                "current_time",
                "t_start_hidden",
                "t_end_hidden",
            ],
        ] = pd.NA

        self._display_columns = [
            "order",
            "json",
            "program",
            "propid",
            "prio",
            "dur",
            "ra",
            "dec",
            "secz18_beg",
            "secz18_end",
            "secz2_beg",
            "secz2_end",
            "t_start",
            "t_end",
            "slew_s",
            "secz_i",
            "secz_f",
        ]

        
        self._df = self._df[[*self._display_columns, *self._df.columns[~self._df.columns.isin(self._display_columns)]]]
        self._df.set_index("name", inplace=True)

        self.ac = AirmassCalculator()
        self.ac.load_airmass_grid()

        # timezone load
        datetime_date = datetime.strptime(self.date, "%y%m%d")
        self.observer_shift = self.observer.timezone.utcoffset(datetime_date).total_seconds() * u.s

        ctio_observer = ctio_location()
        self.datetime = Time(datetime_date, format='datetime')
        dummy_sidereal_time = ctio_observer.local_sidereal_time(self.datetime).to(u.hourangle)
        self.sidereal_shift = dummy_sidereal_time.hour * u.hour
        self.midnight = self.observer.midnight(self.datetime, which="next")

        # moon load
        self.moon = MoonCalculator()


    def display(self):
        print(self._df[self._display_columns].to_string())

    def _slew_time(self, coord1, coord2):
        """Calculate slew time between two coordinates."""
        sep = coord1.separation(coord2).to(u.deg).value
        slew_rate = 5 / 11.0
        # readout is about 29 s, and slew time graph has y-intercept of ~29 s
        # So 29 s is added either way
        slew_time = sep / slew_rate + 29.0

        # account for equatorial mount flip
        if (coord1.dec < -30.17 * u.deg) and (coord2.dec > -30.17 * u.deg):
            slew_time += 180.0
        elif (coord1.dec > -30.17 * u.deg) and (coord2.dec < -30.17 * u.deg):
            slew_time += 180.0

        return pd.Timedelta(slew_time, unit="s")

    def _update_obs_times(self):
        """Based on target order, calculate slew time
        and update start/end times."""
        all_orders = self._df.order.dropna().unique()

        for i, order in enumerate(all_orders):
            if i == 0:
                self._df.loc[self._df.order == order, "t_start_hidden"] = self.start_time
                self._df.loc[self._df.order == order, "slew_s"] = pd.Timedelta(0.0, unit="m")
                self._df.loc[self._df.order == order, "t_end_hidden"] = (
                    self.start_time + self._df.loc[self._df.order == order, "dur"]
                )
            else:
                # calculate slew time
                prev_coord = self._df.loc[self._df.order == all_orders[i-1], "coord"].item()
                curr_coord = self._df.loc[self._df.order == order, "coord"].item()
                slew_time = self._slew_time(prev_coord, curr_coord)

                prev_t_end = self._df.loc[self._df.order == all_orders[i-1], "t_end_hidden"].item()
                self._df.loc[self._df.order == order, "t_start_hidden"] = prev_t_end
                self._df.loc[self._df.order == order, "slew_s"] = slew_time
                self._df.loc[self._df.order == order, "t_end_hidden"] = (
                    prev_t_end + self._df.loc[self._df.order == order, "dur"] + slew_time
                )

        self._move_unused_time_to_middle()
        self._update_airmasses()


    def _move_unused_time_to_middle(self):
        """If gap at end of night, shift targets to end to make empty room
        in middle of night. Cutpoint is either midnight or when targets'
        latest obstime is end of night (whichever is later).
        """
        obs_end = self._df.loc[~self._df.order.isna(), 't_end_hidden'].max()
        unused_time = TimeDelta(self.end_time - obs_end)
        cutpoint1_mask = np.array([(abs(x - self.end_time) < 1 * u.min) for x in self._df.late_limit]) & ~self._df.order.isna()
        cutpoint1 = self._df.loc[cutpoint1_mask, 't_start_hidden'].min()
        cutpoint = max(cutpoint1, self.midnight)
        mask = (~self._df.order.isna()) & (self._df.t_start_hidden >= cutpoint)
        self._df.loc[mask, 't_start_hidden'] = [x + unused_time for x in self._df.loc[mask, 't_start_hidden']]
        self._df.loc[mask, 't_end_hidden'] = [x + unused_time for x in self._df.loc[mask, 't_end_hidden']]
        
    
    def _update_plots(self):
        """Generate airmass + moon distance plots."""
        self.save_airmass_plot()
        self.save_moon_distance_plot()


    def save_moon_distance_plot(self):
        """Save plot of moon distances."""
        save_dir = f"../../decat_pointings/2025A/{self.date}"
        fig = self.moon.plot_moon_distances(
            self._df, self.start_time, self.end_time,
            [self.sidereal_shift, self.observer_shift]
        )
        fig.savefig(os.path.join(save_dir, "moon_distances.png"), bbox_inches='tight', dpi=300)
        plt.close()


    def _update_airmasses(self):
        """Update airmasses of generated schedule."""
        self._df["secz_i"] = [
            round(self.ac.query(row.t_start_hidden, row.ra, row.dec)[0], 2) for row in self._df.itertuples()
        ]
        self._df["secz_f"] = [
            round(self.ac.query(row.t_end_hidden, row.ra, row.dec)[0], 2) for row in self._df.itertuples()
        ]

        # underground_mask = self._df.secz_i.isna() | self._df.secz_f.isna()
        daylight_mask = (self._df.t_end_hidden > self.end_time) | (self._df.t_start_hidden < self.start_time)
        not_visible_mask = daylight_mask  # | underground_mask

        # formatting
        self._df.loc[not_visible_mask, ["t_start", "t_end"]] = pd.NaT
        self._df.loc[not_visible_mask, ["secz_i", "secz_f", "order"]] = pd.NA
        self._df.loc[not_visible_mask, "slew_s"] = pd.Timedelta(0.0, unit="m")

        self._df.loc[~not_visible_mask, "t_start"] = self._df.loc[~not_visible_mask, "t_start_hidden"].apply(
            universal_to_local_time, args=(self.observer,)
        )
        self._df.loc[~not_visible_mask, "t_end"] = self._df.loc[~not_visible_mask, "t_end_hidden"].apply(
            universal_to_local_time, args=(self.observer,)
        )
        self._df.loc[~not_visible_mask, "t_start"] = self._df.loc[~not_visible_mask, "t_start"].apply(
            nice_time_formatting
        )
        self._df.loc[~not_visible_mask, "t_end"] = self._df.loc[~not_visible_mask, "t_end"].apply(
            nice_time_formatting
        )

        self._df.loc[~not_visible_mask, "slew_s"] = self._df.loc[~not_visible_mask, "slew_s"].apply(
            nice_duration_formatting
        )

        self._df.sort_values(by="order", inplace=True)
        self.save()
        self._update_plots()


    def move_target(self, name, order):
        """Move target to spot 'order'. Move all targets that were before it
        one spot down."""
        prev_order = self._df.loc[name, "order"]

        if pd.isna(prev_order):
            prev_order = 9999

        if order < prev_order:
            self._df.loc[(self._df.order < prev_order) & (self._df.order >= order), "order"] += 1

        elif order > prev_order:
            self._df.loc[(self._df.order <= order) & (self._df.order > prev_order), "order"] -= 1

        self._df.loc[name, "order"] = order
        self._update_obs_times()

    def remove_target(self, name):
        """Remove target from obs plan. Equivalent to sending order to inf."""
        prev_order = self._df.loc[name, "order"]

        self._df.loc[self._df.order > prev_order, "order"] -= 1
        self._df.loc[name, ["order", "secz_i", "secz_f"]] = pd.NA
        self._df.loc[name, ["t_start", "t_end", "t_start_hidden", "t_end_hidden"]] = pd.NaT
        self._df.loc[name, "slew_s"] = pd.Timedelta(0.0, unit="m")

        self._update_obs_times()

    def _find_closest_object(self, coord):
        """Finds closest object to coordinate. Returns name."""
        min_slew = pd.Timedelta(1000.0, unit="m")
        best_name = None

        for obj in self._df.itertuples():
            if (obj.coord == coord) or pd.isna(obj.order):
                continue
            slew_time = self._slew_time(obj.coord, coord)
            if slew_time < min_slew:
                min_slew = slew_time
                best_name = obj.Index

        return best_name

    def add_target(self, name):
        """Add target to obsplan. Slots in to nearest object currently in plan."""
        coord = self._df.loc[name, "coord"]
        nearest_name = self._find_closest_object(coord)
        order = self._df.loc[nearest_name, "order"] + 1
        self._df.loc[self._df.order >= order, "order"] += 1
        self._df.loc[name, "order"] = order

        self._update_obs_times()

    def swap_targets(self, name, other):
        """Swap two targets in observing chart."""
        order1 = self._df.loc[name, "order"]
        order2 = self._df.loc[other, "order"]
        self._df.loc[name, "order"] = order2
        self._df.loc[other, "order"] = order1

        self._update_obs_times()

    def change_prio(self, name, prio):
        """Change the prio of a single target."""
        self._df.loc[name, "prio"] = prio

    def change_prio_program(self, program, prio):
        """Change the prio of all targets in a program."""
        self._df.loc[self._df.program == program, "prio"] = prio

    def _reset_unobserved_fields(self):
        """Reset fields on current observing info
        for targets not being observed.
        """
        # reset some values
        self._df.loc[
            self._df.order.isna(),
            [
                "t_start",
                "t_end",
                "t_start_hidden",
                "t_end_hidden",
                "secz_i",
                "secz_f",
                "eff_prio",
            ],
        ] = pd.NA
        self._df.loc[self._df.order.isna(), "slew_s"] = pd.Timedelta(0.0, unit="m")

    def generate_optimal_schedule(self, with_priorities=True):
        """Optimize observing program based on prio,
        transition times, and time in sky.
        """
        # reset
        self._df["cumul_program_time"] = 0
        self._df.loc[
            :,
            [
                "t_start_hidden",
                "t_end_hidden",
                "secz_i",
                "secz_f",
                "order",
                "eff_prio",
            ],
        ] = pd.NA
        self._df["slew_s"] = pd.Timedelta(minutes=0.0)

        self._df["current_time"] = self.start_time

        total_time = pd.Timedelta((self.end_time - self.start_time).value, unit="d")
        current_order = 0

        while (sum(self._df.order.isna()) > 0) and (self._df.current_time.iloc[0] < self.end_time):
            unscheduled_mask = self._df.order.isna()

            if current_order == 0:
                self._df.loc[unscheduled_mask, "slew_s"] = pd.Timedelta(minutes=0.0)

            else:
                prev_target = self._df.loc[self._df.order == current_order - 1, "coord"].item()
                self._df.loc[unscheduled_mask, "slew_s"] = [
                    self._slew_time(prev_target, coord) for coord in self._df.loc[unscheduled_mask, "coord"]
                ]

            too_early_mask = self._df.current_time < self._df.early_limit - self._df.slew_s
            too_late_mask = self._df.current_time > self._df.late_limit - self._df.slew_s - self._df.dur

            obs_time_left = self._df.late_limit - self._df.current_time

            if with_priorities:
                self._df["eff_prio"] = (
                    20.0 * self._df.slew_s + obs_time_left
                ) / total_time + 10 * self._df.prio
            else:
                self._df["eff_prio"] = (20.0 * self._df.slew_s + obs_time_left) / total_time

            self._df.loc[~unscheduled_mask | (too_early_mask | too_late_mask), "eff_prio"] = 9999.0

            best_target = self._df.eff_prio.idxmin()

            if self._df.loc[best_target, "eff_prio"] == 9999:
                self._df.current_time += pd.Timedelta(5.0, unit="m")  # no good target, add gap
                continue

            self._df.loc[best_target, "t_start_hidden"] = self._df.current_time[0]
            self._df.current_time += self._df.loc[best_target, "slew_s"]
            self._df.current_time += pd.Timedelta(self._df.loc[best_target, "dur"].value, unit="m")
            self._df.loc[best_target, "t_end_hidden"] = self._df.current_time[0]
            self._df.loc[best_target, "order"] = current_order
            current_order += 1

        self._reset_unobserved_fields()
        self._df.loc[:, 'order'] *= 10
        self._move_unused_time_to_middle()
        self._update_airmasses()

    def _write_obsplan_header(self, fn):
        moon_illumination = astroplan.moon_illumination(self.start_time)
        with open(fn, "w+") as f:
            f.write(f"MOON ILLUMINATION: {round(moon_illumination, 2)}\n{MANY_DASHES}")

        for twi_horizon in [10, 12, 14, 16]:
            ut_start, ut_end = calc_night_limits(observer=self.observer, date=self.date, horizon=twi_horizon)
            ut_start_formatted = ut_start.to_datetime().strftime("%m/%d %H:%M")
            ut_end_formatted = ut_end.to_datetime().strftime("%m/%d %H:%M")
            with open(fn, "a") as f:
                f.write(f"\nUT -{twi_horizon} deg twilight:\t{ut_start_formatted}\t|\t{ut_end_formatted}")

        with open(fn, "a") as f:
            f.write(f"\n{MANY_DASHES}")

        for twi_horizon in [10, 12, 14, 16]:
            ut_start, ut_end = calc_night_limits(observer=self.observer, date=self.date, horizon=twi_horizon)
            local_start = universal_to_local_time(ut_start, self.observer)
            local_end = universal_to_local_time(ut_end, self.observer)
            local_start_formatted = local_start.strftime("%m/%d %H:%M")
            local_end_formatted = local_end.strftime("%m/%d %H:%M")
            with open(fn, "a") as f:
                f.write(
                    f"\nLOCAL -{twi_horizon} deg twilight:\t{local_start_formatted}\t|\t{local_end_formatted}"
                )

        local_midnight = universal_to_local_time(self.midnight, self.observer).strftime("%m/%d %H:%M")
        with open(fn, "a") as f:
            f.write(f"\n{MANY_DASHES}\nLOCAL night MIDPOINT: {local_midnight}\n")
            f.write(OBS_PLAN_HEADER)


    def to_obsplan_file(self, fn):
        """Generate obsplan file for specific schedule."""
        self._write_obsplan_header(fn)
        column_width = 70
        num_mins = -61
        cumul_dur = 0

        for entry in self._df.loc[~self._df.order.isna()].itertuples():
            start_time = entry.t_start
            dur = entry.dur.value + entry.slew_s / 60.0

            name = entry.json
            pname = entry.program
            ra = round(entry.ra.value)
            dec = round(entry.dec.value)

            if cumul_dur > num_mins + 60:
                with open(fn, "a") as f:
                    f.write(f"\n\n!!! {start_time}")
                num_mins += 60

            if entry.early_limit == self.start_time:
                if entry.late_limit < self.end_time:
                    latest = entry.late_limit - dur * u.min
                    latest_local = universal_to_local_time(latest, self.observer).strftime("%H:%M")
                    with open(fn, "a") as f:
                        f.write((f"\n{pname} {name} [{ra} {dec}] ({round(dur)} min)").ljust(column_width))
                        f.write(f"------> SLEW BEFORE {latest_local}")
            elif entry.late_limit == self.end_time:
                if entry.early_limit > self.start_time:
                    earliest_local = universal_to_local_time(entry.early_limit, self.observer).strftime(
                        "%H:%M"
                    )
                    with open(fn, "a") as f:
                        f.write((f"\n{pname} {name} [{ra} {dec}] ({round(dur)} min)").ljust(column_width))
                        f.write(f"------> SLEW AFTER {earliest_local}")
            else:
                latest = entry.late_limit - dur * u.min
                latest_local = universal_to_local_time(latest, self.observer).strftime("%H:%M")
                earliest = entry.early_limit
                earliest_local = universal_to_local_time(earliest, self.observer).strftime("%H:%M")
                with open(fn, "a") as f:
                    f.write((f"\n{pname} {name} [{ra} {dec}] ({round(dur)} min)").ljust(column_width))
                    f.write(f"------> SLEW BETWEEN {earliest_local} and {latest_local}")

            cumul_dur += dur

    
    def load(self):
        """Load plan from directory (date)."""
        save_dir = f"../../decat_pointings/2025A/{self.date}"
        full_table_fn = os.path.join(save_dir, f"{self.date}_all_targets.csv")
        self._df = pd.read_csv(full_table_fn, index_col=0)
        self._df.name = 'name'

        # fix some column dtypes
        self._df.order = self._df.order.astype(pd.Int64Dtype())
        self._df.ra = self._df.ra.map(lambda x: float(x.split()[0]) * u.deg)
        self._df.dur = self._df.dur.map(lambda x: float(x.split()[0]) * u.min)
        self._df.dec = self._df.dec.map(lambda x: float(x.split()[0]) * u.deg)
        self._df.coord = self._df.apply(lambda x: SkyCoord(ra=x.ra, dec=x.dec), axis=1)
        self._df.late_limit = self._df.late_limit.map(lambda x: Time(x, format='iso'))
        self._df.early_limit = self._df.early_limit.map(lambda x: Time(x, format='iso'))

        # assume order may be updated, update times
        self._update_obs_times()


    def save(self):
        """Save in directory (date)."""
        save_dir = f"../../decat_pointings/2025A/{self.date}"
        obsplan_fn = os.path.join(save_dir, f"{self.date}_obsplan.txt")
        full_table_fn = os.path.join(save_dir, f"{self.date}_all_targets.csv")
        self._df.to_csv(full_table_fn)
        self.to_obsplan_file(obsplan_fn)
        print(f"Obsplan and table saved to {save_dir}")


    def save_airmass_plot(self):
        """Display the airmass plot for [name]."""
        save_dir = f"../../decat_pointings/2025A/{self.date}"
        fig = self.ac.plot_airmass(
            self._df, self.start_time, self.end_time,
            [self.sidereal_shift, self.observer_shift]
        )
        fig.savefig(os.path.join(save_dir, "airmass.png"), bbox_inches='tight', dpi=300)
        plt.close()


def main_loop():
    print("Welcome to the DECAT obsplan generator!")
    observer = ctio_location()
    today_date = universal_to_local_time(Time.now(), observer).strftime("%y%m%d")

    date = input(f"Enter date ({today_date}): ")
    if len(date) != 6:
        date = today_date

    directory = "../../decat_pointings/json_files/2025A"
    obsplan_directory = f"../../decat_pointings/2025A/{date}"
    schedule = generate_nightly_schedule(directory, date=date, observer=observer)

    full_table_fn = os.path.join(obsplan_directory, f"{date}_all_targets.csv")
    if os.path.exists(full_table_fn):
        print("Cached schedule found. Loading now...\n")
        schedule.load()
    else:
        print("Saved schedule not found. Generating new schedule...\n")
        schedule.generate_optimal_schedule(with_priorities=False)

    schedule.display()
    print("\n")

    while True:
        command = input("Enter command, 'help' for list of commands, or 'bye' to quit: ")
        if command.strip() == "bye":
            break
        if command.strip() == "help":
            print("Possible commands:\n")
            print("move [target_name] [order] : Move target [target_name] to the [order]th observing slot.")
            print(
                "swap [target_name] [other_target_name] : Swap observing slots for [target_name] and [other_target_name]."
            )
            print(
                "add [target_name] : Add [target_name] to observing queue without specifying slot. Target is inserted after closest target already being observed."
            )
            print(
                "remove [target_name] : Remove [target_name] from observing queue. Equivalent to setting 'order' to NA."
            )
            print(
                "optimize : Reset all manual assignments, generate schedule based on positions and slew transitions."
            )
            print(
                "optimize-prio : Reset all manual assignments, generate schedule based on PRIORITIES, positions and slew transitions."
            )
            continue

        parts = command.split()
        if len(parts) < 1:
            continue

        if parts[0] == "move":
            if len(parts) != 3:
                print("Invalid command: wrong number of inputs.")
                continue
            try:
                schedule.move_target(parts[1], int(parts[2]))
            except:
                print("Invalid command: invalid inputs.")
                continue

        elif parts[0] == "swap":
            if len(parts) != 3:
                print("Invalid command: wrong number of inputs.")
                continue
            try:
                schedule.swap_targets(parts[1], parts[2])
            except:
                print("Invalid command: invalid inputs.")
                continue

        elif parts[0] == "add":
            if len(parts) != 2:
                print("Invalid command: wrong number of inputs.")
                continue
            try:
                schedule.add_target(parts[1])
            except:
                print("Invalid command: invalid inputs.")
                continue

        elif parts[0] == "remove":
            if len(parts) != 2:
                print("Invalid command: wrong number of inputs.")
                continue
            try:
                schedule.remove_target(parts[1])
            except:
                print("Invalid command: invalid inputs.")
                continue

        elif parts[0] == "optimize":
            if len(parts) != 1:
                print("Invalid command: wrong number of inputs.")
                continue
            schedule.generate_optimal_schedule(with_priorities=False)

        elif parts[0] == "optimize-prio":
            if len(parts) != 1:
                print("Invalid command: wrong number of inputs.")
                continue
            schedule.generate_optimal_schedule(with_priorities=True)

        else:
            print("Invalid command! Type 'help' for options.")
            continue

        print("\n")
        schedule.display()
        print("\n")


if __name__ == "__main__":
    main_loop()
