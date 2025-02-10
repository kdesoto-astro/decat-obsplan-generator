# Where schedule and observing blocks are generated.
import os
from pathlib import Path
import pandas as pd
import warnings

import astropy.units as u
from astropy.table import Table
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroplan.scheduling import (
    Schedule, Scheduler, Transitioner,
    TransitionBlock
)
import astroplan
from astroplan.constraints import _get_altaz
from decat_obsplan_generator.airmass import AirmassCalculator
from decat_obsplan_generator.helpers import (
    ctio_location,
    read_json,
    merge_json_same_night,
    calc_night_limits,
    get_all_jsons_from_directory,
    local_to_universal_time,
    universal_to_local_time,
    MANY_DASHES,
    OBS_PLAN_HEADER,
    get_program_name,
    nice_time_formatting,
    nice_duration_formatting,
)
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
        date = current_datetime.strftime('%y%m%d')

    blocks = []
    night_start, night_end = calc_night_limits(date=date)
    airmass_calc = AirmassCalculator()
    airmass_calc.load_airmass_grid()

    all_ids = []

    for fn in get_all_jsons_from_directory(directory, date):
        names, jsons, propids, ras, decs, exptimes = read_json(fn)
        if jsons is None:
            print(f"Skipped: {fn}")
            continue
        ra, dec, dur = merge_json_same_night(ras, decs, exptimes)
        early_limit, late_limit = airmass_calc.observable_range(
            ra, dec, night_start, night_end
        )
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
            name += "_"

        block = {
            "name": name,
            "observer": observer,
            "json": jsons[0][:-5],
            "ra": ra,
            "dec": dec,
            "coord": SkyCoord(ra=ra, dec=dec),
            "dur": dur * u.min,
            "early_limit": early_limit2,
            "late_limit": late_limit2,
            "prio": 0, #TODO: UPDATE
            "propid": propids[0]
        }

        block['secz18_beg'] = universal_to_local_time(early_limit, observer).strftime("%H:%M")
        block['secz18_end'] = universal_to_local_time(late_limit, observer).strftime("%H:%M")
        block['secz2_beg'] = universal_to_local_time(early_limit2, observer).strftime("%H:%M")
        block['secz2_end'] = universal_to_local_time(late_limit2, observer).strftime("%H:%M")
        
        block['program'] = get_program_name(block['propid'])
            
        all_ids.append(name)
        blocks.append(block)

    schedule = DECATSchedule(night_start, night_end, blocks, observer=observer)
    return schedule


class DECATSchedule:
    """Modified schedule for DECAT, which includes total program
    allocated times.
    """

    def __init__(self, start_time, end_time, all_blocks, observer=None):
        """Adds program time dictionary."""
        self.start_time = start_time
        self.end_time = end_time

        if observer is None:
            self.observer = ctio_location()
        else:
            self.observer = observer

        program_time_fn = os.path.join(
            Path(__file__).parent.parent.parent.absolute(),
            "data/program_times.csv"
        )
        sched = pd.read_csv(program_time_fn)
        month = start_time.to_datetime().strftime("%b")
        self.program_times = sched.set_index('propid')[month].to_dict()
        self.program_times = {k: v * u.min for (k,v) in self.program_times.items()}

        self._df = pd.DataFrame.from_records(all_blocks)
        self._df.loc[:,[
            't_start', 't_end', 'secz_i', 'secz_f', 'slew_s',
            'order', 'eff_prio', 'current_time'
        ]] = pd.NA
        self._df.set_index('name', inplace=True)

        self._display_columns = [
            'json', 'program', 'propid', 'prio',
            'dur', 'ra', 'dec',
            'secz18_beg', 'secz18_end',
            'secz2_beg', 'secz2_end', 'order', 't_start',
            't_end', 'slew_s', 'secz_i', 'secz_f'
        ]

        self.ac = AirmassCalculator()
        self.ac.load_airmass_grid()

    
    def display(self):
        print(self._df[self._display_columns].to_string())

    
    def _slew_time(self, coord1, coord2):
        """Calculate slew time between two coordinates.
        """
        sep = coord1.separation(coord2).to(u.deg).value
        slew_rate = 5 / 11.0
        # readout is about 29 s, and slew time graph has y-intercept of ~29 s
        # So 29 s is added either way
        slew_time = sep / slew_rate + 29.

        # account for equatorial mount flip
        if (coord1.dec < -30.17 * u.deg) and (coord2.dec > -30.17 * u.deg):
            slew_time += 180.0
        elif (coord1.dec > -30.17 * u.deg) and (coord2.dec < -30.17 * u.deg):
            slew_time += 180.0

        return pd.Timedelta(slew_time, unit='s')


    def _update_obs_times(self):
        """Based on target order, calculate slew time
        and update start/end times."""
        max_order = self._df.order.dropna().max()

        for i in range(max_order + 1):
            if i == 0:
                self._df.loc[self._df.order == i, 't_start'] = self.start_time
                self._df.loc[self._df.order == i, 'slew_s'] = pd.Timedelta(0., unit='m')
                self._df.loc[self._df.order == i, 't_end'] = self.start_time + self._df.loc[self._df.order == i, 'dur']
            else:
                # calculate slew time
                prev_coord = self._df.loc[self._df.order == i - 1, 'coord'].item()
                curr_coord = self._df.loc[self._df.order == i, 'coord'].item()
                slew_time = self._slew_time(prev_coord, curr_coord)

                prev_t_end = self._df.loc[self._df.order == i - 1, 't_end'].item()
                self._df.loc[self._df.order == i, 't_start'] = prev_t_end
                self._df.loc[self._df.order == i, 'slew_s'] = slew_time
                self._df.loc[self._df.order == i, 't_end'] = prev_t_end + self._df.loc[self._df.order == i, 'dur'] + slew_time

        self._update_airmasses()


    def _update_airmasses(self):
        """Update airmasses of generated schedule.
        """
        self._df['secz_i'] = [
            round(self.ac.query(row.t_start, row.ra, row.dec)[0], 2) for row in self._df.itertuples()
        ]
        self._df['secz_f'] = [
            round(self.ac.query(row.t_end, row.ra, row.dec)[0], 2) for row in self._df.itertuples()
        ]

        #underground_mask = self._df.secz_i.isna() | self._df.secz_f.isna()
        daylight_mask = (self._df.t_end > self.end_time) | (self._df.t_start < self.start_time)
        not_visible_mask = daylight_mask #| underground_mask

        # formatting
        self._df.loc[not_visible_mask, ['t_start', 't_end']] = pd.NaT
        self._df.loc[not_visible_mask, ['secz_i', 'secz_f', 'order']] = pd.NA
        self._df.loc[not_visible_mask, 'slew_s'] = pd.Timedelta(0., unit='m')

        self._df.loc[~not_visible_mask, 't_start'] = self._df.loc[~not_visible_mask, 't_start'].apply(
            universal_to_local_time, args=(self.observer,)
        )
        self._df.loc[~not_visible_mask, 't_end'] = self._df.loc[~not_visible_mask, 't_end'].apply(
            universal_to_local_time, args=(self.observer,)
        )
        self._df.loc[~not_visible_mask, 't_start'] = self._df.loc[~not_visible_mask, 't_start'].apply(
            nice_time_formatting
        )
        self._df.loc[~not_visible_mask, 't_end'] = self._df.loc[~not_visible_mask, 't_end'].apply(
            nice_time_formatting
        )

        self._df.loc[~not_visible_mask, 'slew_s'] = self._df.loc[~not_visible_mask, 'slew_s'].apply(
            nice_duration_formatting
        )

        self._df.sort_values(by='order', inplace=True)



    def move_target(self, name, order):
        """Move target to spot 'order'. Move all targets that were before it
        one spot down."""
        prev_order = self._df.loc[name, "order"]

        if order < prev_order:
            self._df.loc[(self._df.order < prev_order) & (self._df.order >= order), "order"] += 1
        
        elif order > prev_order:
            self._df.loc[(self._df.order <= order) & (self._df.order > prev_order), "order"] -= 1

        self._df.loc[name, "order"] = order
        self._update_obs_times()


    def remove_target(self, name):
        """Remove target from obs plan. Equivalent to sending order to inf.
        """
        prev_order = self._df.loc[name, "order"]

        self._df.loc[self._df.order > prev_order, "order"] -= 1
        self._df.loc[name, ["order", "secz_i", "secz_f"]] = pd.NA
        self._df.loc[name, ["t_start", "t_end"]] = pd.NaT
        self._df.loc[name, "slew_s"] = pd.Timedelta(0., unit='m')

        self._update_obs_times()



    def _find_closest_object(self, coord):
        """Finds closest object to coordinate. Returns name.
        """
        min_slew = pd.Timedelta(1000., unit='m')
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
        """Add target to obsplan. Slots in to nearest object currently in plan.
        """
        coord = self._df.loc[name, "coord"]
        nearest_name = self._find_closest_object(coord)
        order = self._df.loc[nearest_name, "order"] + 1
        self._df.loc[self._df.order >= order, "order"] += 1
        self._df.loc[name, "order"] = order

        self._update_obs_times()


    def swap_targets(self, name, other):
        """Swap two targets in observing chart.
        """
        order1 = self._df.loc[name, "order"]
        order2 = self._df.loc[other, "order"]
        self._df.loc[name, 'order'] = order2
        self._df.loc[other, 'order'] = order1

        self._update_obs_times()

    
    def change_prio(self, name, prio):
        """Change the prio of a single target.
        """
        self._df.loc[name, "prio"] = prio


    def change_prio_program(self, program, prio):
        """Change the prio of all targets in a program.
        """
        self._df.loc[self._df.program == program, 'prio'] = prio


    def generate_optimal_schedule(self):
        """Optimize observing program based on prio,
        transition times, and time in sky.
        """
        # reset
        self._df['cumul_program_time'] = 0
        self._df.loc[:,[
            't_start', 't_end', 'secz_i',
            'secz_f', 'order', 'eff_prio',
        ]] = pd.NA
        self._df['slew_s'] = pd.Timedelta(minutes=0.0)

        self._df['current_time'] = self.start_time

        total_time = pd.Timedelta((self.end_time - self.start_time).value, unit='d')
        current_order = 0

        while (sum(self._df.order.isna()) > 0) and (self._df.current_time.iloc[0] < self.end_time):
            unscheduled_mask = self._df.order.isna()

            if current_order == 0:
                self._df.loc[unscheduled_mask, 'slew_s'] = pd.Timedelta(minutes=0.0)

            else:
                prev_target = self._df.loc[self._df.order == current_order - 1, 'coord'].item()
                self._df.loc[unscheduled_mask, 'slew_s'] = [
                    self._slew_time(prev_target, coord) for coord in self._df.loc[unscheduled_mask, 'coord']
                ]

            too_early_mask = self._df.current_time < self._df.early_limit - self._df.slew_s
            too_late_mask = self._df.current_time > self._df.late_limit - self._df.slew_s - self._df.dur
            
            obs_time_left = (self._df.late_limit - self._df.current_time)
            self._df['eff_prio'] = (20. * self._df.slew_s + obs_time_left) / total_time + 10 * self._df.prio
            self._df.loc[~unscheduled_mask | (too_early_mask | too_late_mask), 'eff_prio'] = 9999.

            best_target = self._df.eff_prio.idxmin()

            if self._df.loc[best_target, 'eff_prio'] == 9999:
                self._df.current_time += pd.Timedelta(5.0, unit='m')  # no good target, add gap
                continue
            
            self._df.loc[best_target, 't_start'] = self._df.current_time[0]
            self._df.current_time += self._df.loc[best_target, 'slew_s']
            self._df.current_time += pd.Timedelta(self._df.loc[best_target, 'dur'].value, unit='m')
            self._df.loc[best_target, 't_end'] = self._df.current_time[0]
            self._df.loc[best_target, 'order'] = current_order
            current_order += 1


        # reset some values
        self._df.loc[
            self._df.order.isna(),
            [
                't_start', 't_end', 'secz_i',
                'secz_f', 'eff_prio',
            ]

        ] = pd.NA
        self._df.loc[self._df.order.isna(), 'slew_s'] = pd.Timedelta(0., unit='m')

        self._update_airmasses()



    def _write_obsplan_header(self, fn):
        observer = ctio_location()
        current_datetime = (self.start_time - 6 * u.hour).to_datetime()
        date = current_datetime.strftime('%y%m%d')

        moon_illumination = astroplan.moon_illumination(self.start_time)
        with open(fn, "w+") as f:
            f.write(f"MOON ILLUMINATION: {round(moon_illumination, 2)}\n{MANY_DASHES}")

        for twi_horizon in [10, 12, 14, 16]:
            ut_start, ut_end = calc_night_limits(
                observer=observer,
                date=date,
                horizon=twi_horizon
            )
            ut_start_formatted = ut_start.to_datetime().strftime('%m/%d %H:%M')
            ut_end_formatted = ut_end.to_datetime().strftime('%m/%d %H:%M')
            with open(fn, "a") as f:
                f.write(f"\nUT -{twi_horizon} deg twilight:\t{ut_start_formatted}\t|\t{ut_end_formatted}")

        with open(fn, "a") as f:
            f.write(f"\n{MANY_DASHES}")

        for twi_horizon in [10, 12, 14, 16]:
            ut_start, ut_end = calc_night_limits(
                observer=observer,
                date=date,
                horizon=twi_horizon
            )
            local_start = universal_to_local_time(ut_start, observer)
            local_end = universal_to_local_time(ut_end, observer)
            local_start_formatted = local_start.strftime('%m/%d %H:%M')
            local_end_formatted = local_end.strftime('%m/%d %H:%M')
            with open(fn, "a") as f:
                f.write(f"\nLOCAL -{twi_horizon} deg twilight:\t{local_start_formatted}\t|\t{local_end_formatted}")

        date2 = f'20{date[:2]}-{date[2:4]}-{date[4:6]} 12:00:00' # noon
        t = local_to_universal_time(date2, observer)
        midnight = observer.midnight(t, which='next')
        local_midnight = universal_to_local_time(midnight, observer).strftime('%m/%d %H:%M')
        with open(fn, "a") as f:
            f.write(f"\n{MANY_DASHES}\nLOCAL night MIDPOINT: {local_midnight}\n")
            f.write(OBS_PLAN_HEADER)


    def to_obsplan_file(self, fn):
        """Generate obsplan file for specific schedule.
        """
        observer = ctio_location()
        self._write_obsplan_header(fn)
        column_width = 70
        num_mins = -61

        for i, slot in enumerate(self.slots):
            if not (slot.block and hasattr(slot.block, "name")):
                continue

            start_time = universal_to_local_time(slot.start, observer).strftime('%H:%M')
            dur = slot.dur.to(u.minute).value

            name = slot.block.name
            pname = self.program_names[slot.block.program]
            ra = round(slot.block.ra.value)
            dec = round(slot.block.dec.value)
            if (i > 0):
                prev = self.slots[i-1]
                # incorporate slew time
                if prev.block and not hasattr(prev.block, 'name'):
                    start_time = universal_to_local_time(self.slots[i-1].start, observer).strftime('%H:%M')
                    dur += slot.dur.to(u.minute).value

            if (slot.start - self.start_time).to(u.min).value > num_mins + 60:
                with open(fn, "a") as f:
                    f.write(f"\n\n!!! {start_time}")
                num_mins += 60

            if slot.block.earliest_time == self.start_time:
                if slot.block.latest_time < self.end_time:
                    latest = (slot.block.latest_time - dur * u.min)
                    latest_local = universal_to_local_time(latest, observer).strftime('%H:%M')
                    with open(fn, "a") as f:
                        f.write((f"\n{pname} {name} [{ra} {dec}] ({round(dur)} min)").ljust(column_width))
                        f.write(f"------> BEFORE {latest_local}")
            elif slot.block.latest_time == self.end_time:
                if slot.block.earliest_time > self.start_time:
                    earliest = slot.block.earliest_time
                    earliest_local = universal_to_local_time(earliest, observer).strftime('%H:%M')
                    with open(fn, "a") as f:
                        f.write((f"\n{pname} {name} [{ra} {dec}] ({round(dur)} min)").ljust(column_width))
                        f.write(f"------> AFTER {earliest_local}")
            else:
                latest = (slot.block.latest_time - dur * u.min)
                latest_local = universal_to_local_time(latest, observer).strftime('%H:%M')
                earliest = slot.block.earliest_time
                earliest_local = universal_to_local_time(earliest, observer).strftime('%H:%M')
                with open(fn, "a") as f:
                    f.write((f"\n{pname} {name} [{ra} {dec}] ({round(dur)} min)").ljust(column_width))
                    f.write(f"------> BETWEEN {earliest_local} and {latest_local}")


def main_loop():
    print("Welcome to the DECAT obsplan generator!")
    observer = ctio_location()
    today_date = universal_to_local_time(Time.now(), observer).strftime('%y%m%d')

    date = input(f"Enter date ({today_date}): ")
    if len(date) != 6:
        date = today_date

    directory = "../../data/json_files/2025A"
    schedule = generate_nightly_schedule(directory, date=date, observer=observer)
    schedule.generate_optimal_schedule()
    schedule.display()

    while True:
        command = input("Enter command (increase/decrease) and amount, 'help' for list of commands, or 'bye' to quit: ")
        if command.strip() == 'bye':
            break
        if command.strip() == 'help':
            print("Possible commands:\n")
            print("move [target_name] [order] : Move target [target_name] to the [order]th observing slot.")
            print("swap [target_name] [other_target_name] : Swap observing slots for [target_name] and [other_target_name].")
            print("add [target_name] : Add [target_name] to observing queue without specifying slot. Target is inserted after closest target already being observed.")
            print("remove [target_name] : Remove [target_name] from observing queue. Equivalent to setting 'order' to NA.")
            print("optimize : Reset all manual assignments, generate schedule based on positions and slew transitions.")
            continue

        parts = command.split()
        if len(parts) < 1:
            continue

        if parts[0] == 'move':
            if len(parts) != 3:
                print("Invalid command: wrong number of inputs.")
                continue
            try:
                schedule.move_target(parts[1], int(parts[2]))
            except:
                print("Invalid command: invalid inputs.")
                continue

        elif parts[0] == 'swap':
            if len(parts) != 3:
                print("Invalid command: wrong number of inputs.")
                continue
            try:
                schedule.swap_targets(parts[1], parts[2])
            except:
                print("Invalid command: invalid inputs.")
                continue
        
        elif parts[0] == 'add':
            if len(parts) != 2:
                print("Invalid command: wrong number of inputs.")
                continue
            try:
                schedule.add_target(parts[1])
            except:
                print("Invalid command: invalid inputs.")
                continue

        elif parts[0] == 'remove':
            if len(parts) != 2:
                print("Invalid command: wrong number of inputs.")
                continue
            try:
                schedule.remove_target(parts[1])
            except:
                print("Invalid command: invalid inputs.")
                continue

        elif parts[0] == 'optimize':
            if len(parts) != 1:
                print("Invalid command: wrong number of inputs.")
                continue
            schedule.generate_optimal_schedule()

        else:
            raise ValueError("Invalid command! Type 'help' for options.")

        schedule.display()

if __name__ == "__main__":
    main_loop()