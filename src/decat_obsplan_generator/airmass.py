# Stores all airmass calculations
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import matplotlib.dates as mdates

from .helpers import ctio_location

class AirmassCalculator:
    """Class that performs airmass calculations and caches
    for later access."""

    def __init__(self, location=None):
        """Initializes location airmass is calculated from.
        """
        if location is None:
            self._loc = ctio_location()
        else:
            self._loc = location

        pref_a = self._loc.pressure.value
        pref_b = self._loc.temperature.value
        pref_c = self._loc.relative_humidity
        file_prefix = f"{pref_a}_{pref_b}_{pref_c}"
        cache_dir = os.path.join(
            Path(__file__).parent.parent.parent.absolute(),
            f"data/airmass_grids"
        )
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_fn = os.path.join(cache_dir, f"{file_prefix}.feather")

        # hour angle / declination grid
        # here we only go to HA = 5:15:00 because that's where DECam cuts off
        ha_deg_range = np.linspace(0., 78.75, num = 1_000) # symmetric so only positives
        dec_range = np.linspace(-90.0, 40.0, num = 1_000) # above 40deg not visible
        _ha_grid, _dec_grid = np.meshgrid(ha_deg_range, dec_range)
        self._ha_grid = _ha_grid.ravel()
        self._dec_grid = _dec_grid.ravel()


    def calculate_airmass(self, times, coordinate: SkyCoord):
        """Return airmass value at specific times
        for single coordinate."""
        return self._loc.altaz(times, target=coordinate).secz


    def generate_airmass_grid(self):
        """From grid of hour angles and declinations, generate
        """
        dummy_time = Time.now() # just to convert from HA to RA and back

        ra_values = (self._loc.local_sidereal_time(dummy_time).degree - self._ha_grid) % 360
        sky_coords = SkyCoord(ra=ra_values * u.deg, dec=self._dec_grid * u.deg, frame='icrs')

        # Convert to AltAz frame (to then retrieve airmass)
        airmasses = self._loc.altaz(time=dummy_time, target=sky_coords).secz
        airmasses[airmasses < 0.0] = np.nan # remove invalid airmasses

        self._cache = pd.DataFrame(
            np.array([self._ha_grid, self._dec_grid, airmasses]).T,
            columns=['ha', 'dec', 'airmass']
        )
        self._cache.to_feather(self._cache_fn)


    def load_airmass_grid(self):
        """Load and return airmass grid.
        """
        if not os.path.exists(self._cache_fn):
            raise FileNotFoundError(
                "Cache not yet created! Please run self.generate_airmass_grid() first."
            )
        self._cache = pd.read_feather(self._cache_fn)
        return self._cache.copy()


    def query_airmass(self, ra, dec, times):
        """From a RA/dec and set of times,
        retrieve set of airmasses from cache.
        """
        sidereal_degs = self._loc.local_sidereal_time(times).degree
        hour_angles = sidereal_degs - ra.degree

        return self._query_from_ha_dec(hour_angles, dec.degree)


    def _query_from_ha_dec(self, ha, dec):
        """Query cache from hour angle(s) and SINGLE declination.
        """
        if not np.isscalar(dec):
            raise TypeError("dec must be a scalar")

        if dec >= 40.0: # out of bounds
            return np.nan

        # take absolute values, airmasses symmetric around HA
        abs_ha = np.atleast_1d(np.abs(ha))
        oob_mask = (abs_ha >= 78.75) # out of bounds for DECam

        # 0.1 deg dec mask, grid is dense enough for this
        dec_df = self._cache.loc[(self._cache.dec - dec).abs() < 0.1]

        airmasses = []
        for hour_angle in abs_ha[~oob_mask]:
            candidate_df = dec_df.loc[(dec_df.ha - hour_angle).abs() < 0.1]
            sep_sq = self._separation_squared(
                candidate_df.ha, candidate_df.dec,
                hour_angle, dec
            )
            airmasses.append(candidate_df.loc[sep_sq.idxmin(), 'airmass'])

        airmasses_full = np.nan * np.ones(len(abs_ha))
        airmasses_full[~oob_mask] = airmasses

        return airmasses_full
            

    def _separation_squared(self, ra_arr, dec_arr, ra2, dec2):
        """Helper function to calculate separation
        directly from floats in degrees. Assumes
        small angle approximation.
        """
        return ((ra_arr - ra2)*np.cos((dec_arr+dec2) * np.pi/360.))**2+(dec_arr - dec2)**2


    def plot_airmass(self, fig, ax, ra, dec, times=None, **plot_kwargs):
        """Add airmass plot to ax, assuming
        target is at RA/Dec. If times is None, plot
        across entire 24 hours from now.
        """
        if times is None:
            start_time = Time.now()
            delta_times = np.linspace(0, 24.0, num=500) * u.hour
            time_arr = start_time + delta_times
        else:
            time_arr = np.atleast_1d(times)

        hour_angles = self._loc.local_sidereal_time(time_arr).degree - ra.to(u.deg).value
        airmasses = self._query_from_ha_dec(hour_angles, dec.to(u.deg).value)

        ax.plot(time_arr.to_datetime(), airmasses, **plot_kwargs)

        # Formatting x-axis ticks to the desired format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.invert_yaxis()
        ax.set_ylabel("Airmass")
        ax.axhline(1.8, linestyle='dashed', color='grey')

        # Auto-format the x-axis for better readability
        fig.autofmt_xdate()

        return fig, ax











    