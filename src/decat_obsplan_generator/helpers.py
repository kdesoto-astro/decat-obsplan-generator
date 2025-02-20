import os
import glob
import json
from pathlib import Path
from datetime import datetime
import pytz
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np
from astroplan import Observer

MANY_DASHES = '-' * 51

OBS_PLAN_HEADER = """
*********************************************
*** ALL TIMES BELOW IN LOCAL CHILEAN TIME!*** 
*** Note: Local Chilean time is now EDT+2 ***
*********************************************

*********************************************************
*** WHAT TO DO IF YOU FALL BEHIND SCHEDULE:
*** This can happen for various reasons (technical issues, scheduling mistakes etc).
*** Look for targets that are marked OPTIONAL, and skip when you get to them.
*** Stay within targets' time windows if you adjust schedule!!!
*********************************************************

**** PLEASE READ ABOVE ^^^^^^^^ BEFORE OBSERVING
"""


def get_program_name(propid):
    program_time_fn = os.path.join(
        Path(__file__).parent.parent.parent.absolute(),
        "data/program_times.csv"
    )
    sched = pd.read_csv(program_time_fn, index_col=0)
    return sched.loc[propid, 'name']

def ctio_location(
    temperature=5.0 * u.deg_C,
    pressure=780.0 * u.mbar,
    relative_humidity=0.5,
):
    """Generate Observer (which inherits from EarthLocation)
    for CTIO.
    """
    return Observer.at_site(
        "CTIO",
        timezone="Etc/GMT+3",
        pressure=pressure,
        temperature=temperature,
        relative_humidity=relative_humidity,
    )

def local_to_universal_time(local_date_time_str, observer):
    local_date_time = datetime.strptime(local_date_time_str, '%Y-%m-%d %H:%M:%S')
    # Localize this datetime object to the given timezone
    local_time = observer.timezone.localize(local_date_time)
    # Convert to UT using astropy's Time
    ut_time = Time(local_time).utc
    return ut_time

def universal_to_local_time(ut_time, observer):
    try:
        local_time = ut_time.to_datetime(timezone=pytz.utc).astimezone(observer.timezone)
        return local_time
    except:
        return None

def nice_time_formatting(local_time):
    if pd.isna(local_time):
        return local_time
    return local_time.strftime("%H:%M")

def nice_duration_formatting(duration):
    try:
        return duration.seconds
    except:
        return round(duration)

def calc_night_limits(observer=None, date=None, horizon=14):
    """Calculate beginning and end of night.
    If date is None, assume for tonight.
    """
    if observer is None:
        observer = ctio_location()
    # date in YYYYMMDD format
    if date is None:
        t = Time.now()
    else:
        date = f'20{date[:2]}-{date[2:4]}-{date[4:6]} 12:00:00' # noon
        t = local_to_universal_time(date, observer)

    twi = observer.tonight(t, horizon=-horizon*u.deg)
    twi[0].format = 'iso'
    twi[1].format = 'iso'
    return twi[0], twi[1]


def read_json(json_fn):
    """Taken from slewtimes.py file in decat_pointings."""
    propids, fns, ras, decs, exptimes = [], [], [], [], []
    names = []
    with open(json_fn) as json_file:
        try:
            pointings = json.load(json_file)
        except:
            return None, None, None, None, None, None
        lastra = np.nan
        lastdec = np.nan
        for pointing in pointings:
            ra_key = None
            dec_key = None
            deltara_key = None
            deltadec_key = None

            for k in pointing.keys():
                if k.lower() == 'ra':
                    ra_key = k
                elif k.lower() == 'dec':
                    dec_key = k
                elif k.lower() == 'deltara':
                    deltara_key = k
                elif k.lower() == 'deltadec':
                    deltadec_key = k
            
            if (ra_key is not None) and (dec_key is not None):
                ra = pointing[ra_key]
                dec = pointing[dec_key]
            
            elif (deltara_key is not None) and (deltadec_key is not None):
                deltara = pointing[deltara_key]
                deltadec = pointing[deltadec_key]
                if deltara[0] == "E":
                    ra = lastra + float(deltara[1:])
                elif deltara[0] == "W":
                    ra = lastra - float(deltara[1:])
                else:
                    raise ValueError("Invalid delta-RA!")
                
                if deltadec[0] == "N":
                    dec = lastdec + float(deltadec[1:])
                elif deltadec[0] == "S":
                    dec = lastdec - float(deltadec[1:])
                else:
                    raise ValueError("Invalid delta-dec!")
                    
            else:
                print("No ra/dec found: using previous ra/dec.")
                ra = lastra
                dec = lastdec

            if ":" in str(ra):
                c = SkyCoord(str(ra) + " " + str(dec), unit=(u.hourangle, u.deg))
                ra = c.ra.degree
                dec = c.dec.degree

            propids.append(pointing["propid"])
            ras.append(float(ra))
            decs.append(float(dec))
            fns.append(json_fn.split("/")[-1])
            names.append(pointing["object"])
            lastra = float(ra)
            lastdec = float(dec)
            expname = np.array(list(pointing.keys()))[
                np.array([k.lower() == "exptime" for k in pointing.keys()])
            ][0]
            exptimes.append(float(pointing[expname]))

    return names, fns, propids, np.round(ras, 2) * u.deg, np.round(decs, 2) * u.deg, exptimes


def merge_json_same_night(ras, decs, exptimes):
    """
    Calculate the total duration including
    slew + readout times, within a single json's
    coordinates.
    """
    avg_ra = np.mean(ras.value) * u.deg
    avg_dec = np.mean(decs.value) * u.deg

    total_dur = exptimes[0]

    for i in range(1, len(ras)):
        delra = (ras[i] - ras[i-1]) * np.cos((decs[i] + decs[i-1])/2.)
        deg_separation = np.sqrt((delra)**2 + (decs[i] - decs[i-1])**2).to(u.deg).value
        total_dur += 220./100.*deg_separation + 29.
        total_dur += exptimes[1]

    return np.round(avg_ra, 1), np.round(avg_dec, 1), np.round(total_dur / 60., 1)


def get_all_jsons_from_directory(directory, date):
    """From directory and date, extract all jsons
    from directory associated with that date.
    """
    base_jsons = glob.glob(
        os.path.join(directory, "*", date, "*.json")
    )
    prios = []
    for json in base_jsons:
        base_name = json.split("/")[-1][:-5]
        prefix = base_name.split("_")[-1]
        if prefix[0] == "P":
            prios.append(int(prefix[1]) - 1)
        else:
            prios.append(0)

    # get all jsons in lower-prio subdirectories
    low_prio_jsons = glob.glob(
        os.path.join(directory, "*", date, "*", "*.json")
    )
    for json in low_prio_jsons:
        prios.append(int(json.split("/")[-2]))

    return base_jsons + low_prio_jsons, prios


