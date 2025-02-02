import astropy.units as u
from astroplan import Observer
import json
import numpy as np


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


def read_json(json_fn):
    """Taken from slewtimes.py file in decat_pointings."""
    propids,fns,ras,decs,exptimes = [],[],[],[],[]
    with open(json_fn) as json_file: 
        pointings = json.load(json_file)
        lastra = np.nan
        lastdec = np.nan
        for pointing in pointings:
            try:
                raname = np.array(list(pointing.keys()))[np.array([k.lower() == 'ra' for k in pointing.keys()])][0]
                ra = pointing[raname]
                decname = np.array(list(pointing.keys()))[np.array([k.lower() == 'dec' for k in pointing.keys()])][0]
                dec = pointing[decname]
            except:
                print('ra/dec not found, using previous exposure ra/dec')
                ra = lastra
                dec = lastdec
            if ":" in str(ra):
                c = SkyCoord(str(ra)+' '+str(dec), unit=(u.hourangle, u.deg))
                ra = c.ra.degree
                dec = c.dec.degree
            
            propids.append(pointing['propid'])
            ras.append(float(ra))
            decs.append(float(dec))
            fns.append(json_fn.split('/')[-1].split('.json')[0])
            lastra = ra
            lastdec = dec
            expname = np.array(list(pointing.keys()))[np.array([k.lower() == 'exptime' for k in pointing.keys()])][0]
            exptimes.append(float(pointing[expname]))

    return fns,propids,ras,decs,exptimes
