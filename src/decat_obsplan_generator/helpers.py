import astropy.units as u
from astroplan import Observer


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
