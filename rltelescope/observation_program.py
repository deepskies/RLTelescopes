"""
An adaptation of the observation program written by Eric Neilson in Astrotact:  https://github.com/ehneilsen/astrotact.git

Designed to step through the night and calculate variables for a given point site/calculation combo.

The program is written so that you can calculate the result of an action before taking it,
so to take an action call "update_observation(action)".
This updates the "obvervation" parameter of the class.
To just calculate the expected outcome of a action call "calculate_obversation"

While not recommended, calling the internal "_observation",
updates the observation state based on the currently stored action.

"""
import os.path

import pandas as pd
import configparser
import ast
import numpy as np
import numexpr
import random

from astropy.coordinates import get_sun, get_moon
import astroplan
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from skybright import skybright

import warnings

warnings.filterwarnings("ignore")


class ObservationProgram:
    def __init__(self, config_path, duration):

        self.angle_units = u.deg

        self.config = configparser.ConfigParser()
        assert os.path.exists(config_path)
        self.config.read(config_path)
        self.duration = duration * u.second
        self.observatory = self.init_observatory()
        self.slew_rate_deg_sec = self.init_slew()
        self.calc_sky = skybright.MoonSkyModel(self.config)
        self.set_optics()

        self.seeing = self.config.getfloat("weather", "seeing")
        self.clouds = self.config.getfloat("weather", "cloud_extinction")
        self.band = "g"
        # self.reset()

    def set_optics(self):
        self.optics_fwhm = self.config.getfloat("optics", "fwhm")

        try:
            self.band_wavelength = ast.literal_eval(
                self.config.get("bands", "wavelengths")
            )
        except KeyError:
            self.band_wavelength = {
                "u": 380.0,
                "g": 475.0,
                "r": 635.0,
                "i": 775.0,
                "z": 925.0,
                "Y": 1000.0,
            }

        self.wait_time_seconds = 300 * u.second

        try:
            self.filter_change_time_seconds = self.config.getfloat(
                "bands", "filter_change_rate"
            )
        except KeyError:
            self.filter_change_time_seconds = 0.0 * u.second

    def _action(self):
        return {
            "mjd": self.mjd,
            "end_mjd": self.end_mjd,
            "decl": self.decl,
            "ra": self.ra,
            "band": self.band,
            "exposure_time_days": self.exposure_time_days,
        }

    def reset(self):
        self.start_time, self.end_time = self.init_time_start()
        self.mjd = self.start_time
        self.decl = np.asarray([0])
        self.ra = np.asarray([0])
        self.band = "g"
        self.exposure_time_days = 300 / (60 * 60 * 24) * u.day
        self.end_mjd = self.start_time + self.exposure_time_days

        self.action = self._action()
        self.observation = self._observation()

    def init_observatory(self):
        latitude = (
            self.config.getfloat("Observatory Position", "latitude") * self.angle_units
        )
        longitude = (
            self.config.getfloat("Observatory Position", "longitude") * self.angle_units
        )
        elevation = self.config.getfloat("Observatory Position", "elevation") * u.m

        return astroplan.Observer(
            longitude=longitude, latitude=latitude, elevation=elevation
        )

    def init_time_start(self):
        "Sets the start and end time for an obversation period, starting at sunset. Units of days from mjd"
        # TODO Check for moon !!
        year = random.choice([i + 10 for i in range(0, 14)])
        month = random.choice([i + 1 for i in range(11)])
        day = random.choice([i + 1 for i in range(28)])
        date = f"20{year}-{str(month).zfill(2)}-{str(day).zfill(2)}T01:00:00Z"

        time = Time(date, format="isot")
        sunset = self.observatory.sun_set_time(time).mjd * u.day
        end_time = sunset + self.duration

        # if sunset.isinstance(np.ma.core.MaskedArray):
        #     sunset, end_time = self.init_time_start()

        return sunset, end_time

    def invalid_action(self, observation):
        "Verify an action is valid under the given constraints"
        RIGHT_ANGLE = (90 * u.deg).to_value(self.angle_units) * self.angle_units
        radians = self.angle_units.to(u.radian)

        airmass_limit = self.config.getfloat("constraints", "airmass_limit")
        cos_zd_limit = 1.0011 / airmass_limit - 0.0011 * airmass_limit

        # From the spherical cosine formula
        site_lat = (
            self.config.getfloat("Observatory Position", "latitude") * self.angle_units
        )

        # TODO Fix radian conversion
        cos_lat = np.cos(site_lat * radians)
        sin_lat = np.sin(site_lat * radians)
        cos_dec = np.cos(observation["decl"] * radians)
        sin_dec = np.sin(observation["decl"] * radians)
        cos_ha_limit = (cos_zd_limit - sin_dec * sin_lat) / (cos_dec * cos_lat)

        max_sun_alt = (
            self.config.getfloat("constraints", "max_sun_alt") * self.angle_units
        )
        cos_sun_zd_limit = np.cos((RIGHT_ANGLE - max_sun_alt) * radians)
        cos_sun_dec = np.cos(observation["sun_decl"] * radians)
        sin_sun_dec = np.sin(observation["sun_decl"] * radians)
        cos_sun_ha_limit = (cos_sun_zd_limit - sin_sun_dec * sin_lat) / (
            cos_sun_dec * cos_lat
        )

        # Assuming a 300 second between observations
        start_mjd = observation["mjd"] - 0.003472222

        # Airmass limits
        ha_change = 2 * np.pi * (start_mjd - observation["mjd"]) * 24 / 23.9344696
        ha_change = ha_change * radians
        ha = observation["ha"] + ha_change
        in_airmass_limit = np.cos(ha) > cos_ha_limit

        # Moon angle
        min_moon_angle = self.config.getfloat("constraints", "min_moon_angle")
        in_moon_limit = observation["moon_angle"] > min_moon_angle

        # Solar ZD.
        # Ignore Sun's motion relative to ICRS during the exposure
        sun_ha = observation["sun_ha"] * radians + ha_change
        in_sun_limit = np.cos(sun_ha) < cos_sun_ha_limit

        invalid = np.logical_not(
            np.logical_and(
                np.logical_and(in_airmass_limit, in_sun_limit), in_moon_limit
            )
        )
        return invalid

    def init_slew(self):
        "Sets the slew rate for the telescope, in angle/sec"
        slew_expr = (
            self.config.getfloat("slew", "slew_expr") * self.angle_units / u.second
        )
        return slew_expr

    @staticmethod
    def calc_airmass(hzcrds=None, zd=None):
        "Airmass at the given hz and zd coordinates"
        if hzcrds is not None:
            cos_zd = np.cos(np.radians(90) - hzcrds.alt.rad)
        else:
            cos_zd = np.cos(np.radians(zd))

        a = numexpr.evaluate("462.46 + 2.8121/(cos_zd**2 + 0.22*cos_zd + 0.01)")
        airmass = numexpr.evaluate("sqrt((a*cos_zd)**2 + 2*a + 1) - a * cos_zd")
        airmass[hzcrds.alt.rad < 0] = np.nan
        return airmass

    def current_coord(self, ra, decl):
        radians = self.angle_units.to(u.radian)
        return SkyCoord(ra=ra * radians * u.radian, dec=decl * radians * u.radian)

    def trans_to_altaz(self, mjd, coord):
        alt_az = self.observatory.altaz(time=mjd, target=coord)
        return alt_az

    def calculate_obversation(self, action):
        RIGHT_ANGLE = (90 * u.deg).to_value(self.angle_units)
        radians = self.angle_units.to(u.radian)

        try:
            time = Time(action["mjd"], format="mjd")

        except u.core.UnitConversionError:
            time = Time(action["mjd"] * u.day, format="mjd")

        exposure = {}
        exposure["seeing"] = self.seeing
        exposure["clouds"] = self.clouds
        try:
            exposure["lst"] = self.observatory.local_sidereal_time(
                time, "mean"
            ).to_value(self.angle_units)
        except TypeError:
            exposure["lst"] = self.observation["lst"].to_value(self.angle_units)

        current_coords = self.current_coord(action["ra"], action["decl"])
        hzcrds = self.trans_to_altaz(time, current_coords)
        exposure["az"] = hzcrds.az.to_value(self.angle_units)
        exposure["alt"] = hzcrds.alt.to_value(self.angle_units)
        exposure["zd"] = RIGHT_ANGLE - exposure["alt"]
        exposure["ha"] = exposure["lst"] - action["ra"]
        exposure["airmass"] = ObservationProgram.calc_airmass(hzcrds)

        # Sun coordinates
        sun_crds = get_sun(time)
        exposure["sun_ra"] = sun_crds.ra.to_value(self.angle_units)
        exposure["sun_decl"] = sun_crds.dec.to_value(self.angle_units)
        sun_hzcrds = self.trans_to_altaz(time, sun_crds)
        exposure["sun_az"] = sun_hzcrds.az.to_value(self.angle_units)
        exposure["sun_alt"] = sun_hzcrds.alt.to_value(self.angle_units)
        exposure["sun_zd"] = RIGHT_ANGLE - exposure["sun_alt"]
        exposure["sun_ha"] = exposure["lst"] - exposure["sun_ra"]

        # Moon coordinates
        site_location = self.observatory.location
        moon_crds = get_moon(location=site_location, time=time)
        exposure["moon_ra"] = moon_crds.ra.to_value(self.angle_units)
        exposure["moon_decl"] = moon_crds.dec.to_value(self.angle_units)
        moon_hzcrds = self.observatory.moon_altaz(time)
        exposure["moon_az"] = moon_hzcrds.az.to_value(self.angle_units)
        exposure["moon_alt"] = moon_hzcrds.alt.to_value(self.angle_units)
        exposure["moon_zd"] = RIGHT_ANGLE - exposure["moon_alt"]
        exposure["moon_ha"] = exposure["lst"] - exposure["moon_ra"]
        exposure["moon_airmass"] = ObservationProgram.calc_airmass(moon_hzcrds)

        # Moon phase
        exposure["moon_phase"] = astroplan.moon.moon_phase_angle(time)
        exposure["moon_illu"] = self.observatory.moon_illumination(time)
        # Moon brightness
        moon_elongation = moon_crds.separation(sun_crds)
        alpha = 180.0 - moon_elongation.deg
        # Allen's _Astrophysical Quantities_, 3rd ed., p. 144
        exposure["moon_Vmag"] = -12.73 + 0.026 * np.abs(alpha) + 4e-9 * (alpha**4)

        exposure["moon_angle"] = moon_crds.separation(current_coords).to_value(
            self.angle_units
        )

        exposure["sky_mag"] = self.calc_sky(
            action["mjd"],
            action["ra"],
            action["decl"],
            np.asarray([self.band for _ in range(len(action["ra"]))]),
            moon_crds=moon_crds,
            moon_elongation=moon_elongation.deg,
            sun_crds=sun_crds,
        )

        m0 = self.calc_sky.m_zen[self.band]

        nu = 10 ** (-1 * self.clouds / 2.5)

        pt_seeing = self.seeing * exposure["airmass"] ** 0.6
        fwhm500 = np.sqrt(pt_seeing**2 + self.optics_fwhm**2)

        wavelength = self.band_wavelength[self.band]
        band_seeing = pt_seeing * (500.0 / wavelength) ** 0.2
        fwhm = np.sqrt(band_seeing**2 + self.optics_fwhm**2)
        exposure["fwhm"] = fwhm

        exposure["tau"] = ((nu * (0.9 / fwhm500)) ** 2) * (
            10 ** ((exposure["sky_mag"] - m0) / 2.5)
        )

        # exposure["tau"] = 0.0 if ~np.isfinite(exposure["tau"]) else exposure["tau"]

        exposure["teff"] = exposure["tau"] * action["exposure_time_days"]

        for key in action:
            if key not in ["band"]:
                exposure[key] = action[key]

        # exposure["slew"] = self.calculate_slew(current_coords, self.band)
        # exposure = pd.DataFrame(exposure, index=[0]).fillna(0).to_dict("records")[0]

        exposure["invalid"] = self.invalid_action(exposure)
        return exposure

    def calculate_slew(self, new_coords, band):
        original_coords = self.current_coord(self.ra, self.decl)

        coord_sep = original_coords.separation(new_coords)
        slew_time_seconds = self.slew_rate_deg_sec * coord_sep
        slew_time_days = slew_time_seconds.to_value(u.day)  # Convert to days
        return slew_time_days

    def _observation(self):
        exposure = self.calculate_obversation(self.action)
        return exposure

    def update_mjd(self, ra, decl, band):

        if not self.check_nighttime():
            self.advance_to_nighttime()

        original_coords = self.current_coord(self.ra, self.decl)

        if (ra is None) and (decl is None):
            updated_coords = original_coords

        elif (ra is not None) and (decl is None):
            updated_coords = SkyCoord(ra=ra * u.degree, dec=self.decl * u.degree)
        elif (ra is None) and (decl is not None):
            updated_coords = SkyCoord(ra=self.ra * u.degree, dec=decl * u.degree)
        else:
            updated_coords = SkyCoord(ra=ra * u.degree, dec=decl * u.degree)

        slew_time_days = self.calculate_slew(updated_coords, band)

        self.mjd = self.end_mjd + slew_time_days
        self.end_mjd = self.mjd + self.exposure_time_days

    def check_nighttime(self):
        time = Time(self.mjd * u.day, format="mjd")
        is_night = self.observatory.is_night(time)
        return is_night

    def advance_to_nighttime(self):
        time = Time(self.mjd * u.day, format="mjd")
        self.mjd = self.observatory.sun_set_time(time, which="next").mjd
        self.end_mjd = self.mjd + self.exposure_time_days

    def update_observation(
        self, ra=None, decl=None, band=None, exposure_time_days=None, reward=None
    ):
        # Updates the observation based on input. Any parameters not given
        # are held constant

        self.update_mjd(ra, decl, band)

        self.ra = ra if ra is not None else self.ra
        self.decl = decl if decl is not None else self.decl
        self.band = band if band is not None else self.band
        self.exposure_time_days = (
            exposure_time_days
            if exposure_time_days is not None
            else self.exposure_time_days
        )

        self.action = self._action()
        self.observation = self._observation()


if __name__ == "__main__":
    obs_config_path = os.path.abspath("train_configs" "/default_obsprog.conf")
    ObservationProgram(obs_config_path, duration=1)
