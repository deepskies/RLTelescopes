"""
Testing relating to the observation program itself.
Checks things like units, and validity of variable groups
"""

import pytest
from rltelescope.observation_program import ObservationProgram


def default_setup():
    default_config = ""
    default_schedule_duration = 1
    return ObservationProgram(default_config, default_schedule_duration)


def test_start_time():
    "Start time and end time are the observation period apart at all times"
    obsprog = default_setup()

    start_time_uct_mjd = ""
    end_time_uct_mjd = ""

    assert start_time_uct_mjd == obsprog.start_time
    assert end_time_uct_mjd == obsprog.end_mjd
    assert start_time_uct_mjd == end_time_uct_mjd - obsprog.exposure_time_days


def test_vertical_valid_observation():
    "A straight up observation is valid"
    obsprog = default_setup()

    obsprog.mjd = ""
    observation = obsprog.update_observation(ra="", decl="", band="").observation
    validity = obsprog.invalid_action(observation)

    assert validity is False


def test_horizonal_invalid_observation():
    "An invalid observation points at the horizon"
    obsprog = default_setup()

    obsprog.mjd = ""
    observation = obsprog.update_observation(ra="", decl="", band="").observation
    validity = obsprog.invalid_action(observation)

    assert validity is True


def test_vertical_invalid_observation():
    "A straight down observation is invalid"
    obsprog = default_setup()

    obsprog.mjd = ""
    observation = obsprog.update_observation(ra="", decl="", band="").observation
    validity = obsprog.invalid_action(observation)

    assert validity is True


def test_day_invalid_observation():
    "An observation in the middle of the day is invalid"
    obsprog = default_setup()

    obsprog.mjd = ""
    observation = obsprog.update_observation(ra="", decl="", band="").observation
    validity = obsprog.invalid_action(observation)

    assert validity is True


def test_mjd_conversion():
    "The time update should set the previous start date to the end date"

    obsprog = default_setup()
    start_time = obsprog.mjd
    original_end_time = obsprog.end_mjd
    init_mjd = obsprog.start_time

    obsprog.update_mjd(ra=obsprog.ra, decl=obsprog.decl, band=obsprog.band)

    assert start_time == init_mjd
    assert obsprog.mjd == original_end_time


def test_sky_coordinate_conversion():
    "Sky Coordinate Object should have the same returned ra and decl when transformed back to degrees as the passed input"
    obsprog = default_setup()

    ra = ""
    decl = ""
    skycoord = obsprog.current_coord(ra, decl)

    assert ra == skycoord.ra
    assert decl == skycoord.decl


def test_alt_az_conversion():
    "Samsies as before"
    obsprog = default_setup()

    ra = ""
    decl = ""
    mjd = ""
    current_coords = obsprog.current_coord(ra, decl)
    alt_az = obsprog.trans_to_altaz(mjd, current_coords)
    trans_ra, trans_decl = alt_az.ra, alt_az.decl

    assert ra == trans_ra
    assert decl == trans_decl


def test_calculation_obs_no_time_update():
    "Calculate Observation should not change the state of the obsprog"
    obsprog = default_setup()
    start_mjd = obsprog.mjd

    action = {}
    obsprog.calculate_obversation(action)

    assert start_mjd == obsprog.mjd


def test_angle_ranges():
    "All the angles are in the right range"


def test_airmass_range():
    "Airmass is limited"
    obsprog = default_setup()
    random_actions = [{}]
    ra_vars = []
    ra_range = []

    decl_vars = []
    decl_range = []

    for action in random_actions:
        observation = obsprog.calculate_obversation(action)
        for var in ra_vars:
            assert ra_range[0] <= observation[var]
            assert ra_range[1] >= observation[var]

        for var in decl_vars:
            assert decl_range[0] <= observation[var]
            assert decl_range[1] >= observation[var]


def test_nighttime_is_night():
    "As it says on the tin"
    obsprog = default_setup()

    night_mjd = ""
    obsprog.mjd = night_mjd

    assert obsprog.check_nighttime() is True


def test_daytime_is_day():
    "As it says on the tin"
    obsprog = default_setup()

    day_mjd = ""
    obsprog.mjd = day_mjd

    assert obsprog.check_nighttime() is False


def test_dusk_is_day():
    "As it says on the tin"
    obsprog = default_setup()

    dusk_mjd = ""
    obsprog.mjd = dusk_mjd

    assert obsprog.check_nighttime() is False


def test_dawn_is_day():
    "As it says on the tin"
    obsprog = default_setup()

    dawn_mjd = ""
    obsprog.mjd = dawn_mjd

    assert obsprog.check_nighttime() is False


def test_day_advance_to_night():
    obsprog = default_setup()

    day_mjd = ""
    obsprog.mjd = day_mjd

    assert obsprog.check_nighttime() is False

    obsprog.advance_to_nighttime()
    assert obsprog.mjd == day_mjd
    assert obsprog.check_nighttime() is True


def test_dusk_advance_to_night():
    obsprog = default_setup()

    day_mjd = ""
    obsprog.mjd = day_mjd

    assert obsprog.check_nighttime() is False

    obsprog.advance_to_nighttime()
    assert obsprog.mjd == day_mjd
    assert obsprog.check_nighttime() is True


def test_dawn_advance_to_night():
    obsprog = default_setup()

    day_mjd = ""
    obsprog.mjd = day_mjd

    assert obsprog.check_nighttime() is False

    obsprog.advance_to_nighttime()
    assert obsprog.mjd == day_mjd
    assert obsprog.check_nighttime() is True


if __name__ == "__main__":
    pytest.main()
