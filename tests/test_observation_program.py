'''
Testing relating to the observation program itself.
Checks things like units, and validity of variable groups
'''

import pytest
from rltelescope.observation_program import ObservationProgram

def test_start_time():
    "Start time and end time are the obversation period appart at all times"

def test_vertical_valid_observation():
    "A straight up obversation is valid"

def test_horizonal_invalid_obversation():
    "An invalid obversation points at the horizon"

def test_vertical_invalid_obversation():
    "A straight down obversation is invalid"



if __name__ == "__main__":
    pytest.main()
