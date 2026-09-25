"""
Archive of curves this project's search_common.py has been pointed at over
time. Previously these lived as a long chain of overriding module-level
assignments in search_common.py (only the last assignment before the next
config section ever took effect) -- moved here so they're preserved without
shadowing the active config. Not imported by anything; copy an entry into
the ACTIVE CURVE block in search_common.py to use it.
"""

from sage.all import QQ
from parse_genus3 import parse_hyperelliptic_db_entry

CURVE_ARCHIVE = [
    {
        "coeffs": [QQ(1), QQ(4), QQ(-2), QQ(-18), QQ(1), QQ(38), QQ(25)],
        "data_pts": [QQ(-1)],
        "terminate_when": 11,
    },
    {
        "comment": "# doesn't find y=0 point... added a special function to find these...maybe ok...",
        "coeffs": [QQ(1), QQ(4), QQ(12), QQ(16), QQ(-12), QQ(-20), QQ(12)],
        "data_pts": [QQ(-2)],
        "terminate_when": 2,
    },
    {
        "coeffs": [QQ(4), QQ(0), QQ(-12), QQ(-4), QQ(12), QQ(8), QQ(-7)],
        "data_pts": [QQ(1)],
        "terminate_when": 3,
    },
    {
        "coeffs": [QQ(1), QQ(2), QQ(-11), QQ(-12), QQ(56), QQ(16), QQ(-116)],
        "data_pts": [QQ(-3)],
        "terminate_when": 3,
    },
    {
        "coeffs": [QQ(1), QQ(2), QQ(1), QQ(-6), QQ(2), QQ(8), QQ(-7)],
        "data_pts": [QQ(1)],
        "terminate_when": 2,
    },
    {
        "coeffs": [QQ(4), QQ(0), QQ(-16), QQ(24), QQ(-16), QQ(5)],
        "data_pts": [QQ(1)],
        "terminate_when": 2,
    },
    {
        "coeffs": [QQ(1), QQ(4), QQ(2), QQ(-18), QQ(21), QQ(-10), QQ(1)],
        "data_pts": [QQ(1)],
        "terminate_when": 4,
    },
    {
        "coeffs": [QQ(1), QQ(6), QQ(10), QQ(7), QQ(1), QQ(0)],
        "data_pts": [QQ(-1)],
        "terminate_when": 3,
    },
    {
        "coeffs": [QQ(1), QQ(2), QQ(3), QQ(2), QQ(5), QQ(8), QQ(-4)],
        "data_pts": [QQ(-5)/QQ(3)],
        "terminate_when": 3,
    },
    {
        "comment": "deg 5",
        "coeffs": [QQ(4), QQ(4), QQ(-16), QQ(-19), QQ(16), QQ(20)],
        "data_pts": [QQ(-1)],
        "terminate_when": 2,
    },
    {
        "comment": "genus 3 test curve",
        "coeffs": [QQ(1), QQ(0), QQ(0), QQ(0), QQ(2), QQ(0), QQ(-4), QQ(0), QQ(1)],
        "data_pts": [QQ(0)],
        "terminate_when": 4,
    },
    {
        "comment": "Y² = -20x^7 - 15x^6 - 10x^5 - 5x^4 + 4x^3 + 3x^2 + 2x + 1",
        "coeffs": [QQ(-20), QQ(-15), QQ(-10), QQ(-5), QQ(4), QQ(3), QQ(2), QQ(1)],
        "data_pts": [QQ(0)],
        "terminate_when": 4,
    },
    {
        "comment": "db_entry = '9995456:2498864:[2*x^7-4*x^6-5*x^5+10*x^4+5*x^3-8*x^2-3*x+1,x^2+x]'",
        "coeffs": parse_hyperelliptic_db_entry('9995408:2498852:[x^8-x^6+x^3+2*x^2+x,x^2+x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 5,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9999936:1249992:[x^6+3*x^5+5*x^4+5*x^3+4*x^2+2*x,x^4+x^3+x^2+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 2,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9999872:4999936:[x^7-x^4+x^3-x^2,x^2+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 2,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9999868:4999934:[2*x^5+6*x^4+5*x^3+x^2+x+1,x^4+x^3+x]'),
        "data_pts": [QQ(0)],
        "terminate_when": 4,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9999609:9999609:[-3*x^6-6*x^5-8*x^4-4*x^3-x^2+x,x^4+x^2+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 1,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9999469:9999469:[-x^7+2*x^6+x^5-5*x^4+x^3+2*x^2-2*x,x^3+x^2+1]'),
        "data_pts": [QQ(1)],
        "terminate_when": 1,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9998993:9998993:[x^7+x^6-4*x^5+x^4+4*x^3-3*x^2-x+1,x^2]'),
        "data_pts": [QQ(0)],
        "terminate_when": 2,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9998809:9998809:[x^7-3*x^6-3*x^5+5*x^4-2*x^3-4*x^2+2*x-1,x^2+x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 2,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9998659:9998659:[-x^6+3*x^4-7*x^2-12*x-9,x^4+x^3+x^2+x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 3,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9998263:9998263:[3*x^7+x^6-3*x^5-2*x^4+10*x^3-12*x^2+5*x-1,x^4+x^2+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 2,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9998039:9998039:[x^4+2*x^3+x^2+x+1,x^4+x^3+x^2]'),
        "data_pts": [QQ(0)],
        "terminate_when": 2,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9997256:9997256:[x^7+x^6-2*x^5-5*x^4-x^3+2*x^2-1,x^4+x^2+x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 2,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9997199:9997199:[3*x^3+x^2-2*x,x^4+x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 3,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9996680:2499170:[-x^7-x^6+8*x^5-13*x^4+12*x^3-6*x^2+x,x^4+x]'),
        "data_pts": [QQ(0)],
        "terminate_when": 2,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9996392:2499098:[x^8+3*x^7-2*x^6-8*x^5+3*x^4+7*x^3-5*x^2-2*x+1,x^3+x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 2,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9995549:9995549:[x^8+3*x^7+2*x^6+x^5+3*x^4+x^3+x,x]'),
        "data_pts": [QQ(0)],
        "terminate_when": 3,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9995087:9995087:[-x^7-x^6-2*x^5+x^2,x^4+x^3+x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 3,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9995008:4997504:[-x^8+5*x^6-x^5-8*x^4+4*x^3+4*x^2-4*x,x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 1,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9995008:624688:[x^7-x^6-3*x^5+x^4-x^2,x^3+x^2+x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 2,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9997263:3332421:[x^7+x^6-4*x^5-2*x^4+x^3-x,x^4+x^3+x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 4,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9994635:3331545:[x^7+2*x^6-x^5+8*x^3+3*x^2-5*x-2,x^4+x^3+x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 4,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9996912:3332304:[x^5+2*x^4+x^3-x^2-2*x-1,x^4+x^3+x^2]'),
        "data_pts": [QQ(0)],
        "terminate_when": 3,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9995408:2498852:[x^8-x^6+x^3+2*x^2+x,x^2+x+1]'),
        "data_pts": [QQ(0)],
        "terminate_when": 5,
    },
    {
        "coeffs": parse_hyperelliptic_db_entry('9996352:312386:[-2*x^6-6*x^5+x^4+18*x^3+10*x^2-17*x-15,x^4+x^3+x]'),
        "data_pts": [QQ(0)],
        "terminate_when": 3,
    },
    {
        "coeffs": [QQ(1), QQ(4), QQ(2), QQ(-30), QQ(33), QQ(-10), QQ(1)],
        "data_pts": [QQ(1)],
        "terminate_when": 4,
    },
    {
        "coeffs": [QQ(1), QQ(0), QQ(-4), QQ(10), QQ(-24), QQ(24), QQ(-7)],
        "data_pts": [QQ(2)],
        "terminate_when": 3,
    },
    {
        "coeffs": [QQ(1), QQ(4), QQ(4), QQ(4), QQ(8), QQ(-8), QQ(-12)],
        "data_pts": [QQ(-1)],
        "terminate_when": 3,
    },
    {
        "coeffs": [QQ(4), QQ(-4), QQ(-36), QQ(5), QQ(96), QQ(64)],
        "data_pts": [QQ(-1)],
        "terminate_when": 4,
    },
    {
        "comment": "$y^2 = 4x^6 + 9x^4 - 4x^3 + 2x^2 - 4x + 1$ # rank 2",
        "coeffs": [QQ(4), QQ(0), QQ(9), QQ(-4), QQ(2), QQ(-4), QQ(1)],
        "data_pts": [QQ(0)],
        "terminate_when": 3,
    },
    {
        "comment": "$y^2 = 4x^6 - 12x^5 + 16x^4 - 8x^3 - 3x^2 + 4x$ # rank 2",
        "coeffs": [QQ(4), QQ(-12), QQ(16), QQ(-8), QQ(-3), QQ(4), QQ(0)],
        "data_pts": [QQ(1)],
        "terminate_when": 3,
    },
    {
        "coeffs": [QQ(1), QQ(-12), QQ(30), QQ(2), QQ(-15), QQ(2), QQ(1)],
        "data_pts": [QQ(1)],
        "terminate_when": 12,
    },
    {
        "comment": "prestige curve lol, rank 4",
        "coeffs": [QQ(1), QQ(8), QQ(10), QQ(-10), QQ(-11), QQ(2), QQ(1)],
        "data_pts": [QQ(-1)],
        "terminate_when": 11,
    },
    {
        "comment": "attack curve, i guess y² = 8x⁵ + 16x⁴ - 60x³ + 69x² - 36x + 8",
        "coeffs": [QQ(8), QQ(16), QQ(-60), QQ(69), QQ(-36), QQ(8)],
        "data_pts": [QQ(1)/QQ(2)],
        "terminate_when": 2,
    },
    {
        "comment": "claude generated this curve, not in the lmfdb as of Jan 3 2026 y² = -3x⁶ + 11x⁵ + 6x⁴ - 9x³ + 2x² + x + 25",
        "coeffs": [QQ(-3), QQ(11), QQ(6), QQ(-9), QQ(2), QQ(1), QQ(25)],
        "data_pts": [QQ(0)/QQ(1)],
        "terminate_when": 2,
    },
    {
        "comment": "$$y^2 = x^5 + x + 2$$ DATA_PTS_GENUS2 = [QQ(1)/QQ(1)] # just the x values lol",
        "coeffs": [QQ(1), QQ(0),QQ(0),QQ(0),QQ(1),QQ(7)],
        "data_pts": [QQ(1)],
        "terminate_when": 30,
    },
    {
        "comment": "y^2 = x^5 + 3x^3 + 2x^2 + 5x + 4",
        "coeffs": [QQ(1), QQ(0), QQ(3), QQ(2), QQ(5), QQ(4)],
        "data_pts": [3],  # NOTE: bare int in the original, not QQ(3) -- wrap in QQ() before using
        "terminate_when": 3,
    },
]
# Note: Hindes' curve is the currently active curve in search_common.py, so
# it isn't duplicated here.
