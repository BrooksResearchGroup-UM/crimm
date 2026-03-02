"""Pytest configuration for crimm tests.

Skips test files that require pyCHARMM or are standalone scripts.
These files import pyCHARMM at module level which initializes the
CHARMM runtime and hangs during pytest collection.
"""

import os

_HERE = os.path.dirname(__file__)

collect_ignore = [
    os.path.join(_HERE, "test_minimal_psf.py"),
    os.path.join(_HERE, "test_crimm_blade.py"),
    os.path.join(_HERE, "test_huge_waterbox.py"),
    os.path.join(_HERE, "test_solvate_1lsa.py"),
    os.path.join(_HERE, "test_solvate_2lzt.py"),
    os.path.join(_HERE, "test_whole_model_with_ions.py"),
    os.path.join(_HERE, "test_solvate_5oph.py"),
    os.path.join(_HERE, "test_psf_crd_comparison.py"),
    os.path.join(_HERE, "test_psf_crd_io.py"),
    os.path.join(_HERE, "test_psf_fixes_verification.py"),
]
