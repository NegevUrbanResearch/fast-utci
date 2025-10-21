"""
Test constants and fixtures for fast-utci testing.

Contains validation periods, test parameters, and other constants
used across the test suite.
"""

# Validation analysis period (moved from config)
# August 15th, full day (0-23 hours)
VALIDATION_ANALYSIS_PERIOD = {
    'start_month': 8,
    'start_day': 15,
    'start_hour': 0,
    'end_month': 8,
    'end_day': 15,
    'end_hour': 23
}

# Validation target hour (1-2 PM)
VALIDATION_TARGET_HOURS = [13]

# Test data paths
TEST_MODEL_FILE = "data/3d_models/100.gltf"
TEST_EPW_FILE = "data/weather/ISR_Beer.Sheva.401900_MSI.epw"

