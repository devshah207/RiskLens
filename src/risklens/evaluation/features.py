"""Shared feature groups for the fixed historical experiments."""
from risklens.features.dataset import FULL_FEATURE_COLUMNS
EXTERNAL_FEATURE_COLUMNS = ("india_vix_level", "sp500_previous_return_1d")
MULTIFACTOR_FEATURE_COLUMNS = FULL_FEATURE_COLUMNS + EXTERNAL_FEATURE_COLUMNS
