import pytest

import btyd
from btyd.datasets import load_cdnow_summary_data_with_monetary_value

import pandas as pd


@pytest.fixture(scope="package")
def cdnow() -> pd.DataFrame:
    """Create an RFM dataframe for multiple tests and fixtures."""
    rfm_df = load_cdnow_summary_data_with_monetary_value()
    return rfm_df


@pytest.fixture(scope="package")
def fitted_bgm(cdnow):
    """For running multiple tests on a single BetaGeoModel fit() instance."""

    bgm = btyd.BetaGeoModel().fit(cdnow)
    return bgm
