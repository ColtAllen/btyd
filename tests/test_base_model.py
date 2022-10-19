from __future__ import generator_stop
from __future__ import annotations

import os
from abc import ABC
import warnings
import inspect

import pytest

import pandas as pd
import numpy as np
import arviz as az

import pymc as pm
import aesara.tensor as at

import btyd
from btyd.models import BaseModel, PredictMixin


def test_deprecated():
    """
    GIVEN the deprecated() function for DeprecationWarnings,
    WHEN it is called,
    THEN one warning of category DeprecationWarning containing `deprecated` in the message is returned.
    """

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        btyd.deprecated()
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated" in str(w[-1].message)


@pytest.mark.parametrize("obj", [BaseModel, PredictMixin])
def test_isabstract(obj):
    """
    GIVEN the BaseModel and PredictMixin model factory objects,
    WHEN they are inspected for inheritance from ABC,
    THEN they should both identify as abstract objects.
    """

    assert inspect.isabstract(obj) is True


class TestBaseModel:
    def test_repr(self):
        """
        GIVEN a BaseModel that has not been instantiated,
        WHEN repr() is called on this object,
        THEN a string representation containing library name, module and model class are returned.
        """

        assert repr(BaseModel) == "<class 'btyd.models.BaseModel'>"

    def test_abstract_methods(self):
        """
        GIVEN the BaseModel model factory object,
        WHEN its abstract methods are overridden,
        THEN they should all return None.
        """

        # Override abstract methods:
        BaseModel.__abstractmethods__ = set()

        # Create concrete class for testing:
        class ConcreteBaseModel(BaseModel):
            pass

        # Instantiate concrete testing class and call all abstrast methods:
        concrete_base = ConcreteBaseModel()
        model = concrete_base._model()
        log_likelihood = concrete_base._log_likelihood()
        generate_rfm_data = concrete_base.generate_rfm_data()

        assert model is None
        assert log_likelihood is None
        assert generate_rfm_data is None

    def test_sample(self):
        """
        GIVEN the _sample() static method,
        WHEN a numpy array and sample quantity are provided,
        THEN a numpy array of the specified length containing some or all of the original elements is returned.
        """
        posterior_distribution = np.array([0.456, 0.358, 1.8, 2.0, 0.999])
        samples = 7
        posterior_samples = BaseModel._sample(posterior_distribution, samples)
        assert len(posterior_samples) == samples

        # Convert numpy arrays to sets to test intersections of elements.
        dist_set = set(posterior_distribution.flatten())
        sample_set = set(posterior_samples.flatten())
        assert len(sample_set.intersection(dist_set)) <= len(posterior_distribution)

    def test_dataframe_parser(self, cdnow):
        """
        GIVEN an RFM dataframe,
        WHEN the _dataframe_parser() static method is called on it,
        THEN five numpy arrays should be returned.
        """

        parsed = BaseModel._dataframe_parser(cdnow)
        assert len(parsed) == 5

    def test_check_inputs(self):
        """
        GIVEN separate arrays for frequency, recency, T, and monetary value,
        WHEN _check_inputs() is called,
        THEN None should be returned unless any of the arrays violate input assumptions.
        """

        frequency = np.array([0, 1, 2])
        recency = np.array([0, 1, 10])
        T = np.array([5, 6, 15])
        monetary_value = np.array([2.3, 490, 33.33])
        assert (
            BaseModel()._check_inputs(frequency, recency, T, monetary_value) is None
        )

        with pytest.raises(ValueError):
            bad_recency = T + 1
            BaseModel()._check_inputs(frequency, bad_recency, T)

        with pytest.raises(ValueError):
            bad_recency = recency.copy()
            bad_recency[0] = 1
            BaseModel()._check_inputs(frequency, bad_recency, T)

        with pytest.raises(ValueError):
            bad_freq = np.array([0, 0.5, 2])
            BaseModel()._check_inputs(bad_freq, recency, T)

        # with pytest.raises(ValueError):
        #     bad_monetary_value = monetary_value.copy()
        #     bad_monetary_value[0] = 0
        #     BaseModel()._check_inputs(frequency, recency, T, bad_monetary_value)


class TestPredictMixin:
    def test_abstract_methods(self):
        """
        GIVEN the PredictMixin model factory object,
        WHEN its abstract methods are overridden,
        THEN they should all return None.
        """

        # Override abstract methods:
        PredictMixin.__abstractmethods__ = set()

        # Create concrete class for testing:
        class ConcretePredictMixin(PredictMixin):
            pass

        # Instantiate concrete testing class and call all abstrast methods:
        concrete_api = ConcretePredictMixin()
        cond_prob_alive = concrete_api._conditional_probability_alive()
        cond_n_prchs_to_time = (
            concrete_api._conditional_expected_number_of_purchases_up_to_time()
        )
        n_prchs_to_time = concrete_api._expected_number_of_purchases_up_to_time()
        prob_n_prchs_to_time = concrete_api._probability_of_n_purchases_up_to_time()

        assert cond_prob_alive is None
        assert cond_n_prchs_to_time is None
        assert cond_prob_alive is None
        assert n_prchs_to_time is None
