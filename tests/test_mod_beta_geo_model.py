from __future__ import generator_stop
from __future__ import annotations

import os

import pytest

import pandas as pd
import numpy as np
import arviz as az

import pymc as pm
import aesara.tensor as at

import btyd


class TestModBetaGeoModel:
    @pytest.fixture(scope="class")
    def fitted_mbgm(self, cdnow):
        """For running multiple tests on a single GammaGammaModel fit() instance."""

        mbgm = btyd.ModBetaGeoModel().fit(cdnow)
        return mbgm
        
    def test_hyperparams(self):
        """
        GIVEN an uninstantiated ModBetaGeoModel,
        WHEN it is instantiated with custom values for hyperparams,
        THEN ModBetaGeoModel._hyperparams should differ from defaults and match the custom values set by the user.
        """

        custom_hyperparams = {
            "alpha_prior_alpha": 2.0,
            "alpha_prior_beta": 4.0,
            "r_prior_alpha": 3.0,
            "r_prior_beta": 2.0,
            "phi_prior_lower": 0.1,
            "phi_prior_upper": 0.99,
            "kappa_prior_alpha": 1.1,
            "kappa_prior_m": 2.5,
        }

        default_mbgm = btyd.ModBetaGeoModel()
        custom_mbgm = btyd.ModBetaGeoModel(hyperparams=custom_hyperparams)

        assert default_mbgm._hyperparams != custom_mbgm._hyperparams
        assert custom_hyperparams == custom_mbgm._hyperparams

    def test_log_likelihood(self):
        """
        GIVEN the ModBetaGeo log-likelihood function,
        WHEN it is called with the specified inputs and parameters,
        THEN term values and output should match those in the paper.
        """

        values = {
            "frequency": 200,
            "recency": 38,
            "T": 40,
            "r": 0.53,
            "alpha": 6.18,
            "a": 0.89,
            "b": 1.614,
        }

        # Test output.
        loglike_out = btyd.ModBetaGeoModel._log_likelihood(self, **values).eval()
        expected = np.array([91.808242])
        np.testing.assert_allclose(loglike_out, expected, rtol=1e-04)

    def test_repr(self, fitted_mbgm):
        """
        GIVEN a declared ModBetaGeo concrete class object,
        WHEN the string representation is called on this object,
        THEN string representations of library name, module, ModBetaGeoModel class, parameters, and # rows used in estimation are returned.
        """

        assert (
            repr(btyd.ModBetaGeoModel)
            == "<class 'btyd.models.mod_beta_geo_model.ModBetaGeoModel'>"
        )
        assert repr(btyd.ModBetaGeoModel()) == "<btyd.ModBetaGeoModel>"

        # Expected parameters may vary slightly due to rounding errors.
        expected = [
            "<btyd.ModBetaGeoModel: Parameters {'alpha': 6.2, 'r': 0.5, 'a': 0.9, 'b': 1.6} estimated with 2357 customers.>",
            "<btyd.ModBetaGeoModel: Parameters {'alpha': 6.2, 'r': 0.5, 'a': 0.9, 'b': 1.7} estimated with 2357 customers.>",
            "<btyd.ModBetaGeoModel: Parameters {'alpha': 6.3, 'r': 0.5, 'a': 0.9, 'b': 1.6} estimated with 2357 customers.>",
        ]
        assert repr(fitted_mbgm) in expected

    def test_model(self, fitted_mbgm):
        """
        GIVEN an instantiated ModBetaGeo model,
        WHEN _model is called,
        THEN it should contain the specified random variables.
        """

        model = fitted_mbgm._model()
        expected = "[ModBetaGeoModel::alpha, ModBetaGeoModel::r, ModBetaGeoModel::phi, ModBetaGeoModel::kappa, ModBetaGeoModel::a, ModBetaGeoModel::b]"
        assert str(model.unobserved_RVs) == expected

    def test_fit(self, fitted_mbgm):
        """
        GIVEN a ModBetaGeoModel() object,
        WHEN it is fitted,
        THEN the new instantiated attributes should include an arviz InferenceData class and dict with required model parameters.
        """

        assert isinstance(fitted_mbgm._idata, az.InferenceData)

        # Check if arviz methods are supported.
        summary = az.summary(
            data=fitted_mbgm._idata,
            var_names=[
                "ModBetaGeoModel::a",
                "ModBetaGeoModel::b",
                "ModBetaGeoModel::alpha",
                "ModBetaGeoModel::r",
            ],
        )
        assert isinstance(summary, pd.DataFrame)

    def test_unload_params(self, fitted_mbgm):
        """
        GIVEN a Bayesian ModBetaGeoModel fitted on the CDNOW dataset,
        WHEN its parameters are checked via self._unload_params(),
        THEN they should be within 1e-01 tolerance of the MLE parameters in https://github.com/mplatzer/BTYDplus.
        """

        expected = np.array([6.18, 0.53, 0.89, 1.614])
        np.testing.assert_allclose(
            expected, np.array(fitted_mbgm._unload_params()), rtol=1e-01
        )

    def test_posterior_sampling(self, fitted_mbgm):
        """
        GIVEN a Bayesian ModBetaGeoModel fitted on the CDNOW dataset,
        WHEN its posterior parameter distributions are sampled via self._unload_params(posterior = True),
        THEN the length of the numpy arrays should match that of the n_samples argument.
        """

        sampled_posterior_params = fitted_mbgm._unload_params(posterior=True)

        assert len(sampled_posterior_params) == 4
        assert sampled_posterior_params[0].shape == (100,)

    def test_conditional_expected_number_of_purchases_up_to_time(self, fitted_mbgm):
        """
        GIVEN a Bayesian ModBetaGeoModel fitted on the CDNOW dataset,
        WHEN self._conditional_expected_number_of_purchases_up_to_time() is called with t = 39,
        THEN it should return a value within 1e-02 tolerance to the expected MLE output from the Hardie Excel spreadsheet.
        """

        fitted_mbgm._frequency = 2
        fitted_mbgm._recency = 30.43
        fitted_mbgm._T = 38.86
        t = 39

        expected = np.array(1.226)
        actual = fitted_mbgm._conditional_expected_number_of_purchases_up_to_time(t)
        np.testing.assert_allclose(expected, actual, rtol=1e-01)

    def test_expected_number_of_purchases_up_to_time(self, fitted_mbgm):
        """
        GIVEN a Bayesian ModBetaGeoModel fitted on the CDNOW dataset,
        WHEN self._expected_number_of_purchases_up_to_time() is called,
        THEN it should return a value within 1e-02 tolerance to the expected MLE output from the Hardie Excel spreadsheet.
        """

        times = np.array([0.1429, 1.0, 3.00, 31.8571, 32.00, 78.00])
        expected = np.array([[0.0078, 0.0532, 0.1506, 1.0405, 1.0437, 1.8576]])
        actual = fitted_mbgm._expected_number_of_purchases_up_to_time(times, None)
        np.testing.assert_allclose(actual, expected, rtol=1e-01)

    def test_conditional_probability_alive(self, fitted_mbgm):
        """
        GIVEN a fitted ModBetaGeoModel object,
        WHEN self._conditional_probability_alive() is called,
        THEN output should always be between 0 and 1.
        """

        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                for k in range(j, 100, 10):
                    assert (
                        0
                        <= fitted_mbgm._conditional_probability_alive(
                            None, None, False, 100, i, j, k
                        )
                        < [1.0]
                    )

    def test_probability_of_n_purchases_up_to_time(self, fitted_mbgm):
        """
        GIVEN a fitted ModBetaGeoModel object,
        WHEN self._probability_of_n_purchases_up_to_time() is called,
        THEN output should approximate that of the BTYD R package: https://cran.r-project.org/web/packages/BTYD/BTYD.pdf
        """

        # probability that a customer will make 10 repeat transactions in the
        # time interval (0,2]
        expected = np.array(2.247434e-08)
        actual = fitted_mbgm._probability_of_n_purchases_up_to_time(2, 10)
        np.testing.assert_allclose(expected, actual, rtol=1e-01)

        # probability that a customer will make no repeat transactions in the
        # time interval (0,39]
        expected = 0.5737864
        actual = fitted_mbgm._probability_of_n_purchases_up_to_time(39, 0)
        np.testing.assert_allclose(expected, actual, rtol=1e-02)

        # PMF
        expected = np.array(
            [
                0.0019995214,
                0.0015170236,
                0.0011633150,
                0.0009003148,
                0.0007023638,
                0.0005517902,
                0.0004361913,
                0.0003467171,
                0.0002769613,
                0.0002222260,
            ]
        )
        actual = np.array(
            [
                fitted_mbgm._probability_of_n_purchases_up_to_time(30, n)
                for n in range(11, 21)
            ]
        ).flatten()
        np.testing.assert_allclose(expected, actual, rtol=1e-00)

    def test_quantities_of_interest(self):
        """
        GIVEN the _quantities_of_interest PredictMixin attribute,
        WHEN the keys of the '_quantities_of_interest' call dictionary attribute are called,
        THEN they should match the list of expected keys.
        """

        expected = [
            "cond_prob_alive",
            "cond_n_prchs_to_time",
            "n_prchs_to_time",
            "prob_n_prchs_to_time",
        ]
        actual = list(btyd.ModBetaGeoModel._quantities_of_interest.keys())
        assert actual == expected

    @pytest.mark.parametrize(
        "qoi",
        [
            "cond_prob_alive",
            "cond_n_prchs_to_time",
            "n_prchs_to_time",
            "prob_n_prchs_to_time",
        ],
    )
    def test_predict_mean(self, fitted_mbgm, cdnow, qoi):
        """
        GIVEN a fitted ModBetaGeoModel,
        WHEN all four quantities of interest are called via ModBetaGeoModel.predict() with and w/o data for posterior mean predictions,
        THEN expected output instances and datatypes should be returned.
        """

        array_out = fitted_mbgm.predict(method=qoi, t=10, n=5)
        assert isinstance(array_out, np.ndarray)

        array_out = fitted_mbgm.predict(method=qoi, rfm_df=cdnow, t=10, n=5)
        assert isinstance(array_out, np.ndarray)

    @pytest.mark.parametrize(
        "qoi, draws",
        [
            ("cond_prob_alive", 100),
            ("cond_n_prchs_to_time", 200),
            ("n_prchs_to_time", 300),
            ("prob_n_prchs_to_time", 400),
        ],
    )
    def test_predict_full(self, fitted_mbgm, cdnow, qoi, draws):
        """
        GIVEN a fitted ModBetaGeoModel,
        WHEN all four quantities of interest are called via ModBetaGeoModel.predict() for full posterior predictions,
        THEN expected output instances and dimensions should be returned.
        """

        array_out = fitted_mbgm.predict(
            method=qoi,
            rfm_df=cdnow,
            t=10,
            n=5,
            sample_posterior=True,
            posterior_draws=draws,
        )

        assert isinstance(array_out, np.ndarray)
        assert len(array_out) == draws

    def test_generate_rfm_data(self, fitted_mbgm):
        """
        GIVEN a fitted ModBetaGeoModel,
        WHEN synthetic data is generated from its parameters,
        THEN the resultant dataframe should contain the expected column names and row count.
        """

        # Test default value of size argument.
        synthetic_df = fitted_mbgm.generate_rfm_data()
        assert len(synthetic_df) == 1000

        # Test custom value of size argument.
        synthetic_df = fitted_mbgm.generate_rfm_data(size=123)
        assert len(synthetic_df) == 123

        expected_cols = ["frequency", "recency", "T", "lambda", "p", "alive"]
        actual_cols = list(synthetic_df.columns)

        assert actual_cols == expected_cols

    def test_save(self, fitted_mbgm):
        """
        GIVEN a fitted ModBetaGeoModel object,
        WHEN self.save_model() is called,
        THEN the external JSON and CSV files should exist.
        """

        # Remove saved file if it already exists:
        try:
            os.remove("mbgnbd.json")
        except FileNotFoundError:
            pass
        finally:
            assert os.path.isfile("mbgnbd.json") == False

            fitted_mbgm.save("mbgnbd.json")
            assert os.path.isfile("mbgnbd.json") == True

    def test_load(self, fitted_mbgm):
        """
        GIVEN fitted and unfitted ModBetaGeoModel objects,
        WHEN parameters of the fitted model are loaded from an external JSON and CSV via self.load_model(),
        THEN InferenceData unloaded parameters should match, raising exceptions otherwise and if predictions attempted without RFM data.
        """

        mbgm_new = btyd.ModBetaGeoModel()
        mbgm_new.load("mbgnbd.json")
        assert isinstance(mbgm_new._idata, az.InferenceData)
        # assert mbgm_new._idata.posterior.keys() ==  'sample'
        assert mbgm_new._unload_params() == fitted_mbgm._unload_params()

        # assert param exception (need another saved model and additions to self.load_model())
        # assert prediction exception

        os.remove("mbgnbd.json")
