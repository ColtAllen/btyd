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
import btyd.utils as utils

class TestGammaGammaModel:

    @pytest.fixture(scope='class')
    def fitted_ggm(self,cdnow):
        """ For running multiple tests on a single GammaGammaModel fit() instance. """

        ggm = btyd.GammaGammaModel().fit(cdnow)
        return ggm
    
    def test_hyperparams(self):
        """
        GIVEN an uninstantiated GammaGammaModel,
        WHEN it is instantiated with custom values for hyperparams,
        THEN GammaGammaModel._hyperparams should differ from defaults and match the custom values set by the user. 
        """
    
        custom_hyperparams = {
                "p_prior_alpha": 2.0,
                "p_prior_beta": 3.0,
                "q_prior_alpha": .4,
                "q_prior_beta": 1.2,
                "v_prior_alpha": .99,
                "v_prior_beta": 4.0,
            }

        default_ggm = btyd.GammaGammaModel()
        custom_ggm = btyd.GammaGammaModel(hyperparams = custom_hyperparams)

        assert default_ggm._hyperparams != custom_ggm._hyperparams
        assert custom_hyperparams == custom_ggm._hyperparams
    
    def test_log_likelihood(self):
        """
        GIVEN the GammaGamma log-likelihood function,
        WHEN it is called with the specified inputs,
        THEN the output should equal the specified value. 
        """
        
        values = {
            'frequency': 50,
            'monetary_value': 38,
            'p': 40,
            'q': 0.25,
            'v': 4.,
        }

        loglike_out = btyd.GammaGammaModel._log_likelihood(self,**values).eval()
        expected = np.array([100.7957])
        np.testing.assert_allclose(loglike_out,expected,rtol=1e-04)
    
    def test_repr(self,fitted_ggm):
        """
        GIVEN a declared GammaGamma concrete class object,
        WHEN the string representation is called on this object,
        THEN string representations of library name, module, GammaGammaModel class, parameters, and # rows used in estimation are returned.
        """

        assert repr(btyd.GammaGammaModel) == "<class 'btyd.models.gamma_gamma_model.GammaGammaModel'>"
        assert repr(btyd.GammaGammaModel()) == "<btyd.GammaGammaModel>"
        
        # Expected parameters may vary slightly due to rounding errors.
        expected = [
             "<btyd.GammaGammaModel: Parameters {'p': 6.25, 'q': 3.74, 'v': 15.44} estimated with 2357 customers.>",
              "<btyd.GammaGammaModel: Parameters {'p': 6.25, 'q': 3.74, 'v': 15.44} estimated with 2357 customers.>",
        ]
        assert any(expected) == True
    
    def test_model(self,fitted_ggm):
        """
        GIVEN an instantiated GammaGamma model,
        WHEN _model is called,
        THEN it should contain the specified random variables.
        """

        model = fitted_ggm._model()
        expected = '[GammaGammaModel::p, GammaGammaModel::q, GammaGammaModel::v]'
        assert str(model.unobserved_RVs) == expected
    
    def test_fit(self,fitted_ggm):
        """
        GIVEN a GammaGammaModel() object,
        WHEN it is fitted,
        THEN the new instantiated attributes should include an arviz InferenceData class and dict with required model parameters.
        """

        assert isinstance(fitted_ggm._idata,az.InferenceData)

        # Check if arviz methods are supported.
        summary = az.summary(
            data=fitted_ggm._idata, 
            var_names=['GammaGammaModel::p','GammaGammaModel::q','GammaGammaModel::v']
            )
        assert isinstance(summary,pd.DataFrame)
    
    def test_unload_params(self, fitted_ggm):
        """
        GIVEN a Bayesian GammaGammaModel fitted on the CDNOW dataset,
        WHEN its parameters are checked via self._unload_params(),
        THEN they should be within 1e-01 tolerance of the MLE parameters from the original paper.
        """

        expected = np.array([6.25, 3.74, 15.44])
        np.testing.assert_allclose(expected, np.array(fitted_ggm._unload_params()),rtol=1e-01)

    def test_posterior_sampling(self,fitted_ggm):
        """
        GIVEN a Bayesian GammaGammaModel fitted on the CDNOW dataset,
        WHEN its posterior parameter distributions are sampled via self._unload_params(posterior = True),
        THEN the length of the numpy arrays should match that of the n_samples argument.
        """

        sampled_posterior_params = fitted_ggm._unload_params(posterior=True)

        assert len(sampled_posterior_params) == 4
        assert sampled_posterior_params[0].shape == (100,)

    def test_conditional_expected_average_profit(self, fitted_ggm, cdnow):
        """
        GIVEN a GammaGammaModel fitted on the CDNOW dataset,
        WHEN self._conditional_expected_average_profit() is called on the first 10 rows of CDNOW,
        THEN the output should match that from Hardie's notes.
        """

        summary = cdnow.head(10)
        estimates = fitted_ggm._conditional_expected_average_profit(summary["frequency"], summary["monetary_value"])
        expected = np.array(
            [24.65, 18.91, 35.17, 35.17, 35.17, 71.46, 18.91, 35.17, 27.28, 35.17]
        )  # from Hardie spreadsheet http://brucehardie.com/notes/025/

        np.testing.assert_allclose(estimates.values, expected, atol=0.1)

    def test_customer_lifetime_value_with_bgf(self, cdnow, fitted_ggm, fitted_bgm):
        """
        GIVEN GammaGammaModel and BetaGeoModel objects fitted on the CDNOW dataset,
        WHEN GammaGammaModel._customer_lifetime_value() is called on the first 10 rows of CDNOW,
        THEN the output should match that from Hardie's notes.
        """

        ggf_clv = fitted_ggm._customer_lifetime_value(
            fitted_bgm,
            cdnow["frequency"],
            cdnow["recency"],
            cdnow["T"],
            cdnow["monetary_value"],
        )

        utils_clv = utils._customer_lifetime_value(
            fitted_bgm,
            cdnow["frequency"],
            cdnow["recency"],
            cdnow["T"],
            fitted_ggm._conditional_expected_average_profit(
                cdnow["frequency"], cdnow["monetary_value"]
            ),
        )
        np.testing.assert_equal(ggf_clv.values, utils_clv.values)

        ggf_clv = fitted_ggm._customer_lifetime_value(
            fitted_bgm,
            cdnow["frequency"],
            cdnow["recency"],
            cdnow["T"],
            cdnow["monetary_value"],
            freq="H",
        )

        utils_clv = utils._customer_lifetime_value(
            fitted_bgm,
            cdnow["frequency"],
            cdnow["recency"],
            cdnow["T"],
            fitted_ggm._conditional_expected_average_profit(
                cdnow["frequency"], cdnow["monetary_value"]
            ),
            freq="H",
        )
        npt.assert_equal(ggf_clv.values, utils_clv.values)
    
    def test_quantities_of_interest(self):
        """
        GIVEN the _quantities_of_interest BaseModel attribute,
        WHEN the keys of the '_quantities_of_interest' call dictionary attribute are called,
        THEN they should match the list of expected keys.
        """

        pass

    def test_predict_mean(self,fitted_bgm,cdnow_customers, qoi, instance):
        """
        GIVEN a fitted BetaGeoModel,
        WHEN all four quantities of interest are called via BetaGeoModel.predict() for posterior mean predictions,
        THEN expected output instances and dimensions should be returned.
        """

        pass
    
    def test_predict_full(self,fitted_bgm,cdnow,qoi, instance, draws):
        """
        GIVEN a fitted BetaGeoModel,
        WHEN all four quantities of interest are called via BetaGeoModel.predict() for full posterior predictions,
        THEN expected output instances and dimensions should be returned.
        """

        pass

    def test_save(self, fitted_ggm, filename):
        """
        GIVEN a fitted GammaGammaModel object,
        WHEN self.save_model() is called,
        THEN the external JSON file should exist.
        """

        # Remove saved file if it already exists:
        try:
            os.remove("./ggm.json")
        except FileNotFoundError:
            pass
        finally:
            assert os.path.isfile("./ggm.json") == False

            fitted_ggm.save("./ggm.json")
            assert os.path.isfile("./ggm.json") == True

    @pytest.mark.parametrize("filename", ["./ggm.json", "./bgnbd.json"])
    def test_load(self, fitted_ggm, filename):
        """
        GIVEN fitted and unfitted GammaGammaModel objects,
        WHEN parameters of the fitted model are loaded from an external JSON  via self.load_model(),
        THEN InferenceData unloaded parameters should match, raising exceptions otherwise and if predictions attempted without RFM data.
        """

        ggm_new = btyd.GammaGammaModel()
        ggm_new.load(filename)
        assert isinstance(ggm_new._idata, az.InferenceData)
        #assert bgm_new._idata.posterior.keys() ==  'sample'
        assert ggm_new._unload_params() == fitted_ggm._unload_params()

        # assert param exception (need another saved model and additions to self.load_model())
        # assert prediction exception

        os.remove(filename)  

