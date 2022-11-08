from __future__ import generator_stop
from __future__ import annotations
import warnings

from typing import Union, Tuple, Dict

import pandas as pd
import numpy as np
import numpy.typing as npt

import pymc as pm
import aesara.tensor as at

from scipy.special import beta, gamma
from scipy.special import hyp2f1

from btyd import BetaGeoModel
from ..generate_data import modified_beta_geometric_nbd_model


class ModBetaGeoModel(BetaGeoModel["ModBetaGeoModel"]):
    r"""
    Also known as the MBG/NBD model.

    Based on [5]_, [6]_, this model has the following assumptions:
    1) Each individual, ``i``, has a hidden ``lambda_i`` and ``p_i`` parameter
    2) These come from a population wide Gamma and a Beta distribution
       respectively.
    3) Individuals purchases follow a Poisson process with rate :math:`\lambda_i*t` .
    4) At the beginning of their lifetime and after each purchase, an
       individual has a p_i probability of dieing (never buying again).

    Parameters
    ----------
    hyperparams: dict
        Dictionary containing hyperparameters for model prior parameter distributions.

    Attributes
    ----------
    _hyperparams: dict
        Hyperparameters of prior parameter distributions for model fitting.
    _param_list: list
        List of estimated model parameters.
    _model: pymc.Model
        Hierarchical Bayesian model to estimate model parameters.
    _idata: ArViZ.InferenceData
        InferenceData object of fitted or loaded model. Used for predictions as well as evaluation plots, and model metrics via the ArViZ library.

    References
    ----------
    .. [5] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
       "Empirical validation and comparison of models for customer base
       analysis,"
       International Journal of Research in Marketing, 24 (3), 201-209.
    .. [6] Wagner, U. and Hoppe D. (2008), "Erratum on the MBG/NBD Model,"
       International Journal of Research in Marketing, 25 (3), 225-226.

    """

    def __init__(self, hyperparams: Dict[float] = None) -> SELF:
        """
        Instantiate new model with custom hyperparameters if desired.

        Parameters
        ----------
        hyperparams: dict
            Dict containing model hyperparameters for parameter prior probability distributions.

        """

        if hyperparams is None:
            self._hyperparams = {
                "alpha_prior_alpha": 1.0,
                "alpha_prior_beta": 1.7,
                "r_prior_alpha": 1.0,
                "r_prior_beta": 1.0,
                "phi_prior_lower": 0.0,
                "phi_prior_upper": 1.0,
                "kappa_prior_alpha": 1.5,
                "kappa_prior_m": 1.0,
            }
        else:
            self._hyperparams = hyperparams

    _param_list = ["alpha", "r", "a", "b"]

    def _model(self) -> pm.Model():
        """
        Hierarchical Bayesian model to estimate model parameters.
        This is an internal method and not intended to be called directly.

        Returns
        -------
        self.model: pymc.Model
            Compiled probabilistic PyMC model to estimate model parameters.
        """

        # Call the parent BetaGeoModel method with the ModBetaGeoModel log-likelihood:
        return super(ModBetaGeoModel, self)._model()

    def _log_likelihood(
        self,
        frequency: npt.ArrayLike,
        recency: npt.ArrayLike,
        T: npt.ArrayLike,
        a: at.var.TensorVariable,
        b: at.var.TensorVariable,
        alpha: at.var.TensorVariable,
        r: at.var.TensorVariable,
    ) -> Union[Tuple[at.var.TensorVariable], at.var.TensorVariable]:
        """

        Log-likelihood function to estimate model parameters for entire population of customers.

        This is an internal method and not intended to be called directly.

        Parameters
        ----------
        frequency: numpy.ndarray
            Numpy array containing total number of transactions for each customer.
        recency: numpy.ndarray
            Numpy array containing recency (i.e., time periods between the first and last transaction) for each customer.
        T: numpy.ndarray
            Numpy array containing total time periods for each customer.
        a: aesara TensorVariable
            Tensor for 'a' shape parameter of Beta distribution.
        b: aesara TensorVariable
            Tensor for 'b' shape parameter of Beta distribution.
        alpha: aesara TensorVariable
            Tensor for 'beta' shape parameter of Gamma distribution. (Confusing, but the term alpha is used in the literature.)
        r: aesara TensorVariable
            Tensor for 'alpha' rate parameter of Gamma distribution.

        Returns
        ----------
        loglike: aesara TensorVariable
            Log-likelihood value for self._model().

        """

        # Recast inputs as Aesara tensor variables
        freq = at.as_tensor_variable(frequency)
        rec = at.as_tensor_variable(recency)
        T = at.as_tensor_variable(T)

        A_1 = at.gammaln(r + freq) - at.gammaln(r) + r * at.log(alpha)
        A_2 = (
            at.gammaln(a + b)
            + at.gammaln(b + freq + 1)
            - at.gammaln(b)
            - at.gammaln(a + b + freq + 1)
        )
        A_3 = -(r + freq) * at.log(alpha + T)
        A_4 = (
            at.log(a)
            - at.log(b + freq)
            + (r + freq) * (at.log(alpha + T) - at.log(alpha + rec))
        )

        loglike = A_1 + A_2 + A_3 + at.logaddexp(A_4, 0)

        return loglike

    def _conditional_expected_number_of_purchases_up_to_time(
        self,
        t: npt.ArrayLike = None,
        n: int = None,
        posterior: bool = False,
        posterior_draws: int = 100,
        frequency: npt.ArrayLike = None,
        recency: npt.ArrayLike = None,
        T: npt.ArrayLike = None,
    ) -> np.ndarray:
        """
        Conditional expected number of repeat purchases up to time t.

        Calculate the expected number of repeat purchases up to time t for a
        randomly choose individual from the population, given they have
        purchase history (frequency, recency, T)
        See Wagner, U. and Hoppe D. (2008).

        This is an internal method and not intended to be called directly.

        Parameters
        ----------
        t: numpy.ndarray
            Array containing times to calculate the expectation for each customer.
        n: int
            NOT USED FOR THIS METHOD. Retained for predictive API consistency only.
        posterior: bool
            Flag to sample from parameter posteriors. Set to True to return predictions as probability distributions.
        posterior_draws: int
            Number of draws from parameter posterior distributions.
        frequency: numpy.ndarray
            Numpy array containing total number of transactions for each customer.
        recency: numpy.ndarray
            Numpy array containing recency (i.e., time periods between the first and last transaction) for each customer.
        T: numpy.ndarray
            Numpy array containing total time periods for each customer.

        Returns
        -------
        cond_n_prchs_to_time: numpy.ndarray
            Point estimates or probability distributions for each customer, dependencing on 'posterior' being set to False or True, respectively.

        """

        # To get rid of these arguments and IF statements, the pertinent unit test must be refactored.
        if frequency is None:
            x = self._frequency
        else:
            x = frequency
        if recency is None:
            recency = self._recency
        if T is None:
            T = self._T

        param_arrays = self._unload_params(posterior, posterior_draws)

        if not posterior:
            param_arrays = [
                np.array(_param).reshape(
                    1,
                )
                for _param in param_arrays
            ]

        cond_n_purchases = []

        for alpha, r, a, b in zip(
            param_arrays[0], param_arrays[1], param_arrays[2], param_arrays[3]
        ):
            hyp_term = hyp2f1(r + x, b + x + 1, a + b + x, t / (alpha + T + t))
            first_term = (a + b + x) / (a - 1)
            second_term = 1 - hyp_term * ((alpha + T) / (alpha + t + T)) ** (r + x)
            numerator = first_term * second_term

            denominator = 1 + (a / (b + x)) * ((alpha + T) / (alpha + recency)) ** (
                r + x
            )

            cond_n_purchase = numerator / denominator
            cond_n_purchases.append(cond_n_purchase)

        return np.array(cond_n_purchases)

    def _conditional_probability_alive(
        self,
        t: npt.ArrayLike = None,
        n: int = None,
        posterior: bool = False,
        posterior_draws: int = 100,
        frequency: npt.ArrayLike = None,
        recency: npt.ArrayLike = None,
        T: npt.ArrayLike = None,
    ) -> np.ndarray:
        """
        Conditional probability alive.

        Compute the probability that a customer with history (frequency,
        recency, T) is currently alive.
        From https://www.researchgate.net/publication/247219660_Empirical_validation_and_comparison_of_models_for_customer_base_analysis
        Appendix A, eq. (5)

        This is an internal method and not intended to be called directly.

        Parameters
        ----------
        t: numpy.ndarray
            NOT USED FOR THIS METHOD. Retained for predictive API consistency only.
        n: int
            NOT USED FOR THIS METHOD. Retained for predictive API consistency only.
        posterior: bool
            Flag to sample from parameter posteriors. Set to True to return predictions as probability distributions.
        posterior_draws: int
            Number of draws from parameter posterior distributions.
        frequency: numpy.ndarray
            Numpy array containing total number of transactions for each customer.
        recency: numpy.ndarray
            Numpy array containing recency (i.e., time periods between the first and last transaction) for each customer.
        T: numpy.ndarray
            Numpy array containing total time periods for each customer.

        Returns
        -------
        cond_prob_alive: float or numpy.ndarray
            Point estimates or probability distributions for each customer, dependencing on 'posterior' being set to False or True, respectively.

        """

        # To get rid of these arguments and IF statements, the pertinent unit test must be refactored.
        if frequency is None:
            frequency = self._frequency
        if recency is None:
            recency = self._recency
        if T is None:
            T = self._T

        param_arrays = self._unload_params(posterior, posterior_draws)

        if not posterior:
            param_arrays = [
                np.array(_param).reshape(
                    1,
                )
                for _param in param_arrays
            ]

        cond_p_alive = []

        for alpha, r, a, b in zip(
            param_arrays[0], param_arrays[1], param_arrays[2], param_arrays[3]
        ):
            cond_alive = np.atleast_1d(
                1.0
                / (
                    1
                    + (a / (b + frequency))
                    * ((alpha + T) / (alpha + recency)) ** (r + frequency)
                )
            )
            cond_p_alive.append(cond_alive)

        return np.array(cond_p_alive)

    def _expected_number_of_purchases_up_to_time(
        self,
        t: npt.ArrayLike = None,
        n: int = None,
        posterior: bool = False,
        posterior_draws: int = 100,
    ) -> Union[float, np.ndarray]:
        """
        Return expected number of repeat purchases up to time t.

        Calculate the expected number of repeat purchases up to time t for a
        randomly choose individual from the population.

        This is an internal method and not intended to be called directly.

        Parameters
        ----------
        t: numpy.ndarray
            Array containing times to calculate the expectation of the customer population.
        n: int
            NOT USED FOR THIS METHOD. Retained for predictive API consistency only.
        posterior: bool
            Flag to sample from parameter posteriors. Set to True to return predictions as probability distributions.
        posterior_draws: int
            Number of draws from parameter posterior distributions.

        Returns
        -------
        n_prchs_to_time: float or numpy.ndarray
            Point estimate or probability distribution for customer population, contingent on 'posterior' being set to False or True, respectively.

        """

        param_arrays = self._unload_params(posterior, posterior_draws)

        if not posterior:
            param_arrays = [
                np.array(_param).reshape(
                    1,
                )
                for _param in param_arrays
            ]

        expected_purchases = []

        for alpha, r, a, b in zip(
            param_arrays[0], param_arrays[1], param_arrays[2], param_arrays[3]
        ):

            hyp = hyp2f1(r, b + 1, a + b, t / (alpha + t))

            expected_purchase = b / (a - 1) * (1 - hyp * (alpha / (alpha + t)) ** r)
            expected_purchases.append(expected_purchase)

        return np.array(expected_purchases)

    def _probability_of_n_purchases_up_to_time(
        self,
        t: float = None,
        n: int = None,
        posterior: bool = False,
        posterior_draws: int = 100,
    ) -> Union[np.ndarray, float]:
        r"""
        Compute the probability of n purchases up to time t.

        .. math::  P( N(t) = n | \text{model} )

        where N(t) is the number of repeat purchases a customer makes in t
        units of time.

        This is an internal method and not intended to be called directly.

        Parameters
        ----------
        t: float
            Times to calculate the expectation of the customer population.
        n: int
            Number of transactions expected of the customer population.
        posterior: bool
            Flag to sample from parameter posteriors. Set to True to return predictions as probability distributions.
        posterior_draws: int
            Number of draws from parameter posterior distributions.
        posterior_draws: int
            Number of draws from parameter posterior distributions.

        Returns
        -------
        prob_n_prchs_to_time: float or numpy.ndarray
            Point estimate or probability distribution for customer population, contingent on 'posterior' being set to False or True, respectively.
        """

        param_arrays = self._unload_params(posterior, posterior_draws)

        if not posterior:
            param_arrays = [
                np.array(_param).reshape(
                    1,
                )
                for _param in param_arrays
            ]

        prob_n_purchases = []

        for alpha, r, a, b in zip(
            param_arrays[0], param_arrays[1], param_arrays[2], param_arrays[3]
        ):

            _j = np.arange(0, n)

            first_term = (
                beta(a, b + n + 1)
                / beta(a, b)
                * gamma(r + n)
                / gamma(r)
                / gamma(n + 1)
                * (alpha / (alpha + t)) ** r
                * (t / (alpha + t)) ** n
            )
            finite_sum = (
                gamma(r + _j) / gamma(r) / gamma(_j + 1) * (t / (alpha + t)) ** _j
            ).sum()
            second_term = (
                beta(a + 1, b + n)
                / beta(a, b)
                * (1 - (alpha / (alpha + t)) ** r * finite_sum)
            )

            prob_n_purchase = first_term + second_term
            prob_n_purchases.append(prob_n_purchase)

        return np.array(prob_n_purchases)

    # BETA TODO? This attribute can be removed if the attribute resolution order issue of PredictMixin is resolved.
    _quantities_of_interest = {
        "cond_prob_alive": _conditional_probability_alive,
        "cond_n_prchs_to_time": _conditional_expected_number_of_purchases_up_to_time,
        "n_prchs_to_time": _expected_number_of_purchases_up_to_time,
        "prob_n_prchs_to_time": _probability_of_n_purchases_up_to_time,
    }

    def generate_rfm_data(self, size: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic RFM data from fitted model parameters. Useful for posterior predictive checks of model performance.

        Parameters
        ----------
        size: int
            Rows of synthetic RFM data to generate. Default is 1000.

        Returns
        -------
        self.synthetic_df: pd.DataFrame
            dataframe containing ["frequency", "recency", "T", "lambda", "p", "alive", "customer_id"] columns.

        """

        alpha, r, a, b = self._unload_params()

        self.synthetic_df = modified_beta_geometric_nbd_model(
            self._T, r, alpha, a, b, size=size
        )

        return self.synthetic_df
