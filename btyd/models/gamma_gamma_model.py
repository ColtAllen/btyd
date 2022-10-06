from __future__ import generator_stop
from __future__ import annotations
import warnings

from typing import Union, Tuple, Dict

import pandas as pd
import numpy as np
import numpy.typing as npt

import pymc as pm
import aesara.tensor as at

from scipy.special import gammaln, beta, gamma
from scipy.special import hyp2f1
from scipy.special import expit

from . import BaseModel
from ..utils import _customer_lifetime_value


class GammaGammaModel(BaseModel["GammaGammaModel"]):
    """
    The Gamma-Gamma model is used to estimate the average monetary value of customer transactions.

    This implementation is based on the Excel spreadsheet found in [3]_.
    More details on the derivation and evaluation can be found in [4]_.

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
    .. [3] http://www.brucehardie.com/notes/025/
       The Gamma-Gamma Model of Monetary Value.
    .. [4] Peter S. Fader, Bruce G. S. Hardie, and Ka Lok Lee (2005),
       "RFM and CLV: Using iso-value curves for customer base analysis",
       Journal of Marketing Research, 42 (November), 415-430.

    Attributes
    -----------
    penalizer_coef: float
        The coefficient applied to an l2 norm on the parameters
    params_: :obj: Series
        The fitted parameters of the model
    data: :obj: DataFrame
        A DataFrame with the values given in the call to `fit`
    variance_matrix_: :obj: DataFrame
        A DataFrame with the variance matrix of the parameters.
    confidence_intervals_: :obj: DataFrame
        A DataFrame 95% confidence intervals of the parameters
    standard_errors_: :obj: Series
        A Series with the standard errors of the parameters
    summary: :obj: DataFrame
        A DataFrame containing information about the fitted parameters
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
                "p_prior_alpha": 1.,
                "p_prior_beta": 1.,
                "q_prior_alpha": 1.,
                "q_prior_beta": 3,
                "v_prior_alpha": 1.,
                "v_prior_beta": 3.5,
            }
        else:
            self._hyperparams = hyperparams

    _param_list = ["p", "q", "v"]

    def _model(self) -> pm.Model():
        """
        Hierarchical Bayesian model to estimate model parameters.
        This is an internal method and not intended to be called directly.

        Returns
        -------
        self.model: pymc.Model
            Compiled probabilistic PyMC model to estimate model parameters.
        """

        with pm.Model(name=f"{self.__class__.__name__}") as self.model:
            # Priors for lambda parameters.
            p_prior = pm.Weibull(
                name="p",
                alpha=self._hyperparams.get("p_prior_alpha"),
                beta=self._hyperparams.get("p_prior_beta"),
            )
            q_prior = pm.Weibull(
                name="q",
                alpha=self._hyperparams.get("q_prior_alpha"),
                beta=self._hyperparams.get("q_prior_beta"),
            )
            v_prior = pm.Weibull(
                name="v",
                alpha=self._hyperparams.get("v_prior_alpha"),
                beta=self._hyperparams.get("v_prior_beta"),
            )

            logp = pm.Potential(
                "loglike",
                self._log_likelihood(
                    self._frequency, self._monetary_value, p_prior, q_prior, v_prior
                ),
            )

        return self.model

    def _log_likelihood(
        self,
        frequency: npt.ArrayLike,
        monetary_value: npt.ArrayLike,
        p: at.var.TensorVariable,
        q: at.var.TensorVariable,
        v: at.var.TensorVariable,
    ) -> at.var.TensorVariable:
        """

        Log-likelihood function to estimate model parameters for entire population of customers. Equivalent to equation (1a) on page 2 of http://www.brucehardie.com/notes/025/.

        This is an internal method and not intended to be called directly.

        Parameters
        ----------
        frequency: numpy.ndarray
            Numpy array containing total number of transactions for each customer.
        monetary_value: numpy.ndarray
            Numpy array containing average spend per customer.
        p: aesara TensorVariable
            Tensor for 'p' shape parameter of Gamma distribution.
        q: aesara TensorVariable
            Tensor for 'q' shape parameter of Gamma distribution.
        v: aesara TensorVariable
            Tensor for 'v' shape parameter of Gamma distribution.

        Returns
        ----------
        loglike: aesara TensorVariable
            Log-likelihood value for self._model().

        """

        # Recast inputs as Aesara tensor variables
        x = at.as_tensor_variable(frequency)
        m = at.as_tensor_variable(monetary_value)

        loglike = (
            at.gammaln(p * x + q)
            - at.gammaln(p * x)
            - at.gammaln(q)
            + q * at.log(v)
            + (p * x - 1) * at.log(m)
            + (p * x) * at.log(x)
            - (p * x + q) * at.log(x * m + v)
        )

        return loglike

    def predict(
        self,
        method: str,
        rfm_df: pd.DataFrame = None,
        sample_posterior: bool = False,
        posterior_draws: int = 100,
        join_df=False,
        transaction_prediction_model: btyd.Model = None,
        time: int = 12,
        discount_rate: float = 0.01,
        freq: str = "D",
    ) -> np.ndarray:

        """
        Base method for running Gamma-Gamma model predictions.

        Parameters
        ----------
        method: str
            Predictive quantity of interest; accepts 'avg_value' or 'clv'.
        rfm_df: pandas.DataFrame
            Dataframe containing recency, frequency, monetary value, and time period columns.
        sample_posterior: bool
            Flag for sampling from parameter posteriors. Set to 'True' to return predictive probability distributions instead of point estimates.
        posterior_draws: int
            Number of draws from parameter posteriors.
        join_df: bool
            NOT SUPPORTED IN 0.1beta2. Flag to add columns to rfm_df containing predictive outputs.
        transaction_prediction_model: btyd.models
            the model to predict future transactions, literature uses pareto/ndb models but we can also use a different model like beta-geo models
        time: float, optional
            the lifetime expected for the user in months. Default: 12
        discount_rate: float, optional
            the monthly adjusted discount rate. Default: 0.01
        freq: string, optional
            {"D", "H", "M", "W"} for day, hour, month, week. This represents what unit of time your T is measured in.

        Returns
        -------
        predictions: np.ndarray
            Numpy arrays containing predictive quantities of interest.

        """

        # TODO: Will need to resolve disparate data attributes of transaction_prediction_model before making rfm_df optional.
        if rfm_df is not None:
            (
                self._frequency,
                self._recency,
                self._T,
                self._monetary_value,
                _,
            ) = self._dataframe_parser(rfm_df)

        # TODO: Add exception handling for method argument.
        predictions = self._quantities_of_interest.get(method)(
            self, sample_posterior, posterior_draws, transaction_prediction_model, time, discount_rate, freq
        )

        # TODO: After v0.1.0, add arg to automatically merge to RFM dataframe.
        if join_df:
            pass

        # TODO: Additional columns will need to be added for mean, confidence intervals, etc.
        if sample_posterior:
            pass

        return predictions

    def _conditional_expected_average_profit(
        self,
        sample_posterior: bool = False,
        posterior_draws: int = None,
        transaction_prediction_model: btyd.models = None,
        time: int = None,
        discount_rate: float = None,
        freq: str = None,
        frequency: npt.ArrayLike = None, 
        monetary_value: npt.ArrayLike = None,
    ) -> np.ndarray:
        """
        Conditional expectation of the average profit.

        This method computes the conditional expectation of the average profit
        per transaction for a group of one or more customers.

        Equation (5) from:
        http://www.brucehardie.com/notes/025/

        This is an internal method and not intended to be called directly.

        Parameters
        ----------
        frequency: array_like, optional
            a vector containing the customers' frequencies.
            Defaults to the whole set of frequencies used for fitting the model.
        monetary_value: array_like, optional
            a vector containing the customers' monetary values.
            Defaults to the whole set of monetary values used for
            fitting the model.

        Returns
        -------
        array_like:
            The conditional expectation of the average profit per transaction
        """

        # To get rid of these arguments and IF statements, the pertinent unit test must be refactored.
        if frequency is None:
            frequency = self._frequency
        if monetary_value is None:
            monetary_value = self._monetary_value

        param_arrays = self._unload_params(sample_posterior, posterior_draws) 

        if not sample_posterior:
            param_arrays = [
                np.array(_param).reshape(
                    1,
                )
                for _param in param_arrays
            ]

        avg_profit_array = []

        for p, q, v in zip(
            param_arrays[0], param_arrays[1], param_arrays[2]
        ):

            # The expected average profit is a weighted average of individual
            # monetary value and the population mean.
            individual_weight = p * frequency / (p * frequency + q - 1)
            population_mean = v * p / (q - 1)

            avg_profit =  (1 - individual_weight) * population_mean + individual_weight * monetary_value
            avg_profit_array.append(avg_profit)

        return np.array(avg_profit_array)

    def _customer_lifetime_value(
        self,
        sample_posterior: bool = False,
        posterior_draws: int = 100,
        transaction_prediction_model: btyd.models.BaseModel = None,
        time: int = 12,
        discount_rate: float = 0.01,
        freq: str = "D",
        frequency: npt.ArrayLike = None,
        monetary_value: npt.ArrayLike = None,
        recency: npt.ArrayLike = None,
        T: npt.ArrayLike = None,
    ) -> np.ndarray:
        """
        Return customer lifetime value.

        This method computes the average lifetime value for a group of one
        or more customers.

        This is an internal method and not intended to be called directly.

        Parameters
        ----------
        transaction_prediction_model: btyd.models
            the model to predict future transactions, literature uses
            pareto/ndb models but we can also use a different model like beta-geo models
        time: float, optional
            the lifetime expected for the user in months. Default: 12
        discount_rate: float, optional
            the monthly adjusted discount rate. Default: 0.01
        freq: string, optional
            {"D", "H", "M", "W"} for day, hour, month, week. This represents what unit of time your T is measure in.
        frequency: array_like
            the frequency vector of customers' purchases
            (denoted x in literature).
        monetary_value: array_like
            the monetary value vector of customer's purchases
            (denoted m in literature).
        recency: the recency vector of customers' purchases
                 (denoted t_x in literature).
        T: array_like
            customers' age (time units since first purchase)

        Returns
        -------
        Series:
            Series object with customer ids as index and the estimated customer
            lifetime values as values
        """

        # To get rid of these arguments and IF statements, the pertinent unit test must be refactored.
        if frequency is None:
            frequency = self._frequency
        if monetary_value is None:
            monetary_value = self._monetary_value
        if recency is None:
            recency = self._recency
        if T is None:
            T = self._T

        # use the Gamma-Gamma estimates for the monetary_values
        adjusted_monetary_value = self._conditional_expected_average_profit(
            sample_posterior = sample_posterior, posterior_draws = posterior_draws, frequency = frequency, monetary_value = monetary_value
        )

        if sample_posterior:
            clv = np.zeros((posterior_draws,len(frequency))) # initialize clv as zero
        else:
            clv = np.zeros((1,len(frequency)))

        steps = np.arange(1, time + 1)
        factor = {"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24}[freq]

        for i in steps * factor:
            # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
            expected_number_of_transactions = transaction_prediction_model._conditional_expected_number_of_purchases_up_to_time(
                t = i, frequency = frequency, recency = recency, T = T
            ) - transaction_prediction_model._conditional_expected_number_of_purchases_up_to_time(t = i - factor, frequency = frequency, recency = recency, T = T)
            # sum up the CLV estimates of all of the periods and apply discounted cash flow
            clv = np.add(clv,(adjusted_monetary_value * expected_number_of_transactions) / (
                1 + discount_rate
            ) ** (i / factor), out = clv)

        return clv  # return as a series

    _quantities_of_interest = {
        "avg_value": _conditional_expected_average_profit,
        "clv": _customer_lifetime_value,
    }

    def generate_rfm_data(self):
        """
        Not currently supported for GammaGammaModel.
        """
        pass
