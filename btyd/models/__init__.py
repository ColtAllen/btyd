from __future__ import generator_stop
from __future__ import annotations

from abc import ABC, abstractmethod
import warnings
import json
from typing import Union, Tuple, TypeVar, Generic

import numpy as np
import pandas as pd

import pymc as pm
import aesara.tensor as at
import arviz as az


SELF = TypeVar("SELF")


class BaseModel(ABC, Generic[SELF]):
    """
    Abstract class defining all base model methods as well as methods to be overridden when creating a model subclass.
    """

    # These attribute must be defined in model subclasses.
    _quantities_of_interest: dict
    _param_list: list

    @abstractmethod
    def __init__(self) -> SELF:
        """self._param_list must be instantiated here, as well as model hyperpriors."""

    @abstractmethod
    def _model(self) -> None:
        """pymc model defining priors for model parameters and wrapping _log_likelihood in Potential()."""
        pass

    @abstractmethod
    def _log_likelihood(self) -> None:
        """Log-likelihood function for randomly drawn individual from customer population. Must be constructed from aesara tensors."""
        pass

    @abstractmethod
    def generate_rfm_data(self) -> None:
        """Generate synthetic RFM data from parameters of fitted model. This is useful for posterior predictive checks and hyperparameter optimization."""
        pass

    def __repr__(self) -> str:
        """String representation of BTYD model object."""
        classname = self.__class__.__name__
        try:
            row_str = f"estimated with {self._frequency.shape[0]} customers."
        except AttributeError:
            row_str = ""

        try:
            param_vals = np.around(self._unload_params(), decimals=1)
            param_str = str(dict(zip(self._param_list, param_vals)))
            return f"<btyd.{classname}: Parameters {param_str} {row_str}>"
        except AttributeError:
            return f"<btyd.{classname}>"

    def fit(self, rfm_df: pd.DataFrame, tune: int = 1200, draws: int = 1200) -> SELF:
        """
        Fit a custom pymc model with parameter prior definitions to observed RFM data.

        Parameters
        ----------
        rfm_df: pandas.DataFrame
            Pandas dataframe containing customer ids, frequency, recency, T and monetary value columns.
        tune: int
            Number of starting 'burn-in' samples for posterior parameter distribution convergence. These are discarded after the model is fit.
        draws: int
            Number of samples from posterior parameter distrutions after tune period. These are retained for model usage.

        Returns
        -------
        self
            Fitted model with ``_idata`` attribute for model evaluation and predictions.

        """

        (
            self._frequency,
            self._recency,
            self._T,
            self._monetary_value,
            _,
        ) = self._dataframe_parser(rfm_df)

        self._check_inputs(
            self._frequency, self._recency, self._T, self._monetary_value
        )

        with self._model():
            self._idata = pm.sample(
                tune=tune,
                draws=draws,
                chains=4,
                cores=4,
                target_accept=0.95,
                return_inferencedata=True,
            )

        return self

    def _unload_params(
        self, posterior: bool = False, n_samples: int = 100
    ) -> Union[Tuple[np.ndarray], Tuple[np.ndarray]]:
        """
        Extract parameter posteriors from _idata InferenceData attribute of fitted model.
        This is an internal method and not intended to be called directly.

        Parameters
        ----------
        posterior: bool
            Flag for sampling from parameter posterior distribution. Default is set for point estimates.
        n_samples: int

        Returns
        -------
        parameters: tuple(np.ndarray)
            A tuple containing model parameters as numpy arrays.

        """

        # BETA TODO: Raise BTYDException.
        # if dict(filter(lambda item: self.__class__.__name__ not in item[0], self._idata.posterior.get('data_vars').items()))
        # raise BTYDException

        # test param exception (need another saved model and additions to self._unload_params())
        # test prediction exception if attempting to run predictions on instantiated model without RFM data.

        if posterior:
            return tuple(
                [
                    self._sample(
                        self._idata.posterior.get(
                            f"{self.__class__.__name__}::{var}"
                        ).values.flatten(),
                        n_samples,
                    )
                    for var in self._param_list
                ]
            )

        else:
            return tuple(
                [
                    self._idata.posterior.get(f"{self.__class__.__name__}::{var}")
                    .mean()
                    .to_numpy()
                    for var in self._param_list
                ]
            )

    def save_json(self, filename: str) -> None:
        """
        Dump InferenceData from fitted model into a JSON file.

        Parameters
        ----------
        filename: str
            Path and/or filename where model data will be saved.
        """

        if ".json" not in filename:
            raise TypeError("Only JSON file types are supported.")
        else:
            self._idata.to_json(filename)

    def load_json(self, filename: str) -> SELF:
        """
        Load InferenceData from an external file.

        Parameters
        ----------
        filename: str
            Path and/or filename of InferenceData.

        Returns
        -------
        self
            Model object containing ``_idata`` attribute for model evaluation and predictions.
        """

        if ".json" not in filename:
            raise TypeError("Only JSON file types are supported.")
        else:
            self._idata = az.from_json(filename)

            if (
                self.__class__.__name__
                not in list(dict(self._idata.get("posterior")).keys())[0]
            ):
                raise NameError("Incorrect Model Type.")
            else:
                return self

    def _dataframe_parser(self, rfm_df: pd.DataFrame) -> Tuple[np.ndarray]:
        """
        Parse input dataframe into separate RFM components. This is an internal method and not intended to be called directly.

        Parameters
        ----------
        rfm_df: pandas.DataFrame
            Dataframe containing recency, frequency, monetary value, and time period columns.

        Returns
        -------
            Tuple containing numpy arrays for Recency, Frequency, Monetary Value, T, and Customer ID (if provided).
        """

        rfm_df.columns = rfm_df.columns.str.upper()

        # The load_cdnow_summary_with_monetary_value() function needs an ID column for testing.
        if "ID" not in rfm_df.columns:
            customer = rfm_df.index.values
        else:
            customer = rfm_df["ID"].values

        frequency = rfm_df["FREQUENCY"].values
        recency = rfm_df["RECENCY"].values
        T = rfm_df["T"].values

        if self.__class__.__name__ == "GammaGammaModel":
            monetary_value = rfm_df["MONETARY_VALUE"].values
        else:
            monetary_value = None

        return frequency, recency, T, monetary_value, customer

    @staticmethod
    def _sample(param_array: array_like, n_samples: int) -> np.ndarray:
        """
        Utility function for sampling from parameter posteriors. This is an internal method and not intended to be called directly.

        Parameters
        ----------
        param_array: array_like
            Array containing names of model parameters for sampling.
        n_samples: int
            Number of draws from parameter posterior distributions.

        Returns
        -------
            Numpy array of parameter samples for building predictive probability distributions.

        """
        rng = np.random.default_rng()
        return rng.choice(param_array, n_samples, replace=True)

    @staticmethod
    def _check_inputs(
        frequency: array_like,
        recency: array_like = None,
        T: array_like = None,
        monetary_value: array_like = None,
    ) -> None:
        """
        Check validity of inputs.

        Raises ValueError when checks fail.

        The checks go sequentially from recency, to frequency and monetary value:

        - recency > T
        - frequency > T
        - recency[frequency == 0] != 0)
        - recency < 0
        - zero length vector in frequency, recency or T
        - non-integer values in the frequency vector
        - non-positive (<= 0) values in the monetary_value vector for the Gamma-Gamma model
        - non-positive (<= 0) values in the frequency vector for the Gamma-Gamma model

        Parameters
        ----------
        frequency: array_like
            the frequency vector of customers' purchases (denoted x in literature).
        recency: array_like, optional
            the recency vector of customers' purchases (denoted t_x in literature).
        T: array_like, optional
            the vector of customers' age (time since first purchase)
        monetary_value: array_like, optional
            the monetary value vector of customer's purchases (denoted m in literature).
        """

        if recency is not None:
            if T is not None:
                if np.all(recency > T):
                    raise ValueError(
                        "Some values in recency vector are larger than T vector."
                    )
                if np.any(frequency > T):
                    raise ValueError(
                        "Transaction time periods exceed total time periods."
                    )
            if np.any(recency[frequency == 0] != 0):
                raise ValueError(
                    "There exist non-zero recency values when frequency is zero."
                )
            if np.any(recency < 0):
                raise ValueError(
                    "There exist negative recency (ex: last order set before first order)"
                )
            if any(x.shape[0] == 0 for x in [recency, frequency, T]):
                raise ValueError(
                    "There exists a zero length vector in one of frequency, recency or T."
                )
        if np.sum((frequency - frequency.astype(int)) ** 2) != 0:
            raise ValueError("There exist non-integer values in the frequency vector.")
        if monetary_value is not None:
            if np.any(monetary_value <= 0):
                raise ValueError(
                    "There exist non-positive (<= 0) values in the monetary_value vector."
                )
            if np.any(frequency <= 0):
                raise ValueError(
                    "There exist non-positive (<= 0) values in the frequency vector."
                )


class PredictMixin(ABC, Generic[SELF]):
    """
    Abstract class defining predictive methods for all models except GammaGamma.
    In research literature these are commonly referred to as quantities of interest.
    These are internal methods and not intended to be called directly.
    Docstrings for each method are provided in respective model subclass.
    """

    @abstractmethod
    def _conditional_probability_alive(
        self,
        t: float = None,
        n: int = None,
        sample_posterior: bool = False,
        posterior_draws: int = 100,
    ) -> None:
        pass

    @abstractmethod
    def _conditional_expected_number_of_purchases_up_to_time(
        self,
        t: float = None,
        n: int = None,
        sample_posterior: bool = False,
        posterior_draws: int = 100,
    ) -> None:
        pass

    @abstractmethod
    def _expected_number_of_purchases_up_to_time(
        self,
        t: float = None,
        n: int = None,
        sample_posterior: bool = False,
        posterior_draws: int = 100,
    ) -> None:
        pass

    @abstractmethod
    def _probability_of_n_purchases_up_to_time(
        self,
        t: float = None,
        n: int = None,
        sample_posterior: bool = False,
        posterior_draws: int = 100,
    ) -> None:
        pass

    # BETA TODO: this attribute must be re-declared in model subclasses and is retained here for documentation purposes only.
    _quantities_of_interest = {
        "cond_prob_alive": _conditional_probability_alive,
        "cond_n_prchs_to_time": _conditional_expected_number_of_purchases_up_to_time,
        "n_prchs_to_time": _expected_number_of_purchases_up_to_time,
        "prob_n_prchs_to_time": _probability_of_n_purchases_up_to_time,
    }

    def predict(
        self,
        method: str,
        rfm_df: pd.DataFrame = None,
        t: int = None,
        n: int = None,
        sample_posterior: bool = False,
        posterior_draws: int = 100,
        join_df=False,
    ) -> np.ndarray:
        """
        Base method for running model predictions.

        Parameters
        ----------
        method: str
            Predictive quantity of interest; accepts 'cond_prob_alive', 'cond_n_prchs_to_time','n_prchs_to_time', or 'prob_n_prchs_to_time'.
        rfm_df: pandas.DataFrame
            Dataframe containing recency, frequency, monetary value, and time period columns.
        t: int
            Number of time periods for predictions.
        n: int
            Number of transactions predicted.
        sample_posterior: bool
            Flag for sampling from parameter posteriors. Set to 'True' to return predictive probability distributions instead of point estimates.
        posterior_draws: int
            Number of draws from parameter posteriors.
        join_df: bool
            NOT SUPPORTED IN 0.1beta2. Flag to add columns to rfm_df containing predictive outputs.

        Returns
        -------
        predictions: np.ndarray
            Numpy arrays containing predictive quantities of interest.

        """

        if rfm_df is not None:
            (
                self._frequency,
                self._recency,
                self._T,
                self._monetary_value,
                _,
            ) = self._dataframe_parser(rfm_df)

            # Inputs were already checked during model fitting, and only need to be checked again if a new rfm_df is provided.
            self._check_inputs(
                self._frequency, self._recency, self._T, self._monetary_value
            )

        # TODO: Add exception handling for method argument.
        predictions = self._quantities_of_interest.get(method)(
            self, t, n, sample_posterior, posterior_draws
        )

        # TODO: Add arg to automatically merge to RFM dataframe?
        if join_df:
            pass

        if sample_posterior:
            # Additional columns will need to be added for mean, confidence intervals, etc.
            pass

        return predictions
