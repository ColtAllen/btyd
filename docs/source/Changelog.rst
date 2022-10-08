Changelog
=========


0.1beta2
~~~~~~~~
 - ``GammaGammaModel`` Bayesian implementation of Gamma-Gamma model added.
 - Input validation added to ``BaseModel`` class.
 - Fixed array broadcasting bug in ``BetaGeoModel``.
 - Revised frequency and monetary value descriptions in User Guide.
 - Revised SQL code for monetary value calculations in documentation.
 - Added required dependencies to ``setup.cfg``.

0.1beta1
~~~~~~~~~
 - Bayesian predictions now supported, enabling entire probability distributions as well as point estimates for predictive outputs.
 - Streamlined user API to minimize input arguments. All predictive methods are now also called from a single function.
 - ``model._idata`` attribute now persisted as an ``arviz.InferenceData`` object, and can be saved externally in JSON or CSV format. However, only JSONs can be loaded as of this release.
 - Documentation updated to latest versions of ``sphinx`` and ``pydata-sphinx-theme``.
 - Removed *High Level Overview* from documentation.
 - Added deprecation warning for legacy Lifetimes `fitters` module.
 - Removed extraneous ``lifetimes`` import causing build issues.
 - Removed ``psutils`` library dependency.
 - Added ``numpy >=1.20.0`` library dependency.
 - ``utils.posterior_predictive_deviation`` metric removed pending further evaluation.
 - CI/CD pre-commit scripts added.

.. _section-1:

0.1alpha1
~~~~~~~~~~~
 - Forked ``lifetimes`` library v0.11.3 and rebranded as ``btyd``.
 - ``BetaGeoCovarsFitter`` BG/NBD model with time-invariant covariates added to ``fitters`` module.
 - Alpha version of new modeling backend created in `models` module to support Bayesian modeling via ``pymc``.
 - ``BetaGeoModel`` Bayesian implementation of BG/NBD model added.
 - Switched to Apache 2.0 license.
 - New experimental ``posterior_predictive_deviation`` metric added.

.. _section-2:
