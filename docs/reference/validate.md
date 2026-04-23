# Validate

Standard sklearn cross-validation splitters leak signal in time-series settings by training on information that a live model wouldn't have had. `PurgedKFold` mitigates this by removing training samples whose labels overlap a given test fold (the "purge" window), and `EmbargoedKFold` extends that with a post-test embargo to cover autocorrelated return labels. Both subclass `sklearn.model_selection.BaseCrossValidator` and can be passed directly as the `cv` argument to `GridSearchCV`, `cross_val_score`, and every other sklearn CV consumer.

::: fundcloud.validate
    options:
      members:
        - PurgedKFold
        - EmbargoedKFold
