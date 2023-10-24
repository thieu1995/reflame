#!/usr/bin/env python
# Created by "Thieu" at 09:52, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import ClassificationMetric, RegressionMetric
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from skorch import NeuralNetRegressor
import torch

from reflame.base_flnn_torch import BaseFlnn, FLNN
from reflame.utils import validator
from reflame.utils.data_toolkit import ObjectiveScaler
from reflame.utils.evaluator import get_all_classification_metrics, get_all_regression_metrics


class FlnnRegressor(BaseFlnn):
    """
    Defines the general class of Metaheuristic-based FLNN model for Regression problems that inherit the BaseMhaFlnn and RegressorMixin classes.

    Parameters
    ----------
    expand_name : str, default="chebyshev"
        The expand function that will be used. The supported expand functions are:
        {"chebyshev", "legendre", "gegenbauer", "laguerre", "hermite", "power", "trigonometric"}

    n_funcs : int, default=4
        The first `n_funcs` in expand functions list will be used. Valid value from 1 to 10.

    act_name : str, default='sigmoid'
        Activation function for the hidden layer. The supported activation functions are:
        {"relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid",
        "swish",  "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink"}

    obj_name : str, default="MSE"
        Current supported objective functions, please check it here: https://github.com/thieu1995/permetrics

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    obj_weights [Optional]: list, tuple, np.ndarray. Default=None
        The objective weights for multiple objective functions

    Examples
    --------
    >>> from reflame import FlnnRegressor, Data
    >>> from sklearn.datasets import make_regression
    >>>
    >>> ## Make dataset
    >>> X, y = make_regression(n_samples=200, n_features=10, random_state=1)
    >>> ## Load data object
    >>> data = Data(X, y)
    >>> ## Split train and test
    >>> data.split_train_test(test_size=0.2, random_state=1, inplace=True)
    >>> ## Scale dataset
    >>> data.X_train, scaler = data.scale(data.X_train, scaling_methods=("minmax"))
    >>> data.X_test = scaler.transform(data.X_test)
    >>> ## Create model
    >>> model = FlnnRegressor(expand_name="chebyshev", n_funcs=4, act_name="elu",
    >>>                          obj_name="MSE", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
    >>> ## Train the model
    >>> model.fit(data.X_train, data.y_train)
    >>> ## Test the model
    >>> y_pred = model.predict(data.X_test)
    >>> ## Calculate some metrics
    >>> print(model.score(X=data.X_test, y=data.y_test, method="RMSE"))
    >>> print(model.scores(X=data.X_test, y=data.y_test, list_methods=["R2", "NSE", "MAPE"]))
    >>> print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["R2", "NSE", "MAPE", "NNSE"]))
    """

    SUPPORTED_LOSSES = {
        "MAE": torch.nn.L1Loss, "MSE": torch.nn.MSELoss
    }

    def __init__(self, expand_name="chebyshev", n_funcs=4, act_name="elu",
                 obj_name="MSE", max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False, **kwargs):
        super().__init__(expand_name=expand_name, n_funcs=n_funcs, act_name=act_name, obj_name=obj_name,
                         max_epochs=max_epochs, batch_size=batch_size, optimizer=optimizer, optimizer_paras=optimizer_paras, verbose=verbose)
        self.kwargs = kwargs

    def create_network(self, X, y):
        """
        Returns
        -------
            network: FLNN, an instance of FLNN network
            obj_scaler: ObjectiveScaler, the objective scaler that used to scale output
        """
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                size_output = 1
            elif y.ndim == 2:
                size_output = y.shape[1]
            else:
                raise TypeError("Invalid y array shape, it should be 1D vector or 2D matrix.")
        else:
            raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")
        obj_scaler = ObjectiveScaler(obj_name="self", ohe_scaler=None)
        verbose = 1 if self.verbose else 0
        network = NeuralNetRegressor(
            module=FLNN,
            module__size_input=X.shape[1],
            module__size_output=size_output,
            module__expand_name=self.expand_name,
            module__n_funcs=self.n_funcs,
            module__act_name=self.act_name,
            criterion=self.SUPPORTED_LOSSES[self.obj_name],
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            optimizer=getattr(torch.optim, self.optimizer),
            verbose=verbose,
            **self.optimizer_paras, **self.kwargs
        )
        return network, obj_scaler

    def score(self, X, y, method="RMSE"):
        """Return the metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        method : str, default="RMSE"
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._BaseFlnn__score_reg(X, y, method)

    def scores(self, X, y, list_methods=("MSE", "MAE")):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_methods : list, default=("MSE", "MAE")
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseFlnn__scores_reg(X, y, list_methods)

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """Return the list of performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list, default=("MSE", "MAE")
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseFlnn__evaluate_reg(y_true, y_pred, list_metrics)

