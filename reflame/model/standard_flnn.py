#!/usr/bin/env python
# Created by "Thieu" at 09:52, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Tuple
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from skorch import NeuralNetRegressor, NeuralNetClassifier
from reflame.base_flnn_torch import BaseFlnn, FLNN
from reflame.utils.data_toolkit import ObjectiveScaler


class FlnnRegressor(BaseFlnn):
    """
    Defines the class for traditional FLNN network for Regression problems that inherit the BaseFlnn and RegressorMixin classes.

    Parameters
    ----------
    expand_name : str, default="chebyshev"
        The expand function that will be used. The supported expand functions are:
        {"chebyshev", "legendre", "gegenbauer", "laguerre", "hermite", "power", "trigonometric"}

    n_funcs : int, default=4
        The first `n_funcs` in expand functions list will be used. Valid value from 1 to 10.

    act_name : {"relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid",
        "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink",
        "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax" }, default='sigmoid'
        Activation function for the hidden layer.

    obj_name : str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.

    max_epochs : int, default=1000
        Maximum number of epochs / iterations / generations

    batch_size : int, default=32
        The batch size

    optimizer : str, default = "SGD"
        The gradient-based optimizer from Pytorch. List of supported optimizer is:
        ["Adadelta", "Adagrad", "Adam", "Adamax", "AdamW", "ASGD", "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD"]

    optimizer_paras : dict or None, default=None
        The dictionary parameters of the selected optimizer.

    verbose : bool, default=True
        Whether to print progress messages to stdout.

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


class FlnnClassifier(BaseFlnn):
    """
    Defines the class for traditional FLNN network for Classification problems that inherit the BaseFlnn and ClassifierMixin classes.

    Parameters
    ----------
    expand_name : str, default="chebyshev"
        The expand function that will be used. The supported expand functions are:
        {"chebyshev", "legendre", "gegenbauer", "laguerre", "hermite", "power", "trigonometric"}

    n_funcs : int, default=4
        The first `n_funcs` in expand functions list will be used. Valid value from 1 to 10.

    act_name : {"relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid",
        "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink",
        "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax" }, default='sigmoid'
        Activation function for the hidden layer.

    obj_name : str, default=NLLL
        The name of objective for classification problem (binary and multi-class classification)

    max_epochs : int, default=1000
        Maximum number of epochs / iterations / generations

    batch_size : int, default=32
        The batch size

    optimizer : str, default = "SGD"
        The gradient-based optimizer from Pytorch. List of supported optimizer is:
        ["Adadelta", "Adagrad", "Adam", "Adamax", "AdamW", "ASGD", "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD"]

    optimizer_paras : dict or None, default=None
        The dictionary parameters of the selected optimizer.

    verbose : bool, default=True
        Whether to print progress messages to stdout.

    Examples
    --------
    >>> from reflame import FlnnClassifier, Data
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
    >>> model = FlnnClassifier(expand_name="chebyshev", n_funcs=4, act_name="elu",
    >>>                          obj_name="CEL", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
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
        "NLLL": torch.nn.NLLLoss, "PNLLL": torch.nn.PoissonNLLLoss, "GNLLL": torch.nn.GaussianNLLLoss, "KLDL": torch.nn.KLDivLoss,
        "HEL": torch.nn.HingeEmbeddingLoss, "BCEL": torch.nn.BCELoss, "BCELL": torch.nn.BCEWithLogitsLoss,
        "CEL": torch.nn.CrossEntropyLoss
    }
    CLS_OBJ_LOSSES = ["CEL", "HEL", "KLDL"]
    CLS_OBJ_BINARY_1 = ["PNLLL", "HEL", "BCEL", "CEL", "BCELL"]
    CLS_OBJ_BINARY_2 = ["NLLL"]
    CLS_OBJ_MULTI = ["NLLL", "CEL"]

    def __init__(self, expand_name="chebyshev", n_funcs=4, act_name="relu",
                 obj_name="NLLL", max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False, **kwargs):
        super().__init__(expand_name=expand_name, n_funcs=n_funcs, act_name=act_name, obj_name=obj_name,
                         max_epochs=max_epochs, batch_size=batch_size, optimizer=optimizer, optimizer_paras=optimizer_paras, verbose=verbose)
        self.kwargs = kwargs
        self.is_binary = True

    def create_network(self, X, y) -> Tuple[NeuralNetClassifier, ObjectiveScaler]:
        """
        Returns
        -------
            network: FLNN, an instance of FLNN network
            obj_scaler: ObjectiveScaler, the objective scaler that used to scale output
        """
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim != 1:
                raise TypeError("Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on.")
        else:
            raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")
        self.n_labels = len(np.unique(y))
        if self.n_labels > 2:
            self.is_binary = False
            if self.obj_name in self.CLS_OBJ_LOSSES:
                self.return_prob = True
        if self.is_binary:
            obj_scaler = None
            if self.obj_name in self.CLS_OBJ_BINARY_1:
                size_output = 1
            elif self.obj_name in self.CLS_OBJ_BINARY_2:
                size_output = 2
            else:
                raise ValueError(f"Invalid obj_name. For binary classification problem, obj_name has to be one of {self.CLS_OBJ_BINARY_1 + self.CLS_OBJ_BINARY_2}")
        else:
            size_output = self.n_labels
            if self.obj_name in self.CLS_OBJ_MULTI:
                ohe_scaler = OneHotEncoder(sparse=False)
                ohe_scaler.fit(np.reshape(y, (-1, 1)))
                obj_scaler = ObjectiveScaler(obj_name="softmax", ohe_scaler=ohe_scaler)
            else:
                raise ValueError(f"Invalid obj_name. For multi-class classification problem, obj_name has to be one of {self.CLS_OBJ_MULTI}.")

        verbose = 1 if self.verbose else 0
        network = NeuralNetClassifier(
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

    def fit(self, X, y):
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        self.network, self.obj_scaler = self.create_network(X, y)
        if self.is_binary:
            y = y.astype(np.float32)
            if self.obj_name in ("NLLL", ):
                y = y.astype(np.int64)
            if self.obj_name in ("CEL", "BCEL", "BCELL"):
                y = y.reshape((-1, 1))
        else:
            y = y.astype(np.int64)
        self.network.fit(X=X, y=y)
        return self

    def score(self, X, y, method="AS"):
        """
        Return the metric on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric
        since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        method : str, default="AS"
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._BaseFlnn__score_cls(X, y, method)

    def scores(self, X, y, list_methods=("AS", "RS")):
        """
        Return the list of metrics on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric
        since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        list_methods : list, default=("AS", "RS")
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseFlnn__scores_cls(X, y, list_methods)

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Return the list of performance metrics on the given test data and labels.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list, default=("AS", "RS")
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseFlnn__evaluate_cls(y_true, y_pred, list_metrics)
