#!/usr/bin/env python
# Created by "Thieu" at 13:43, 13/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from permetrics import ClassificationMetric, RegressionMetric
from sklearn.base import BaseEstimator
from reflame.utils import expand_util, validator
from reflame.utils.evaluator import get_all_classification_metrics, get_all_regression_metrics


class FLNN(nn.Module):

    SUPPORTED_EXPANDS = ["chebyshev", "legendre", "gegenbauer", "laguerre", "hermite", "power", "trigonometric"]
    SUPPORTED_N_FUNCS = list(range(1, 11))
    SUPPORTED_ACTIVATIONS = ['threshold', 'relu', 'rrelu', 'hardtanh', 'relu6', 'sigmoid', 'hardsigmoid', 'tanh',
           'silu', 'mish', 'hardswish', 'elu', 'celu', 'selu', 'glu', 'gelu', 'hardshrink', 'leakyrelu',
           'logsigmoid', 'softplus', 'softshrink', 'multiheadattention', 'prelu', 'softsign', 'tanhshrink',
           'softmin', 'softmax', 'logsoftmax']

    def __init__(self, size_input=10, size_output=1, expand_name="chebyshev", n_funcs=4, act_name='elu'):
        super(FLNN, self).__init__()
        self.input_nodes = size_input * n_funcs
        self.output_nodes = size_output
        self.expand_name = expand_name
        self.expand_func = getattr(expand_util, f"expand_{self.expand_name}")
        self.n_funcs = n_funcs
        # Define the activation function
        self.act_name = act_name
        if act_name == "softmax":
            self.act_func = nn.Softmax(dim=0)
        else:
            self.act_func = getattr(nn.functional, self.act_name)
        # Create the output layer
        self.output_layer = nn.Linear(self.input_nodes, self.output_nodes, bias=True)

    def transform_X(self, X):
        return self.expand_func(X, self.n_funcs)

    def forward(self, x):
        # expand input before actual forward pass
        x_input = self.transform_X(x.numpy())
        # actual forward pass
        x = torch.tensor(x_input, dtype=self.output_layer.weight.dtype)
        x = self.act_func(self.output_layer(x))
        return x


class BaseFlnn(BaseEstimator):
    """
    Defines the most general class for FLNN network that inherits the BaseEstimator class of Scikit-Learn library.

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
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    CLS_OBJ_LOSSES = None

    SUPPORTED_LOSSES = {
        "MAE": torch.nn.L1Loss, "MSE": torch.nn.MSELoss
    }
    SUPPORTED_OPTIMIZERS = ["Adadelta", "Adagrad", "Adam", "Adamax", "AdamW", "ASGD",
                            "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD"]

    def __init__(self, expand_name="chebyshev", n_funcs=4, act_name="elu",
                 obj_name=None, max_epochs=1000, batch_size=32,
                 optimizer="SGD", optimizer_paras=None, verbose=False):
        super().__init__()
        self.module = FLNN
        self.expand_name = validator.check_str("expand_name", expand_name, FLNN.SUPPORTED_EXPANDS)
        self.n_funcs = validator.check_int("n_funcs", n_funcs, [FLNN.SUPPORTED_N_FUNCS[0], FLNN.SUPPORTED_N_FUNCS[-1]])
        self.act_name = validator.check_str("act_name", act_name, FLNN.SUPPORTED_ACTIVATIONS)
        self.obj_name = validator.check_str("obj_name", obj_name, list(self.SUPPORTED_LOSSES.keys()))
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = validator.check_str("optimizer", optimizer, self.SUPPORTED_OPTIMIZERS)
        self.optimizer_paras = {} if optimizer_paras is None else optimizer_paras
        self.verbose = verbose

        self.weights = {}
        self.network, self.obj_scaler, self.loss_train = None, None, None
        self.n_labels, self.obj_scaler = None, None

    @staticmethod
    def _check_method(method=None, list_supported_methods=None):
        if type(method) is str:
            return validator.check_str("method", method, list_supported_methods)
        else:
            raise ValueError(f"method should be a string and belongs to {list_supported_methods}")

    def create_network(self, X, y):
        return None, None

    def fit(self, X, y):
        self.network, self.obj_scaler = self.create_network(X, y)
        y_scaled = self.obj_scaler.transform(y)
        X = torch.tensor(X, dtype=torch.float32)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        y_scaled = torch.tensor(y_scaled, dtype=torch.float32)
        self.network.fit(X, y=y_scaled)
        return self

    def predict(self, X, return_prob=False):
        """
        Inherit the predict function from BaseFlnn class, with 1 more parameter `return_prob`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        return_prob : bool, default=False
            It is used for classification problem:

            - If True, the returned results are the probability for each sample
            - If False, the returned results are the predicted labels
        """
        if return_prob:
            return self.network.predict_proba(X)
        else:
            return self.network.predict(X)

    def __evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred, decimal=8)
        return rm.get_metrics_by_list_names(list_metrics)

    def __evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        cm = ClassificationMetric(y_true, y_pred, decimal=8)
        return cm.get_metrics_by_list_names(list_metrics)

    def __score_reg(self, X, y, method="RMSE"):
        """Return the metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        method : str, default="RMSE"
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        method = self._check_method(method, list(self.SUPPORTED_REG_METRICS.keys()))
        y_pred = self.network.predict(X)
        return RegressionMetric(y, y_pred, decimal=6).get_metric_by_name(method)[method]

    def __scores_reg(self, X, y, list_methods=("MSE", "MAE")):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_methods : list, default=("MSE", "MAE")
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        y_pred = self.network.predict(X)
        rm = RegressionMetric(y_true=y, y_pred=y_pred, decimal=6)
        return rm.get_metrics_by_list_names(list_methods)

    def __score_cls(self, X, y, method="AS"):
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
        method = self._check_method(method, list(self.SUPPORTED_CLS_METRICS.keys()))
        return_prob = False
        if self.n_labels > 2:
            if method in self.CLS_OBJ_LOSSES:
                return_prob = True
        y_pred = self.predict(X, return_prob=return_prob)
        cm = ClassificationMetric(y_true=y, y_pred=y_pred, decimal=6)
        return cm.get_metric_by_name(method)[method]

    def __scores_cls(self, X, y, list_methods=("AS", "RS")):
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
        list_errors = list(set(list_methods) & set(self.CLS_OBJ_LOSSES))
        list_scores = list((set(self.SUPPORTED_CLS_METRICS.keys()) - set(self.CLS_OBJ_LOSSES)) & set(list_methods))
        t1 = {}
        if len(list_errors) > 0:
            return_prob = False
            if self.n_labels > 2:
                return_prob = True
            y_pred = self.predict(X, return_prob=return_prob)
            cm = ClassificationMetric(y, y_pred, decimal=6)
            t1 = cm.get_metrics_by_list_names(list_errors)
        y_pred = self.predict(X, return_prob=False)
        cm = ClassificationMetric(y, y_pred, decimal=6)
        t2 = cm.get_metrics_by_list_names(list_scores)
        return {**t2, **t1}

    def evaluate(self, y_true, y_pred, list_metrics=None):
        """Return the list of performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        pass

    def score(self, X, y, method=None):
        """Return the metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        method : str, default="RMSE"
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        pass

    def scores(self, X, y, list_methods=None):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_methods : list, default=("MSE", "MAE")
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        pass

    def save_loss_train(self, save_path="history", filename="loss.csv"):
        """
        Save the loss (convergence) during the training process to csv file.

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if self.loss_train is None:
            print(f"{self.__class__.__name__} model doesn't have training loss!")
        else:
            data = {"epoch": list(range(1, len(self.loss_train) + 1)), "loss": self.loss_train}
            pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
        """
        Save evaluation metrics to csv file

        Parameters
        ----------
        y_true : ground truth data
        y_pred : predicted output
        list_metrics : list of evaluation metrics
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        """
        Save the predicted results to csv file

        Parameters
        ----------
        X : The features data, nd.ndarray
        y_true : The ground truth data
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X, return_prob=False)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="model.pkl"):
        """
        Save model to pickle file

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".pkl" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="model.pkl"):
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))
