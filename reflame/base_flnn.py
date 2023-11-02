#!/usr/bin/env python
# Created by "Thieu" at 13:43, 13/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from permetrics import RegressionMetric, ClassificationMetric
from sklearn.base import BaseEstimator
from mealpy import get_optimizer_by_name, Optimizer, get_all_optimizers, FloatVar
from reflame.utils import activation, validator, expand_util
from reflame.utils.evaluator import get_all_regression_metrics, get_all_classification_metrics


class FLNN:
    """This class defines the general Functional Link Neural Network (FLNN) model

    Parameters
    ----------
    size_input : int, default=5
        The number of input features

    size_output : int, default=1
        The number of output labels

    expand_name : str, default="chebyshev"
        The expand function that will be used. The supported expand functions are:
        {"chebyshev", "legendre", "gegenbauer", "laguerre", "hermite", "power", "trigonometric"}
        
    n_funcs : int, default=4
        The first `n_funcs` in expand functions list will be used. Valid value from 1 to 10.
    
    act_name : str, default='sigmoid'
        Activation function for the hidden layer. The supported activation functions are: 
        {"relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid", 
        "swish",  "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink"}
    """
    def __init__(self, size_input=5, size_output=1, expand_name="chebyshev", n_funcs=4, act_name='elu'):
        self.input_nodes = size_input * n_funcs
        self.output_nodes = size_output
        self.size_w = self.input_nodes * self.output_nodes
        self.size_b = size_output
        self.expand_name = expand_name
        self.expand_func = getattr(expand_util, f"expand_{self.expand_name}")
        self.n_funcs = n_funcs
        self.act_name = act_name
        self.act_func = getattr(activation, self.act_name)
        self.weights = {
            "w": np.random.rand(self.input_nodes, self.output_nodes),
            "b": np.random.rand(self.output_nodes, ),
        }
    
    def transform_X(self, X):
        return self.expand_func(X, self.n_funcs)

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object
            Returns a trained FLNN model.
        """
        # H = self.act_func(np.dot(X, self.weights["w1"]) + self.weights["b"])
        # self.weights["w2"] = np.linalg.pinv(H) @ y
        return self

    def predict(self, X):
        """Predict using the Extreme Learning Machine model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        X = self.transform_X(X)
        y_pred = self.act_func(np.dot(X, self.weights["w"]) + self.weights["b"])
        return y_pred

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_weights_size(self):
        return np.sum([item.size for item in self.weights.values()])

    def update_weights_from_solution(self, solution):
        w = np.reshape(solution[:self.size_w], (self.input_nodes, self.output_nodes))
        b = np.reshape(solution[self.size_w:], self.output_nodes)
        self.set_weights({"w": w, "b": b})


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

    act_name : str, default='sigmoid'
        Activation function for the hidden layer. The supported activation functions are:
        {"relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid",
        "swish",  "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink"}
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    CLS_OBJ_LOSSES = None

    def __init__(self, expand_name="chebyshev", n_funcs=4, act_name="elu"):
        super().__init__()
        self.expand_name = expand_name
        self.n_funcs = n_funcs
        self.act_name = act_name
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
        self.network.fit(X, y_scaled)
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
        pred = self.network.predict(X)
        if return_prob:
            return pred
        return self.obj_scaler.inverse_transform(pred)

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


class BaseMhaFlnn(BaseFlnn):
    """
    Defines the most general class for Metaheuristic-based FLNN model that inherits the BaseFlnn class

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

    obj_name : None or str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    verbose : bool, default=True
        Whether to print progress messages to stdout.
    """

    SUPPORTED_OPTIMIZERS = list(get_all_optimizers().keys())
    SUPPORTED_CLS_OBJECTIVES = get_all_classification_metrics()
    SUPPORTED_REG_OBJECTIVES = get_all_regression_metrics()

    def __init__(self, expand_name="chebyshev", n_funcs=4, act_name="elu", obj_name=None, optimizer="BaseGA", optimizer_paras=None, verbose=True):
        super().__init__(expand_name=expand_name, n_funcs=n_funcs, act_name=act_name)
        self.obj_name = obj_name
        self.optimizer_paras = optimizer_paras
        self.optimizer = self._set_optimizer(optimizer, optimizer_paras)
        self.verbose = verbose
        self.network, self.obj_scaler = None, None
        self.obj_weights = None

    def _set_optimizer(self, optimizer=None, optimizer_paras=None):
        if type(optimizer) is str:
            opt_class = get_optimizer_by_name(optimizer)
            if type(optimizer_paras) is dict:
                return opt_class(**optimizer_paras)
            else:
                return opt_class(epoch=500, pop_size=50)
        elif isinstance(optimizer, Optimizer):
            if type(optimizer_paras) is dict:
                return optimizer.set_parameters(optimizer_paras)
            return optimizer
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def _get_history_loss(self, optimizer=None):
        list_global_best = optimizer.history.list_global_best
        # 2D array / matrix 2D
        global_obj_list = np.array([agent.target.fitness for agent in list_global_best])
        # Make each obj_list as an element in array for drawing
        return global_obj_list

    def objective_function(self, solution=None):
        pass

    def fit(self, X, y, lb=(-10.0, ), ub=(10.0, ), save_population=False):
        self.network, self.obj_scaler = self.create_network(X, y)
        y_scaled = self.obj_scaler.transform(y)
        self.X_temp, self.y_temp = X, y_scaled
        problem_size = self.network.get_weights_size()
        if type(lb) in (list, tuple, np.ndarray) and type(ub) in (list, tuple, np.ndarray):
            if len(lb) == len(ub):
                if len(lb) == 1:
                    lb = np.array(lb * problem_size, dtype=float)
                    ub = np.array(ub * problem_size, dtype=float)
                elif len(lb) != problem_size:
                    raise ValueError(f"Invalid lb and ub. Their length should be equal to 1 or problem_size.")
            else:
                raise ValueError(f"Invalid lb and ub. They should have the same length.")
        elif type(lb) in (int, float) and type(ub) in (int, float):
            lb = (float(lb), ) * problem_size
            ub = (float(ub), ) * problem_size
        else:
            raise ValueError(f"Invalid lb and ub. They should be a number of list/tuple/np.ndarray with size equal to problem_size")
        log_to = "console" if self.verbose else "None"
        if self.obj_name is None:
            raise ValueError("obj_name can't be None")
        else:
            if self.obj_name in self.SUPPORTED_REG_OBJECTIVES.keys():
                minmax = self.SUPPORTED_REG_OBJECTIVES[self.obj_name]
            elif self.obj_name in self.SUPPORTED_CLS_OBJECTIVES.keys():
                minmax = self.SUPPORTED_CLS_OBJECTIVES[self.obj_name]
            else:
                raise ValueError("obj_name is not supported. Please check the library: permetrics to see the supported objective function.")
        problem = {
            "obj_func": self.objective_function,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": minmax,
            "log_to": log_to,
            "save_population": save_population,
            "obj_weights": self.obj_weights
        }
        self.optimizer.solve(problem)
        self.network.update_weights_from_solution(self.optimizer.g_best.solution)
        self.loss_train = self._get_history_loss(optimizer=self.optimizer)
        return self
