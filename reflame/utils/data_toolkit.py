#!/usr/bin/env python
# Created by "Thieu" at 18:17, 24/10/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox, yeojohnson
from scipy.special import inv_boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


class TimeSeriesDifferencer:

    def __init__(self, interval=1):
        if interval < 1:
            raise ValueError("Interval for differencing must be at least 1.")
        self.interval = interval

    def difference(self, X):
        self.original_data = X.copy()
        return np.array([X[i] - X[i - self.interval] for i in range(self.interval, len(X))])

    def inverse_difference(self, diff_data):
        if self.original_data is None:
            raise ValueError("Original data is required for inversion.")
        return np.array([diff_data[i - self.interval] + self.original_data[i - self.interval] for i in range(self.interval, len(self.original_data))])


class FeatureEngineering:
    def __init__(self):
        """
        Initialize the FeatureEngineering class
        """
        # Check if the threshold is a valid number
        pass

    def create_threshold_binary_features(self, X, threshold):
        """
        Perform feature engineering to add binary indicator columns for values below the threshold.
        Add each new column right after the corresponding original column.

        Args:
        X (numpy.ndarray): The input 2D matrix of shape (n_samples, n_features).
        threshold (float): The threshold value for identifying low values.

        Returns:
        numpy.ndarray: The updated 2D matrix with binary indicator columns.
        """
        # Check if X is a NumPy array
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X should be a NumPy array.")
        # Check if the threshold is a valid number
        if not (isinstance(threshold, int) or isinstance(threshold, float)):
            raise ValueError("Threshold should be a numeric value.")

        # Create a new matrix to hold the original and new columns
        X_new = np.zeros((X.shape[0], X.shape[1] * 2))
        # Iterate over each column in X
        for idx in range(X.shape[1]):
            feature_values = X[:, idx]
            # Create a binary indicator column for values below the threshold
            indicator_column = (feature_values < threshold).astype(int)
            # Add the original column and indicator column to the new matrix
            X_new[:, idx * 2] = feature_values
            X_new[:, idx * 2 + 1] = indicator_column
        return X_new


class LabelEncoder:
    """
    Encode categorical features as integer labels.
    """

    def __init__(self):
        self.unique_labels = None
        self.label_to_index = {}

    @staticmethod
    def check_y(y):
        y = np.squeeze(np.asarray(y))
        if y.ndim != 1:
            raise ValueError("y label should have shape like 1-D vector.")
        return y

    def fit(self, y):
        """
        Fit label encoder to a given set of labels.

        Parameters:
        -----------
        y : array-like
            Labels to encode.
        """
        y = self.check_y(y)
        self.unique_labels = np.unique(y)
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}

    def transform(self, y):
        """
        Transform labels to encoded integer labels.

        Parameters:
        -----------
        y : array-like (1-D vector)
            Labels to encode.

        Returns:
        --------
        encoded_labels : array-like
            Encoded integer labels.
        """
        y = self.check_y(y)
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        return np.array([self.label_to_index[label] for label in y])

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        """
        Transform integer labels to original labels.

        Parameters:
        -----------
        y : array-like
            Encoded integer labels.

        Returns:
        --------
        original_labels : array-like
            Original labels.
        """
        y = self.check_y(y)
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        return np.array([self.unique_labels[i] if i in self.label_to_index.values() else "unknown" for i in y])


class ObjectiveScaler:
    """
    For label scaler in classification (binary and multiple classification)
    """
    def __init__(self, obj_name="sigmoid", ohe_scaler=None):
        """
        ohe_scaler: Need to be an instance of One-Hot-Encoder for softmax scaler (multiple classification problem)
        """
        self.obj_name = obj_name
        self.ohe_scaler = ohe_scaler

    def transform(self, data):
        if self.obj_name == "sigmoid" or self.obj_name == "self":
            return data
        elif self.obj_name == "hinge":
            data = np.squeeze(np.array(data))
            data[np.where(data == 0)] = -1
            return data
        elif self.obj_name == "softmax":
            data = self.ohe_scaler.transform(np.reshape(data, (-1, 1)))
            return data

    def inverse_transform(self, data):
        if self.obj_name == "sigmoid":
            data = np.squeeze(np.array(data))
            data = np.rint(data).astype(int)
        elif self.obj_name == "hinge":
            data = np.squeeze(np.array(data))
            data = np.ceil(data).astype(int)
            data[np.where(data == -1)] = 0
        elif self.obj_name == "softmax":
            data = np.squeeze(np.array(data))
            data = np.argmax(data, axis=1)
        return data


class Log1pScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # LogETransformer doesn't require fitting, so we simply return self.
        return self

    def transform(self, X):
        # Apply the natural logarithm to each element of the input data
        return np.log1p(X)

    def inverse_transform(self, X):
        # Apply the exponential function to reverse the logarithmic transformation
        return np.expm1(X)


class LogeScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # LogETransformer doesn't require fitting, so we simply return self.
        return self

    def transform(self, X):
        # Apply the natural logarithm (base e) to each element of the input data
        return np.log(X)

    def inverse_transform(self, X):
        # Apply the exponential function to reverse the logarithmic transformation
        return np.exp(X)


class SqrtScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # SqrtScaler doesn't require fitting, so we simply return self.
        return self

    def transform(self, X):
        # Apply the square root transformation to each element of the input data
        return np.sqrt(X)

    def inverse_transform(self, X):
        # Apply the square of each element to reverse the square root transformation
        return X ** 2


class BoxCoxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, lmbda=None):
        self.lmbda = lmbda

    def fit(self, X, y=None):
        # Estimate the lambda parameter from the data if not provided
        if self.lmbda is None:
            _, self.lmbda = boxcox(X.flatten())
        return self

    def transform(self, X):
        # Apply the Box-Cox transformation to the data
        X_new = boxcox(X.flatten(), lmbda=self.lmbda)
        return X_new.reshape(X.shape)

    def inverse_transform(self, X):
        # Inverse transform using the original lambda parameter
        return inv_boxcox(X, self.lmbda)


class YeoJohnsonScaler(BaseEstimator, TransformerMixin):

    def __init__(self, lmbda=None):
        self.lmbda = lmbda

    def fit(self, X, y=None):
        # Estimate the lambda parameter from the data if not provided
        if self.lmbda is None:
            _, self.lmbda = yeojohnson(X.flatten())
        return self

    def transform(self, X):
        # Apply the Yeo-Johnson transformation to the data
        X_new = boxcox(X.flatten(), lmbda=self.lmbda)
        return X_new.reshape(X.shape)

    def inverse_transform(self, X):
        # Inverse transform using the original lambda parameter
        return inv_boxcox(X, self.lmbda)


class SinhArcSinhScaler(BaseEstimator, TransformerMixin):
    # https://stats.stackexchange.com/questions/43482/transformation-to-increase-kurtosis-and-skewness-of-normal-r-v
    def __init__(self, epsilon=0.1, delta=1.0):
        self.epsilon = epsilon
        self.delta = delta

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.sinh(self.delta * np.arcsinh(X) - self.epsilon)

    def inverse_transform(self, X):
        return np.sinh((np.arcsinh(X) + self.epsilon) / self.delta)


class DataTransformer(BaseEstimator, TransformerMixin):

    SUPPORTED_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "max-abs": MaxAbsScaler,
                         "log1p": Log1pScaler, "loge": LogeScaler, "sqrt": SqrtScaler,
                         "sinh-arc-sinh": SinhArcSinhScaler, "robust": RobustScaler,
                         "box-cox": BoxCoxScaler, "yeo-johnson": YeoJohnsonScaler}

    def __init__(self, scaling_methods=('standard', ), list_dict_paras=None):
        if type(scaling_methods) is str:
            if list_dict_paras is None:
                self.list_dict_paras = [{}]
            elif type(list_dict_paras) is dict:
                self.list_dict_paras = [list_dict_paras]
            else:
                raise TypeError(f"You use only 1 scaling method, the list_dict_paras should be dict of parameter for that scaler.")
            self.scaling_methods = [scaling_methods]
        elif type(scaling_methods) in (tuple, list, np.ndarray):
            if list_dict_paras is None:
                self.list_dict_paras = [{}, ]*len(scaling_methods)
            elif type(list_dict_paras) in (tuple, list, np.ndarray):
                self.list_dict_paras = list(list_dict_paras)
            else:
                raise TypeError(f"Invalid type of list_dict_paras. Supported type are: tuple, list, or np.ndarray of parameter dict")
            self.scaling_methods = list(scaling_methods)
        else:
            raise TypeError(f"Invalid type of scaling_methods. Supported type are: str, tuple, list, or np.ndarray")

        self.scalers = [self._get_scaler(technique, paras) for (technique, paras) in zip(self.scaling_methods, self.list_dict_paras)]

    def _get_scaler(self, technique, paras):
        if technique in self.SUPPORTED_SCALERS.keys():
            if type(paras) is not dict:
                paras = {}
            return self.SUPPORTED_SCALERS[technique](**paras)
        else:
            raise ValueError(f"Invalid scaling technique. Supported techniques are {self.SUPPORTED_SCALERS.keys()}")

    def fit(self, X, y=None):
        for idx, _ in enumerate(self.scalers):
            X = self.scalers[idx].fit_transform(X)
        return self

    def transform(self, X):
        for scaler in self.scalers:
            X = scaler.transform(X)
        return X

    def inverse_transform(self, X):
        for scaler in reversed(self.scalers):
            X = scaler.inverse_transform(X)
        return X


class Data:
    """
    The structure of our supported Data class

    Parameters
    ----------
    X : np.ndarray
        The features of your data

    y : np.ndarray
        The labels of your data
    """

    SUPPORT = {
        "scaler": list(DataTransformer.SUPPORTED_SCALERS.keys())
    }

    def __init__(self, X=None, y=None, name="Unknown"):
        self.X = X
        self.y = self.check_y(y)
        self.name = name
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

    @staticmethod
    def check_y(y):
        if y is None:
            return y
        y = np.squeeze(np.asarray(y))
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        return y

    @staticmethod
    def scale(X, scaling_methods=('standard', ), list_dict_paras=None):
        X = np.squeeze(np.asarray(X))
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if X.ndim >= 3:
            raise TypeError(f"Invalid X data type. It should be array-like with shape (n samples, m features)")
        scaler = DataTransformer(scaling_methods=scaling_methods, list_dict_paras=list_dict_paras)
        data = scaler.fit_transform(X)
        return data, scaler

    @staticmethod
    def encode_label(y):
        y = np.squeeze(np.asarray(y))
        if y.ndim != 1:
            raise TypeError(f"Invalid y data type. It should be a vector / array-like with shape (n samples,)")
        scaler = LabelEncoder()
        data = scaler.fit_transform(y)
        return data, scaler

    def split_train_test(self, test_size=0.2, train_size=None,
                         random_state=41, shuffle=True, stratify=None, inplace=True):
        """
        The wrapper of the split_train_test function in scikit-learn library.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                        train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        if not inplace:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def set_train_test(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Function use to set your own X_train, y_train, X_test, y_test in case you don't want to use our split function

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray
        X_test : np.ndarray
        y_test : np.ndarray
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        return self
