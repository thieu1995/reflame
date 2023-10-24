#!/usr/bin/env python
# Created by "Thieu" at 10:12, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from reflame import Data, FlnnRegressor


def get_dataset():
    from sklearn import datasets

    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    dt_obj = Data(X, y)
    return dt_obj


## Load data object
data = get_dataset()

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", "minmax"))
data.y_test = scaler_y.transform(data.y_test)

## Create model
model = FlnnRegressor(expand_name="chebyshev", n_funcs=4, act_name="elu",
                      obj_name="MSE", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)

## Train the model
model.fit(data.X_train, data.y_train)

## Test the model
y_pred = model.predict(data.X_test)
print(y_pred)

## Calculate some metrics
print(model.score(X=data.X_test, y=data.y_test, method="RMSE"))
print(model.scores(X=data.X_test, y=data.y_test, list_methods=["R2", "NSE", "MAPE"]))
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["R2", "NSE", "MAPE", "NNSE"]))