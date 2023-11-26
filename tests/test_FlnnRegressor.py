#!/usr/bin/env python
# Created by "Thieu" at 11:25, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from reflame import FlnnRegressor

np.random.seed(42)


def test_FlnnRegressor_class():
    X = np.random.uniform(low=0.0, high=1.0, size=(100, 5))
    noise = np.random.normal(loc=0.0, scale=0.1, size=(100, 5))
    y = 2 * X + 1 + noise

    model = FlnnRegressor(expand_name="chebyshev", n_funcs=4, act_name="none",
                          obj_name="MSE", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
    model.fit(X, y)

    pred = model.predict(X)
    assert FlnnRegressor.SUPPORTED_REG_METRICS == model.SUPPORTED_REG_METRICS
    assert len(pred) == X.shape[0]
