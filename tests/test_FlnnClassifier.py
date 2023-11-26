#!/usr/bin/env python
# Created by "Thieu" at 11:27, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from reflame import FlnnClassifier

np.random.seed(41)


def test_FlnnClassifier_class():
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 2, size=100)

    model = FlnnClassifier(expand_name="chebyshev", n_funcs=4, act_name="softmax",
                           obj_name="BCEL", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
    model.fit(X, y)
    pred = model.predict(X)
    assert FlnnClassifier.SUPPORTED_CLS_METRICS == model.SUPPORTED_CLS_METRICS
    assert pred[0] in (0, 1)
