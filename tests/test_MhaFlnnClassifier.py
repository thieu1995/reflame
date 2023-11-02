#!/usr/bin/env python
# Created by "Thieu" at 15:45, 15/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from reflame import MhaFlnnClassifier

np.random.seed(41)


def test_MhaFlnnClassifier_class():
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 2, size=100)

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    model = MhaFlnnClassifier(expand_name="chebyshev", n_funcs=4, act_name="sigmoid",
                              obj_name="NPV", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True)
    model.fit(X, y)
    pred = model.predict(X)
    assert MhaFlnnClassifier.SUPPORTED_CLS_OBJECTIVES == model.SUPPORTED_CLS_OBJECTIVES
    assert pred[0] in (0, 1)
