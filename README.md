
<p align="center">
<img style="max-width:100%;" src="https://thieu1995.github.io/post/2023-08/reflame.png" alt="Reflame"/>
</p>


---

[![GitHub release](https://img.shields.io/badge/release-1.0.1-yellow.svg)](https://github.com/thieu1995/reflame/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/reflame) 
[![PyPI version](https://badge.fury.io/py/reflame.svg)](https://badge.fury.io/py/reflame)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/reflame.svg)
![PyPI - Status](https://img.shields.io/pypi/status/reflame.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/reflame.svg)
[![Downloads](https://pepy.tech/badge/reflame)](https://pepy.tech/project/reflame)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/reflame/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/reflame/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/reflame.svg)
[![Documentation Status](https://readthedocs.org/projects/reflame/badge/?version=latest)](https://reflame.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/reflame.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10067995.svg)](https://doi.org/10.5281/zenodo.10067995)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


Reflame (REvolutionizing Functional Link Artificial neural networks by MEtaheuristic algorithms) is a Python library that 
implements a framework for training Functional Link Neural Network (FLNN) networks using Metaheuristic Algorithms. It 
provides a comparable alternative to the traditional FLNN network and is compatible with the Scikit-Learn library. 
With Reflame, you can perform searches and hyperparameter tuning using the functionalities provided by the Scikit-Learn library.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimator**: FlnnRegressor, FlnnClassifier, MhaFlnnRegressor, MhaFlnnClassifier
* **Total Official Metaheuristic-based Flnn Regression**: > 200 Models 
* **Total Official Metaheuristic-based Flnn Classification**: > 200 Models
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Supported objective functions (as fitness functions or loss functions)**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://reflame.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics, torch, skorch


# Citation Request 

If you want to understand how Metaheuristic is applied to Functional Link Neural Network, you need to read the paper 
titled "A resource usage prediction system using functional-link and genetic algorithm neural network for multivariate cloud metrics". 
The paper can be accessed at the following [this link](https://doi.org/10.1109/SOCA.2018.00014)


Please include these citations if you plan to use this library:

```code
@software{nguyen_van_thieu_2023_8249046,
  author       = {Nguyen Van Thieu},
  title        = {Revolutionizing Functional Link Neural Network by Metaheuristic Algorithms: reflame - A Python Library},
  month        = 11,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.8249045},
  url          = {https://github.com/thieu1995/reflame}
}

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}

@inproceedings{nguyen2019building,
	author = {Thieu Nguyen and Binh Minh Nguyen and Giang Nguyen},
	booktitle = {International Conference on Theory and Applications of Models of Computation},
	organization = {Springer},
	pages = {501--517},
	title = {Building Resource Auto-scaler with Functional-Link Neural Network and Adaptive Bacterial Foraging Optimization},
	year = {2019},
	url={https://doi.org/10.1007/978-3-030-14812-6_31},
	doi={10.1007/978-3-030-14812-6_31}
}

@inproceedings{nguyen2018resource,
	author = {Thieu Nguyen and Nhuan Tran and Binh Minh Nguyen and Giang Nguyen},
	booktitle = {2018 IEEE 11th Conference on Service-Oriented Computing and Applications (SOCA)},
	organization = {IEEE},
	pages = {49--56},
	title = {A Resource Usage Prediction System Using Functional-Link and Genetic Algorithm Neural Network for Multivariate Cloud Metrics},
	year = {2018},
	url={https://doi.org/10.1109/SOCA.2018.00014},
	doi={10.1109/SOCA.2018.00014}
}

```

# Installation

* Install the [current PyPI release](https://pypi.python.org/pypi/reflame):
```sh 
$ pip install reflame==1.0.1
```

* Install directly from source code
```sh 
$ git clone https://github.com/thieu1995/reflame.git
$ cd reflame
$ python setup.py install
```

* In case, you want to install the development version from Github:
```sh 
$ pip install git+https://github.com/thieu1995/reflame 
```

After installation, you can import Reflame as any other Python module:

```sh
$ python
>>> import reflame
>>> reflame.__version__
```

### Examples

In this section, we will explore the usage of the Reflame model with the assistance of a dataset. While all the 
preprocessing steps mentioned below can be replicated using Scikit-Learn, we have implemented some utility functions 
to provide users with convenience and faster usage.  

#### Combine Reflame library like a normal library with scikit-learn.

```python
### Step 1: Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from reflame import FlnnRegressor, FlnnClassifier, MhaFlnnRegressor, MhaFlnnClassifier

#### Step 2: Reading the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#### Step 3: Next, split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=100)

#### Step 4: Feature Scaling
scaler_X = MinMaxScaler()
scaler_X.fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

le_y = LabelEncoder()  # This is for classification problem only
le_y.fit(y)
y_train = le_y.transform(y_train)
y_test = le_y.transform(y_test)

#### Step 5: Fitting FLNN-based model to the dataset

##### 5.1: Use standard FLNN model for regression problem
regressor = FlnnRegressor(expand_name="chebyshev", n_funcs=4, act_name="elu",
                      obj_name="MSE", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
regressor.fit(X_train, y_train)

##### 5.2: Use standard FLNN model for classification problem 
classifer = FlnnClassifier(expand_name="chebyshev", n_funcs=4, act_name="sigmoid",
                      obj_name="BCEL", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
classifer.fit(X_train, y_train)

##### 5.3: Use Metaheuristic-based FLNN model for regression problem
print(MhaFlnnClassifier.SUPPORTED_OPTIMIZERS)
print(MhaFlnnClassifier.SUPPORTED_REG_OBJECTIVES)
opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
model = MhaFlnnRegressor(expand_name="chebyshev", n_funcs=3, act_name="elu", 
                         obj_name="RMSE", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True)
regressor.fit(X_train, y_train)

##### 5.4: Use Metaheuristic-based FLNN model for classification problem
print(MhaFlnnClassifier.SUPPORTED_OPTIMIZERS)
print(MhaFlnnClassifier.SUPPORTED_CLS_OBJECTIVES)
opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
classifier = MhaFlnnClassifier(expand_name="chebyshev", n_funcs=4, act_name="sigmoid",
                          obj_name="NPV", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True)
classifier.fit(X_train, y_train)

#### Step 6: Predicting a new result
y_pred = regressor.predict(X_test)

y_pred_cls = classifier.predict(X_test)
y_pred_label = le_y.inverse_transform(y_pred_cls)

#### Step 7: Calculate metrics using score or scores functions.
print("Try my AS metric with score function")
print(regressor.score(X_test, y_test, method="AS"))

print("Try my multiple metrics with scores function")
print(classifier.scores(X_test, y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))
```

#### Utilities everything that Reflame provided

```python
### Step 1: Importing the libraries
from reflame import Data, FlnnRegressor, FlnnClassifier, MhaFlnnRegressor, MhaFlnnClassifier
from sklearn.datasets import load_digits

#### Step 2: Reading the dataset
X, y = load_digits(return_X_y=True)
data = Data(X, y)

#### Step 3: Next, split dataset into train and test set
data.split_train_test(test_size=0.2, shuffle=True, random_state=100)

#### Step 4: Feature Scaling
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
data.y_test = scaler_y.transform(data.y_test)

#### Step 5: Fitting FLNN-based model to the dataset

##### 5.1: Use standard FLNN model for regression problem
regressor = FlnnRegressor(expand_name="chebyshev", n_funcs=4, act_name="tanh",
                      obj_name="MSE", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
regressor.fit(data.X_train, data.y_train)

##### 5.2: Use standard FLNN model for classification problem 
classifer = FlnnClassifier(expand_name="chebyshev", n_funcs=4, act_name="tanh",
                      obj_name="BCEL", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
classifer.fit(data.X_train, data.y_train)

##### 5.3: Use Metaheuristic-based FLNN model for regression problem
print(MhaFlnnClassifier.SUPPORTED_OPTIMIZERS)
print(MhaFlnnClassifier.SUPPORTED_REG_OBJECTIVES)
opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
model = MhaFlnnRegressor(expand_name="chebyshev", n_funcs=3, act_name="elu", 
                         obj_name="RMSE", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True)
regressor.fit(data.X_train, data.y_train)

##### 5.4: Use Metaheuristic-based FLNN model for classification problem
print(MhaFlnnClassifier.SUPPORTED_OPTIMIZERS)
print(MhaFlnnClassifier.SUPPORTED_CLS_OBJECTIVES)
opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
classifier = MhaFlnnClassifier(expand_name="chebyshev", n_funcs=4, act_name="sigmoid",
                          obj_name="NPV", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True)
classifier.fit(data.X_train, data.y_train)

#### Step 6: Predicting a new result
y_pred = regressor.predict(data.X_test)

y_pred_cls = classifier.predict(data.X_test)
y_pred_label = scaler_y.inverse_transform(y_pred_cls)

#### Step 7: Calculate metrics using score or scores functions.
print("Try my AS metric with score function")
print(regressor.score(data.X_test, data.y_test, method="AS"))

print("Try my multiple metrics with scores function")
print(classifier.scores(data.X_test, data.y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))
```

A real-world dataset contains features that vary in magnitudes, units, and range. We would suggest performing 
normalization when the scale of a feature is irrelevant or misleading. Feature Scaling basically helps to normalize 
the data within a particular range.



1) Where do I find the supported metrics like above ["AS", "PS", "RS"]. What is that?
You can find it here: https://github.com/thieu1995/permetrics or use this

```python
from reflame import MhaFlnnClassifier, MhaFlnnRegressor

print(MhaFlnnRegressor.SUPPORTED_REG_OBJECTIVES)
print(MhaFlnnClassifier.SUPPORTED_CLS_OBJECTIVES)
```

2) I got this type of error
```python
raise ValueError("Existed at least one new label in y_pred.")
ValueError: Existed at least one new label in y_pred.
``` 
How to solve this?

+ This occurs only when you are working on a classification problem with a small dataset that has many classes. For 
  instance, the "Zoo" dataset contains only 101 samples, but it has 7 classes. If you split the dataset into a 
  training and testing set with a ratio of around 80% - 20%, there is a chance that one or more classes may appear 
  in the testing set but not in the training set. As a result, when you calculate the performance metrics, you may 
  encounter this error. You cannot predict or assign new data to a new label because you have no knowledge about the 
  new label. There are several solutions to this problem.

+ 1st: Use the SMOTE method to address imbalanced data and ensure that all classes have the same number of samples.

```python
import pandas as pd
from imblearn.over_sampling import SMOTE
from reflame import Data

dataset = pd.read_csv('examples/dataset.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]

X_new, y_new = SMOTE().fit_resample(X, y)
data = Data(X_new, y_new)
```

+ 2nd: Use different random_state numbers in split_train_test() function.

```python
import pandas as pd
from reflame import Data

dataset = pd.read_csv('examples/dataset.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]
data = Data(X, y)
data.split_train_test(test_size=0.2, random_state=10)  # Try different random_state value 
```


# Support (questions, problems)

### Official Links 

* Official source code repo: https://github.com/thieu1995/reflame
* Official document: https://reflame.readthedocs.io/
* Download releases: https://pypi.org/project/reflame/
* Issue tracker: https://github.com/thieu1995/reflame/issues
* Notable changes log: https://github.com/thieu1995/reflame/blob/master/ChangeLog.md
* Official chat group: https://t.me/+fRVCJGuGJg1mNDg1

* This project also related to our another projects which are "optimization" and "machine learning", check it here:
    * https://github.com/thieu1995/mealpy
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/thieu1995/opfunu
    * https://github.com/thieu1995/enoppy
    * https://github.com/thieu1995/permetrics
    * https://github.com/thieu1995/MetaCluster
    * https://github.com/thieu1995/pfevaluator
    * https://github.com/thieu1995/intelelm
    * https://github.com/aiir-team
