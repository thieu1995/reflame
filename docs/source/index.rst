.. Reflame documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Reflame's documentation!
====================================

.. image:: https://img.shields.io/badge/release-1.0.0-yellow.svg
   :target: https://github.com/thieu1995/reflame/releases

.. image:: https://img.shields.io/pypi/wheel/gensim.svg
   :target: https://pypi.python.org/pypi/reflame

.. image:: https://badge.fury.io/py/reflame.svg
   :target: https://badge.fury.io/py/reflame

.. image:: https://img.shields.io/pypi/pyversions/reflame.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/status/reflame.svg
   :target: https://img.shields.io/pypi/status/reflame.svg

.. image:: https://img.shields.io/pypi/dm/reflame.svg
   :target: https://img.shields.io/pypi/dm/reflame.svg

.. image:: https://github.com/thieu1995/reflame/actions/workflows/publish-package.yaml/badge.svg
   :target: https://github.com/thieu1995/reflame/actions/workflows/publish-package.yaml

.. image:: https://pepy.tech/badge/reflame
   :target: https://pepy.tech/project/reflame

.. image:: https://img.shields.io/github/release-date/thieu1995/reflame.svg
   :target: https://img.shields.io/github/release-date/thieu1995/reflame.svg

.. image:: https://readthedocs.org/projects/reflame/badge/?version=latest
   :target: https://reflame.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/github/contributors/thieu1995/reflame.svg
   :target: https://img.shields.io/github/contributors/thieu1995/reflame.svg

.. image:: https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?
   :target: https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8249046.svg
   :target: https://doi.org/10.5281/zenodo.8249046

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


Reflame (REvolutionizing Functional Link Nets by Metaheuristic Algorithms) is a Python library that
implements a framework for training Functional Link Neural Network (FLNN) networks using Metaheuristic Algorithms. It
provides a comparable alternative to the traditional FLNN network and is compatible with the Scikit-Learn library.
With Reflame, you can perform searches and hyperparameter tuning using the functionalities provided by the Scikit-Learn library.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimator**: FlnnRegressor, FlnnClassifier, MhaFlnnRegressor, MhaFlnnClassifier
* **Total Official Metaheuristic-based Flnn Regression**: > 200 Models
* **Total Official Metaheuristic-based Flnn Classification**: > 200 Models
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Supported objective functions (as fitness functions or loss functions)**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://intelelm.readthedocs.io/en/latest/
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics, torch, skorch


.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/reflame.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
