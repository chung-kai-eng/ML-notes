Machine Learning Note
===

- [Data Preprocessing](https://github.com/chung-kai-eng/Eric/blob/master/Preprocessing.md)
- [Regularized Regression & MARS & PLS & kernel overview](https://github.com/chung-kai-eng/Eric/blob/master/Regularized%20Regression_MARS_PLS_SVM.md)
- [Ensemble Learning](https://github.com/chung-kai-eng/Eric/blob/master/Ensemble%20Learning.md)
- [Evaluation Metrics](https://github.com/chung-kai-eng/Eric/blob/master/Evaluation%20Metrics.md)


## Useful package

### Feature engineering
- [Category Encoders](https://github.com/scikit-learn-contrib/category_encoders):encode categorical variable
- [Featuretools](https://www.featuretools.com/): be utilized with [compose](https://github.com/alteryx/compose) & [EvalML](https://github.com/alteryx/evalml)
    - ```Featuretools``` automates the feature engineering process
    - ```EvalML``` automates model building, includes data checks, and even offers tools for model understanding [(Tutorial)](https://evalml.alteryx.com/en/stable/demos/fraud.html)
    - ```Compose``` automates prediction engineering
- [feature-engine](https://github.com/solegalli/feature_engine)
- [imbalance](https://github.com/scikit-learn-contrib/imbalanced-learn): deal with imbalance data issue

### Feature selection
- [feature-engine](https://github.com/solegalli/feature_engine)
- [mlxtend](https://github.com/rasbt/mlxtend): wrapper, greedy algorithm, etc (also include some data visualization tool)

### Modeling
- [sklearn]()
- [xgboost]()
- [lightgbm]()
- [pyearch](https://github.com/scikit-learn-contrib/py-earth): Multivariate Adaptive regression spline
- [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn):  Multi-label classification with focus on label space manipulation
- [seglearn](https://github.com/dmbee/seglearn): Time series and sequence learning using sliding window segmentation
- [pomegranate](https://github.com/jmschrei/pomegranate): Probabilistic modelling for Python, with an emphasis on hidden Markov models. (GMM, HMM, Naive Bayes and Bayes Classifiers, Markov Chains, Discrete Bayesian Networks, Discrete Markov Networks)

### Hyperparameter
- [sklearn-deap](https://github.com/rsteca/sklearn-deap): Use evolutionary algorithms instead of gridsearch in scikit-learn.
- [hyperopt](https://github.com/hyperopt/hyperopt): bayesian hyperparameter
- [scikit-optimize](https://scikit-optimize.github.io/stable/): scikit learn hyperparamter
- [Optuna](https://github.com/optuna/optuna)


### Time series
- [tslearn](https://github.com/tslearn-team/tslearn): time series preprocessing, feature extraction, classification, regression, clustering
- [sktime](https://github.com/alan-turing-institute/sktime): time series classification, regression
- [HMMLearn](https://github.com/hmmlearn/hmmlearn): Implementation of hidden markov models

### AutoML
- [auto-sklearn](https://github.com/automl/auto-sklearn/)
- [Compose, Featuretools, EvalML]()
- [tpot](https://github.com/EpistasisLab/tpot)

[Scikit Learn related projects](https://scikit-learn.org/stable/related_projects.html)
