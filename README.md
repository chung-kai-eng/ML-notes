🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md) 

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
    - Variable transformation, selection, preprocessing
    - Imputatoin, Encoding, Discretization, Outlier Handling
    - Time series features
- [imbalance](https://github.com/scikit-learn-contrib/imbalanced-learn): deal with imbalance data issue
    - [```sklearn resample```](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)
    - [```tensorflow tf.data_sampler```](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#using_tfdata)
    - [```pytorch torch.utils.data.WeightedRandomSampler```](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)

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
- [Optuna](https://github.com/optuna/optuna): [Guidance for Optuna](https://github.com/chung-kai-eng/Eric/blob/master/Optuna_guidance.md)


### Time series
- [tslearn](https://github.com/tslearn-team/tslearn): time series preprocessing, feature extraction, classification, regression, clustering
- [sktime](https://github.com/alan-turing-institute/sktime): time series classification, regression, clustering, annotation (also can be used in data that is univariate, multivariate or panel)
- [HMMLearn](https://github.com/hmmlearn/hmmlearn): Implementation of hidden markov models
- [pytorchforecasting](https://pytorch-forecasting.readthedocs.io/en/stable/index.html): time series forecasting model implemented by pytorch
- [Nixtla](https://nixtlaverse.nixtla.io/#mlforecast): open-source libraries for time series forecasting (include 5 main libraries, `statsforecast`, `MLForecast`, `NeuralForecast`, `Hierarchical Forecast`, `TS features`)

### AutoML
- [auto-sklearn](https://github.com/automl/auto-sklearn/)
- [Compose, Featuretools, EvalML]()
- [tpot](https://github.com/EpistasisLab/tpot)


### Anomaly detection
- [PyOD for tabular dataset](https://github.com/yzhao062/pyod)
- [TODS for time seires](https://github.com/datamllab/tods)
- [Anomaly detection for different types of dataset](https://github.com/yzhao062)

### MLOps
- [mlflow]()
- [clearml]()
- [wandb]()
- [dvc](): command line usage (not only included data version control)
- [comet]()
- [Neptune.ai]()

[Lazyprediction for lists of models](https://github.com/shankarpandala/lazypredict)

[Scikit Learn related projects](https://scikit-learn.org/stable/related_projects.html)
