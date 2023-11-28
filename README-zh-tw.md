🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md) 

# 機器學習筆記

## 目錄

- [數據預處理](https://github.com/chung-kai-eng/Eric/blob/master/Preprocessing.md)
- [正規化回歸、MARS、PLS和核心概述](https://github.com/chung-kai-eng/Eric/blob/master/Regularized%20Regression_MARS_PLS_SVM.md)
- [集成學習](https://github.com/chung-kai-eng/Eric/blob/master/Ensemble%20Learning.md)
- [評估指標](https://github.com/chung-kai-eng/Eric/blob/master/Evaluation%20Metrics.md)

## 相關套件

### 特徵工程
- [Category Encoders](https://github.com/scikit-learn-contrib/category_encoders)：用於編碼分類變數
- [Featuretools](https://www.featuretools.com/)：可與 [Compose](https://github.com/alteryx/compose) 和 [EvalML](https://github.com/alteryx/evalml) 一起使用
    - `Featuretools` 自動化特徵工程過程
    - `EvalML` 自動化模型構建，包括數據檢查，甚至提供模型理解工具（[教程](https://evalml.alteryx.com/en/stable/demos/fraud.html)）
    - `Compose` 自動化預測工程
- [feature-engine](https://github.com/solegalli/feature_engine)
    - 變數轉換、選擇、預處理
    - 填充、編碼、離散化、異常值處理
    - 時間序列特徵
- [imbalance](https://github.com/scikit-learn-contrib/imbalanced-learn)：處理數據不平衡問題
    - [`sklearn resample`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)
    - [`tensorflow tf.data_sampler`](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#using_tfdata)
    - [`pytorch torch.utils.data.WeightedRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)

### 特徵選擇
- [feature-engine](https://github.com/solegalli/feature_engine)
- [mlxtend](https://github.com/rasbt/mlxtend)：包裝器、貪婪算法等（也包括一些數據可視化工具）

### 建模
- [sklearn]()
- [xgboost]()
- [lightgbm]()
- [pyearch](https://github.com/scikit-learn-contrib/py-earth)：多變量自適應回歸樣條
- [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn)：重點關注標籤空間操作的多標籤分類
- [seglearn](https://github.com/dmbee/seglearn)：使用滑動窗口分割的時間序列和序列學習
- [pomegranate](https://github.com/jmschrei/pomegranate)：用於Python的概率建模，重點是隱馬爾可夫模型（GMM、HMM、Naive Bayes和Bayes分類器、馬爾可夫鏈、離散貝葉斯網絡、離散馬爾可夫網絡）

### 超參數調整
- [sklearn-deap](https://github.com/rsteca/sklearn-deap)：在scikit-learn中使用進化算法而不是網格搜索
- [hyperopt](https://github.com/hyperopt/hyperopt)：貝葉斯超參數調整
- [scikit-optimize](https://scikit-optimize.github.io/stable/)：scikit-learn超參數調整
- [Optuna](https://github.com/optuna/optuna)：[Optuna指南](https://github.com/chung-kai-eng/Eric/blob/master/Optuna_guidance.md)

### 時間序列
- [tslearn](https://github.com/tslearn-team/tslearn)：時間序列預處理、特徵提取、分類、回歸、聚類
- [sktime](https://github.com/alan-turing-institute/sktime)：時間序列分類、回歸、聚類、注釋（也可用於單變量、多變量或面板數據）
- [HMMLearn](https://github.com/hmmlearn/hmmlearn)：隱馬爾可夫模型的實現
- [pytorchforecasting](https://pytorch-forecasting.readthedocs.io/en/stable/index.html)：使用pytorch實現的時間序列預測模型

### 自動機器學習
- [auto-sklearn](https://github.com/automl/auto-sklearn/)
- [Compose、Featuretools、EvalML]()
- [tpot](https://github.com/EpistasisLab/tpot)

### 異常檢測
- [PyOD 用於表格數據集](https://github.com/yzhao062/pyod)
- [TODS 用於時間序列](https://github.com/datamllab/tods)
- [不同類型數據集的異常檢測](https://github.com/yzhao062)

### MLOps
- [mlflow]()
- [clearml]()
- [wandb]()
- [dvc]：命令行用法（不僅包括數據版本控制）
- [comet]()
- [Neptune.ai]()

[Lazyprediction 用於模型列表](https://github.com/shankarpand

ala/lazypredict)

[與Scikit Learn相關的項目](https://scikit-learn.org/stable/related_projects.html)
