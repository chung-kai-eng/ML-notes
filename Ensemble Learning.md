###### tags: `python`
Ensemble Learning
===

## Why ensemble?
- average each model performance can reduce the variance
- just like $Z_1, Z_2,...,Z_n$, which $Var(Z_i)=\sigma^2$
- $Var(\bar Z) = \frac{\sigma^2}{n}$

![](https://i.imgur.com/FkLW6Z5.png)

## Embedding
- remember to use FunctionTransformer to make the layer dense
```python=
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomTreesEmbedding

model = make_pipeline(RandomTreesEmbedding(n_estimators = 20, max_depth = 4),
                      FunctionTransformer(lambda x: x.todense(), accept_sparse = True), 
                      SVR())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("rf_embedding + MARS")
rlt = regression_report(y_test, y_pred)

```
## Bagging Regressor, RandomForestRegressor, AdaBoostRegressor

### the main difference between bagging & random forest
- if the number of predictors used in the tree is same as the total input predictors, then random forest is same as bagging
- random forest: random restrict the features used in each split
![](https://i.imgur.com/qfeOFIO.png)
![](https://i.imgur.com/FryE32Q.png)

-  Most or all of the trees will use this strong predictors in the top split. Consequently, **all of the bagged trees** will look **quite similar** to each other. Hence the **predictions** from the **bagged trees will be highly correlated**. Unfortunately, averaging many **highly correlated quantities** doesn't lead to as large of a reduction in variance as averaging many uncorrelated quantities. In particular, this means that bagging will not lead to a substantial reduction in variance over a single tree in this setting. Thus, as for random forest, we usually set the **input features = sqrt(total features)**


[decision tree implementation](https://iter01.com/43652.html)
[Ensemble learning](https://www.quantstart.com/articles/bootstrap-aggregation-random-forests-and-boosted-trees)
[Random Forest](https://blog.datadive.net/selecting-good-features-part-iii-random-forests/?fbclid=IwAR08Aey2Ng3VPVKSLWkSFwG9VXqNWLLt61Q7iItuJ3i45ejEyeXAg86NqFA)
[Select good feature](https://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/)

[RandomForest extension](https://pub.towardsai.net/exploring-the-last-trends-of-random-forest-396bd0347aa1)

### 1. Mean decrease impurity
```python=
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassification

rf = RandomForestRegressor()
rf.fit(X,y)
print(rf.feature_importances_)
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True)
```
- When using the impurity based ranking. Feature selection based on **impurity reduction** is biased towards preferring variables with more categories.
- When the dataset has two (or more) correlated features, then from the point of view of the model, any of these correlated features can be use as the predictor. But once one of them is used, the importance of others is significantly reduced since effctively the impurity they can remove is already removed by the first feature.


### 2. Mean decrease accuracy
```python=
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
 
X = boston["data"]
Y = boston["target"]
 
rf = RandomForestRegressor()
scores = defaultdict(list)
 
#crossvalidate the scores on a number of different random splits of the data
for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[names[i]].append((acc-shuff_acc)/acc)
print "Features sorted by their score:"
print sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True)


```
## the way to show random forest's feature importance
[lightgbm parameter](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
[lightgbm simple parameter setting sample](https://kknews.cc/zh-tw/tech/y3a3x8j.html)
```python=
rfc = RandomForestClassifier(n_estimators = 200, criterion = "entropy", max_depth = 4, class_weight = "balanced_subsample")
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

confmat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.matshow(confmat, cmap = plt.cm.hsv, alpha = 0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x = j, y = i, s = confmat[i,j])
plt.xlabel('pred_result')
plt.ylabel('actu_result')
plt.show()

df_feature_importance = pd.DataFrame(rfc.feature_importances_, index = X_train.columns, 
                                     columns = ['feature importance']).sort_values('feature importance', ascending = False)
print(df_feature_importance)
df_feature_all = pd.DataFrame([tree.feature_importances_ for tree in rfc.estimators_], columns = X_train.columns)
df_feature_all.head()
# melted data (long format)
df_feature_long = pd.melt(df_feature_all, var_name = 'feature name', value_name = 'values')
ax = sns.boxplot(x = "feature name", y = "values", data = df_feature_long, order = df_feature_importance.index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90) 
```

- directly measure the impact of each feature on accuracy of the model. Clealy, for unimportant variables, the permutation should have little to no effect on model accuracy, while permuting important variables should significantly decrease it.
- not directly exposed in sklearn

## Bagging
- aggregation + bootstrap

### Out-of-bag(OOB)
- each bagged tree makes use of around $\frac{2}{3}$ of the observations. The remaining $\frac{1}{3}$ of the observations not used to fit a given bagged tree 
![](https://i.imgur.com/Z9Z8RiN.png)
- use bootstrap sample that is in the bag to train the model, and out-of-bag sample set can be used to test their respective model
- [wikipedia introduction](https://en.wikipedia.org/wiki/Out-of-bag_error)
- OOB and cross-validation are different methods of measuring the error. Over many iterations, two methods should produce a very similar error estimate. Once the OOB error stabilizes, it will converge to the cross-validation error. The advantage of the OOB method is less computation and allow one to test the model as it is being trained
- the conclusion of a study done by Silke Janitza and Roman Hornung, out-of-bag error has shown to overestimate in settings that include an equal number of observations from all response classes (balanced samples), small sample sizes, a large number of predictor variables, small correlation between predictors, and weak effects

### RandomTreesEmbedding
- an unsupervised transformation of the data 
- encode the data by the indices of the leaves a data point ends up in. This index is then encoded in a one-of-K manner, leading to a high dimensional, sparse binary coding, which is beneficial for classification
- can be computed efficiently and be used as a basis for other learning tasks. Tje soze amd sparsity of the code can be influenced by choosing the number of trees and the maximum depth per tree
### package
[open source link](https://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_embedding.html#sphx-glr-auto-examples-ensemble-plot-random-forest-embedding-py)
- sklearn.ensemble.RandomTreesEmbedding
- 
## Gradient Boosting Tree
- start by making a single leaf instead of a tree (initial guess is the average value)
- build tree based on the errors made by the previous tree
- In practice, people often set the maximum number of leaves to be between 8 and 32




## XGBoost
- pre-sorted algorithm & Histogram-based algorithm for computing the best split
    - for each node, enumerate over all features
    - for each feature, sort the instances by feature value
    - use a linear scan to decide the best split along that feature basis information gain
    - take the best split solution along all the features
[XGBoost interpretation](https://www.itread01.com/content/1548076506.html)
[DART](https://arxiv.org/pdf/1505.01866.pdf)
![](https://i.imgur.com/2XP319Z.png)
- Similarity Score
    $$ Similarity \ Score = \frac{RSS}{Number \ of \ Residual + \lambda}$$

![](https://i.imgur.com/ggqwUEi.png)

- the way to prune the XGBoost (based on the gain value) 
- gamma $\gamma$ (the value that represents the threshold of the gain value)
    - if $Gain-\gamma<0$, then remove the branch
    - if $Gain-\gamma>0$, then remain the branch
    - if all the Gain < $\gamma$, then prediction value will always be 0.5
![](https://i.imgur.com/9PlRLSJ.png)
- even when gamma=0, lambda not equal to 0, the prune effect still exists
![](https://i.imgur.com/CmsbcXJ.png)

- when doing hyperparameter, we can use one tree first to see how to set the hyperparameter range (```n_estimators=1```)

```python=
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y) # stratification for making the distribution for train & test set as close as possible 
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', missing=None, seed=42)
clf_xgb.fit(X_train, y_train, verbose=True,
            early_stopping_rounds=10, eval_metrics='aucpr',
            eval_set=[(X_test, y_test)]) # determine the number of tree via testing set(eval_set)
            
# param
# scale_pos_weight (imbalanced data): XGBoost recommends sum(negative instance) / sum(positive instance)
# subsample: random subset of the training set (speed up, diverse slightly different distribution )
# colsample_bytree: select subset of columns to make the tree more diverse
# gamma: 
# max_depth:
# reg_lambda

clf_xgb = xgb.XGBClassifier(objective='binary:logistic', missing=None, gamma=0.25, n_estimators=1, seed=42) # we can get gain, etc

xgb.to_graphviz(clf_xgb, num_tree=0, condition_node_params=node_params, leaf_node_params=leaf_params)
```


## LightGBM
- use **Gradient-Based One Side Sampling** (GOSS) to filter out the data instances for finding a split value while **XGBoost uses pre-sorted algorithm & Histogram-based algorithm** for computing the best split


- try use gridsearch to adjust the parameter
- [The way to tuning lightgbm](https://neptune.ai/blog/lightgbm-parameters-guide)

```python=
#%%
import lightgbm as lgb

# maxdepth, mindatain_leaf, bagging_functon, earlystopping ground, mingainto_split

params = {}
params["learning_rate"] = 0.02
params["boosting_type"] = "gbdt"
params["objective"] = "binary" 
params["metric"] = "binary_logloss" # measurement
params["max_depth"] = 5
params["num_leaves"] = 20
params["is_unbalance"] = True

X_train_lgb = lgb.Dataset(X_train, label = y_train)
lgb_clf = lgb.train(params, X_train_lgb, num_boost_round = 100)

y_pred = lgb_clf.predict(X_test)

# convert probability into binary number
for i in range(len(y_pred)):
    if(y_pred[i] >= 0.5): # setting the threshold
        y_pred[i] = 1
    else:
        y_pred[i] = 0
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# confusion matrix
confmat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.matshow(confmat, cmap = plt.cm.hsv, alpha = 0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x = j, y = i, s = confmat[i,j])
plt.xlabel('pred_result')
plt.ylabel('actu_result')
plt.show()

# show lightgbm feature importance
plt.figure(figsize=(12,6))
lgb.plot_importance(lgb_clf, max_num_features=10)
plt.title("Featurertances")
plt.show()
# the other way to plot feature importance of lightgbm model
feature_import = pd.DataFrame(sorted(zip(lgb_clf.feature_importance(), X.columns)), columns = ['Value', 'Feature'])
plt.figure(figsize = (20, 10))
sns.barplot(x = "Value", y = "Feature", data = feature_import.sort_values(by = "Value", ascending = False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

```


**try to eliminate the correlation among variables first** 

- check the interaction plot first




[Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)

[Recursive feature elimination](https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py)

[Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)


[Learning Curve](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py)

## Stacking
- Models that have their predictions combined must have skill on the problem, but do not need to be the best possible models. This means you do not need to tune the submodels intently, as long as the model shows some advantage over a baseline prediction. (base model output: uncorrelated prediction)

![](https://i.imgur.com/c4vgg9b.png)

- ```Level 0 (base models)``` : the training dataset as the input to make prediction
- ```Level 1 (meta model)``` : take the output of level 0 models as input and the single level 1 model, meta_learner, learns to make prediction from the data

- list of tuple for constructing base model
[python ensemble learning introduction](https://machinelearningmastery.com/super-learner-ensemble-in-python/)

- [stacking for deep learning](https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/)

```python=
# Stacking (adjust base model)
def get_stacking():
    base_model = list()
    base_model.append(('svr', SVR(kernel = 'rbf')))
    base_model.append(('cart', DecisionTreeRegressor()))
    base_model.append(('mars', Earth()))
    base_model.append(('extratree', ExtraTreeRegressor()))
    #base_model.append(('knn', KNeighborsRegressor()))
    # define meta learner
    meta_learner = LinearRegression()
    # define stacking ensemble
    model = StackingRegressor(estimators = base_model, final_estimator = meta_learner, cv = 5)
    return model

model = get_stacking()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Stacking")
print("MAPE: avg = {:.2f}%".format(mape(y_test, y_pred)))
print("model score: %.3f" % model.score(X_test, y_test))
print("RMSE: %.3f" % mean_squared_error(y_test, y_pred))
print("R2: %.3f" % r2_score(y_test, y_pred))
```


### Stack for classification problem
- better results have been seen when using the prediction of class probabilities as input ot the meta-learner instead of labels



## Quantile Regression
### quantile loss
- if (actual - predict) < 0
    -  $(\alpha-1)*(actual-predict)$ 
- if (actual - predict) > 0
    - $\alpha*(actual-predict)$
```python=
def calculate_quantile_loss(quantile, actual, predicted):
    """
    Quantile loss for a given quantile and prediction
    """
    return np.maximum(quantile * (actual - predicted), (quantile - 1) * (actual - predicted))
```
- light blue line shows the 10th percentile. We can think there's a 10 percent chance that the true value is below that predicted value (assign less of a loss to underestimates than to overestimates)
- dark blue line shows the 90th percentile
- medium blue line shows the median
![](https://i.imgur.com/mMGie2Q.png =500x)
![](https://i.imgur.com/R7CqdgA.png =500x)

- Example (4 cases with lower_alpha & upper_alpha)
1. Prediction = 15 with Quantile = 0.1. Actual < Predicted; Loss = (0.1 - 1) * (10 - 15) = 4.5
2. Prediction = 5 with Quantile = 0.1. Actual > Predicted; Loss = 0.1 * (10 - 5) = 0.5
3. Predicted = 15 with Quantile = 0.9. Actual < Predicted; Loss = (0.9 - 1) * (10 - 15) = 0.5
4. Predicted = 5 with Quantile = 0.9. Actual < Predicted; Loss = 0.9 * (10 - 5) = 4.5

[introduciton to quantile regression](https://towardsdatascience.com/quantile-regression-from-linear-models-to-trees-to-deep-learning-af3738b527c3)


## Hyperparameter
- [Complete guide parameter tuning xgboost with code](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

## Customized loss function

- [view the cython code in criterion.pyx](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx)
- [The situation we need to customize loss function (lightGBM for this article)](https://towardsdatascience.com/custom-loss-functions-for-gradient-boosting-f79c1b40466d)
- [LightGBM vs. XGBoost vs. CatBoost](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db)
- [check loss function](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0)




### RANSAC
- RANSAC is an iterative algorithm for the robust estimation of parameters from a subset of inliers from the complete data set
- also can be interpreted as outlier detection method
- [RANSAC Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html)


### out-of-fold prediction
- a type of out-of-sample predictions



[Use Keras Deep Learning Models with Scikit-Learn in Python](https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/)
[Skorch for pytorch wrapper](https://github.com/skorch-dev/skorch)