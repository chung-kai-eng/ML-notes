###### tags: `python`


Metrics for Model Evaluation
===

[SVR theory 暫放](https://zhuanlan.zhihu.com/p/33692660)

[multi-class classification measurement overview](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/)


## Non-symmetric error
- Overestimate and underestimate have different impact on forecast, so this fact should be taken into account to make the evaluation of the model as much as possible correlated with its usefulness
## MAE (Mean Absolute Error)
## <center> MAE : $\frac{1}{n}\Sigma|a_i - x_i|$ </center>

## MSE (Mean Square Error) and RMSE(Root Mean Square Error)
## <center> MSE : $\frac{1}{n}\Sigma(a_i - x_i)^2$ </center>
## <center> RMSE : $\sqrt{\frac{1}{n}\Sigma(a_i - x_i)^2}$ </center>



## When large variation
## MAPE and SMAPE
[Comparison between MAPE & SMAPE](https://trsekhar-123.medium.com/mape-vs-smape-when-to-choose-what-be51a170df16)
- When true value is close to zero, the value would be infinity
- asymmetric error (對負誤差的penalty更大，更偏向於預測不足的模型)

## <center> MAPE: $\frac{1}{n}\Sigma \frac{|(Actual - Predicted)|}{Actual}$ </center>
## <center> SMAPE: $\frac{1}{n}\Sigma \frac{|(Actual - Predicted)|}{(Actual + Predicted)/2}$ </center>
- SMAPE deals with one kind of asymmetric nature, but it creates another asymmetric value because of denominator while we overpredict and underpredict
- SMAPE will be higher if we underpredict compared to overprediction
- it's safer to use SMAPE if there is more sparsity in data, else MAPE is good metric to check the accuracy.

## MASE (mean absolute scaled error)
- a measure of the accuracy of forecasting
- $MASE = mean(\frac{|e_j|}{\frac{1}{T-1}\Sigma^T_{t=2}|Y_t-Y_{t-1}|})$
## MDA (mean directional accuarcy)
- a measure of prediciton accuracy of a forecasting
- $\frac{1}{N}\Sigma_t I_{sgn}(A_t-A_{t-1})=sgn(F_t-A_{t-1})$
    - $A_t$: actual value at time t
    - $F_t$: forecast value at time t$
    - $sgn$: Sign function
    - ![](https://i.imgur.com/cI7blZV.png =200x)
    - ![](https://i.imgur.com/uToVUMa.png)


how to evaluate and compare models that forecast some numerical values


## G means/ F1 score/ AUC, ROC curve, MCAUC (Mean column-wise AUC for multi-class classification)

## ROC/ AUC curve

```python=
def plot_roc_curve(y_test, preds):
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    gmeans = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(gmeans)
    print("Best threshold: {}".format(threshold[idx]))
    plt.figure(figsize = (5, 5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
```



## F2-score/ F0.5 score/ F1 score (F beta-measure)
- precision and recall measure the two types of errors that could be made for the positive class
- maximize precision minimize false positives
- maximize recall minimize false negative
$$
\beta = 0.5, 1, 2 \\
Fbeta = ((1+\beta^2)*precision*recall)\ / \ (\beta^2*precision+recall) \\
$$
- F0.5-Measure: more weight on precision, less weight on recall
- F1-Measure: balance the weight on precision and recall
- F2-Measure: less weight on precision, more weight on recall

$$
F0.5 = (1.25*precision*recall) \  / \ (0.25*precision+recall) \\
F1 = (2*precision*recall)\ / \ (precision+recall) \\
F2 = (5*precision*recall)\  / \ (4*precision+recall)
$$



## MDA (mean directional accuracy)



## Matthews correlation coefficient
$$
MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$
[Matthews correlation coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient#Multiclass_case)

- as a measure of the quality of binary classifications
- also can be extended to multi-class
- **advantage of MCC over accuracy and F1 score** (when data is imbalanced, we might misunderstanding the performance when viewing accuracy & F1 score)





## The way to inspect which features are overfitting
- [Link for tutorial](https://towardsdatascience.com/which-of-your-features-are-overfitting-c46d0762e769)
- **Feature importance says nothing about how the features will perform on new data**
- a pattern should be gerneral enough to hold true also on new data

```python=
# CatBoost
cat = CatBoostClassifier(silent = True).fit(X_train, y_train)
# Show feature importance
fimpo = pd.Series(cat.feature_importances_, index = X_train.columns)
fig, ax = plt.subplots()
fimpo.sort_values().plot.barh(ax = ax)
fig.savefig('fimpo.png', dpi = 200, bbox_inches="tight")
fig.show()
```



```python=+
from catboost import CatBoostClassifier, Pool


shap_train = pd.DataFrame(
                data = cat.get_feature_importance(
                data = Pool(X_train), 
                type = 'ShapValues')[:, :-1], 
                index = X_train.index, 
                columns = X_train.columns
            )
shap_test = pd.DataFrame(
                data = cat.get_feature_importance(
                data = Pool(X_test), 
                type = 'ShapValues')[:, :-1], 
                index = X_test.index, 
                columns = X_test.columns
)
```
- original data vs. corresponding shape values (log_odds)
![](https://i.imgur.com/nDurMWk.png)


- the idea for measuring the performance of a feature on a dataset is to **compute the correlation between the SHAP values of the feature and the target variable**
- If the model has found good patterns on a feature, the SHAP values of that feature must be highly positive correlated with the target variable

- compute the correlation between shap_values & target variable
```python=
np.corrcoef(shap_test['docvis'], y_test)
```
- Due to SHAP values are additive, meaning that the final prediction is the sum of the SHAPs of all the features. Thus, remove the effect of other features before calculating the correlation $\to$ exactly the definition of **partial correlation**

- **```pingouin```** package
```python=
import pingouin
pingouin.partial_corr(
  data = pd.concat([shap_test, y_test], axis = 1).astype(float), 
  x = 'docvis', 
  y = y_test.name,
  x_covar = [feature for feature in shap_test.columns if feature != 'docvis'] 
)
```
**Partial correlation of SHAP values** (ParShap)
- We can repeat the procedure for each feature, both on train & test set
```python=
from pingouin import partial_corr
# Define function for partial correlation
def partial_correlation(X, y):
  out = pd.Series(index = X.columns, dtype = float)
  for feature_name in X.columns:
    out[feature_name] = partial_corr(
      data = pd.concat([X, y], axis = 1).astype(float), 
      x = feature_name, 
      y = y.name,
      x_covar = [f for f in X.columns if f != feature_name] 
    ).loc['pearson', 'r']
  return out
```

- According to the scatter plot, we can view each feature's partial correlation both on training and testing set

```python=
parshap_train = partial_correlation(shap_train, y_train)
parshap_test = partial_correlation(shap_test, y_test)

plt.scatter(parshap_train, parshap_test)

parshap_diff = parshap_test - parshap_train
```

```python=+                    
# Plot parshap
def plot_parshap_train_test(parshap_train, parshap_test, fimpo=None):
    # Plot parshap
    plotmin, plotmax = min(parshap_train.min(), parshap_test.min()), max(parshap_train.max(), parshap_test.max())
    plotbuffer = 0.05 * (plotmax - plotmin)
    fig, ax = plt.subplots(figsize=(20, 20))
    if plotmin < 0:
        ax.vlines(0, plotmin-plotbuffer, plotmax+plotbuffer, color='darkgrey', zorder=0)
        ax.hlines(0, plotmin-plotbuffer, plotmax+plotbuffer, color='darkgrey', zorder=0)
    ax.plot(
        [plotmin-plotbuffer, plotmax+plotbuffer], [plotmin-plotbuffer, plotmax+plotbuffer], 
        color='darkgrey', zorder=0
    )
    sc = ax.scatter(
        parshap_train, parshap_test, 
        edgecolor='grey', c=fimpo, s=10, cmap=plt.cm.get_cmap('Reds'), vmin=0, vmax=fimpo.max())
    ax.set(title='Partial correlation bw SHAP and target...', xlabel='... on Train data', ylabel='... on Test data')
    cbar = fig.colorbar(sc)
    cbar.set_ticks([])
    for txt in parshap_train.index:
        ax.annotate(txt, (parshap_train[txt], parshap_test[txt]+plotbuffer/2), ha='center', va='bottom')
    fig.savefig('parshap.png', dpi = 300, bbox_inches="tight")
    fig.show()
    
```
![](https://i.imgur.com/sIOJI5t.png)



- ```parshap_diff```: the more negative the score, the more overfitting is brought by the feature

- this way only a test to check the correctiness of our line of reasoning. Parshap should **not be used as a method for feature selection**. The fact that some features are prone to overfitting does **not imply that those features don't carry useful information at all**
- ParShap proves extremely helpful in giving us hints on how to debug our model. It allows us to **focus the attention on those features that require some feature engineering or regularization**

### More explanation on **partial correlation**
- Venn Diagrams for interpreting each variable's correlation with target variable
![](https://i.imgur.com/g76MgOf.png)
- correlation
![](https://i.imgur.com/7kpkxDN.png)
- partial correlation
![](https://i.imgur.com/hR9NqbW.png)
- semi-partial correlation