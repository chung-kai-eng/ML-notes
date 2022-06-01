###### tags: `python` `Machine Learning`

Regularized Regression & MARS & PLS & kernel overview
===
[statistic package for python](https://www.statsmodels.org/stable/examples/index.html#robust-regression)

https://bradleyboehmke.github.io/HOML/mars.html
## regularized regression
![](https://i.imgur.com/cdBgFN6.png)

1. Ridge 
![](https://i.imgur.com/MXf9Kyk.png)

2. Lasso 
-  lasso can be used to identify and extract those features with the largest (and most consistent) signal.
![](https://i.imgur.com/cRRcaNn.png)

3. Group Lasso
[Group Lasso](https://leimao.github.io/blog/Group-Lasso/)

4. Elastic Net
![](https://i.imgur.com/cXffBR5.png)

### the difference between Lasso and Ridge
[Ridge vs. Lasso](https://www.youtube.com/watch?v=Xm2C_gTAl8c)
- having a knit in lasso(which is equal to 0)
- For Ridge regression, the larger the lambda is, the slope(coefficient) will more close to 0
![](https://i.imgur.com/nHXZsdP.png)

![](https://i.imgur.com/shRDZLd.png)

## MARS(Multivariate Adaptive Regression Spline)
![](https://i.imgur.com/hlP3Yfd.png)


## PLS (partial least square)
- suitable for the case that the number of predictors is larger than that of observations (p >> n)
- supervised learning considering the projection to latent structure

[loading plot for PLS like PCA](https://stackoverflow.com/questions/56477144/pls-da-loading-plot-in-python)

[interpretation of PLS](https://learnche.org/pid/latent-variable-modelling/projection-to-latent-structures/interpreting-pls-scores-and-loadings)

[sample code for PLS](http://www.science.smith.edu/~jcrouser/SDS293/labs/lab11-py.html)

[variable selesction for PLS](https://nirpyresearch.com/variable-selection-method-pls-python/)

## Gaussian Process Regression vs. Kernel Ridge regression
[GPR vs. KRR](https://scikit-learn.org/stable/modules/gaussian_process.html)
![](https://i.imgur.com/fLFwml2.png)

[Different Kernel for GPR](https://scikit-learn.org/0.24/auto_examples/gaussian_process/plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py)

[kernel introduction](https://www.cs.toronto.edu/~duvenaud/cookbook/)

## Residual (GLS influential points)
[influential point for glm model](https://www.statsmodels.org/stable/examples/notebooks/generated/influence_glm_logit.html)


## SVR
[Different SVR models](https://www.mdpi.com/1424-8220/20/23/6742/htm)
### $\epsilon$-SVR
- minimize the $\epsilon$-insensitive loss function along with the $\frac{1}{2}w^Tw$ regularization term, where $|y_i-f(x_i)_\epsilon|=max(0, |y_i-f(x_i)|-\epsilon$ is the $epsilone$-insensitive loss function

 
$min_{w, b}\frac{1}{2}w^Tw+C\Sigma_{i=1}^l(\kappa_i+\kappa_i^*)$
$subject \ to$ 
$y_i-(A_iw+b)<=\epsilon+\kappa_i$
$(A_iw+b)-y_i<=\epsilon+\kappa_i^*, \ \ \kappa_i, \kappa_i^*>=0$


### Least Squared SVR
- minimize the quadratic loss function along with the $\frac{1}{2}w^Tw$ regularization term

$min_{w, b}\frac{1}{2}w^Tw+C\Sigma_{i=1}^l(y_i-f(x_i))^2$

$min_{w, b, \xi}\frac{c}{2}||w||^2+C_1\Sigma_{i=1}^l(\xi_i^2)$
$subject \ to$
$y_i-(A_iw+b)=\xi_i, \ \ i=1, 2, ..., l$


### Huber SVR
- use huber loss (改善Squared loss function 對outlier的robustness)
- ![](https://i.imgur.com/wGREOYu.png)
- ![](https://i.imgur.com/cZ3Iido.png)
- [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
    - classification loss function: hinge, log, modified_huber, squared_hinge
    - regression loss function: squared_error, huber, epsilon_insensitive
        - log loss: give logistic regression
        - modified_huber: smooth loss that brings to tolerance to outliers as well as prbability estimates



### Twin SVR (TSVR)
- TSVR estimates two non-parallel hyperplanes by solving two quadratic progeamming problems (QPP)






## Focal loss
- 針對 easy example進行down-weighting，因為focal loss希望在訓練過程中盡量去訓練hard example