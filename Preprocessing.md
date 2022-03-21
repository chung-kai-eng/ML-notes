###### tags: `python`

Preprocessing
===

[machine learning issue and solution](https://machinelearningmastery.com/faq/single-faq/what-feature-selection-method-should-i-use/)
[cheat sheet for pandas](http://datacamp-community-prod.s3.amazonaws.com/d4efb29b-f9c6-4f1c-8c98-6f568d88b48f)
[Framework for data preparation](https://machinelearningmastery.com/framework-for-data-preparation-for-machine-learning/)
![](https://i.imgur.com/vZnNYhQ.png)

## ```String``` find, split
[regular expression](https://regexr.com/)

## feature engine
- [feature engine library for preprocessing](https://feature-engine.readthedocs.io/en/1.1.x/)
    - include feature selection , missing value imputation, outlier handling, variable transformation, variable discretization

### drop quasi-constant features (filter out features with 99% same value)
```python=
from feature_engine.selection import DropDuplicateFeatrues, DropConstantFeatures

drop_quasi = DropDuplicateFeatrues(tol=0.99, variables=None, missing_values='raise')
drop_quasi.fit_transform(training_data)
```


## model storage file type
![](https://i.imgur.com/9Voq4TZ.png)

## Encoded
- [category encoder](https://contrib.scikit-learn.org/category_encoders/)
- [Encode guideline python](https://pbpython.com/categorical-encoding.html)
- [performance of different encode method](http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/)
### One Hot encoded issue (=treatment coding: drop one columns)
- drop = "first" or handle_unknown = "ignore" can't happen simultaneously
    - when just doing model preprocessing and fit the model, the better way to deal with is use drop = "first" (encode -> train_test_split)
        - Sometimes, it might have some problems when doing train_test_split(some categories that exist in testing set don't exist in training set (if drop='first', it will raise error))
        - if handle_unknown="ignore", set all unknown categoies as reference line, which might lead to bias when lots of categories being regarded as reference line 
    - when doing cross_validation, we can just compromise to use handle_unknown = "ignore"

### Sum coding
- when doing train_test_split (sometimes there might be some categories not in training set but in testing set). In this situation, sum coding method is good for estimate the effect of each categories


- using column transformer to make pipeline 
```python=
def preprocessing(self):
    numeric_transformer = Pipeline(steps = [('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps = [("one_hot", OneHotEncoder(drop = "first"))])

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numeric_transformer, self.numeric_features),
            ('cat', categorical_transformer, self.categorical_features)], remainder = "passthrough")

    self.X = preprocessor.fit_transform(self.X).toarray() # encoder remember to change type if need to change to dataframe
    cat_col_name = list(preprocessor.transformers_[1][1].named_steps["one_hot"].get_feature_names(self.categorical_features))
    col_name = self.numeric_features + cat_col_name
    self.X = pd.DataFrame(self.X, columns = col_name)
    return self
```


[](https://datascience.stackexchange.com/questions/14025/ordinal-feature-in-decision-tree)
- ordinal variable & categorical variable are treated exactly the same in GBM. However, the result will almost likely be **different** if you have ordinal data formatted as categorical.

## Feature selection
![](https://i.imgur.com/lXcPv6g.png)
- filter
    - statistical methods
    - feature importance method
- wrapper: 
    - [Genetic algorithm for feature selection (sklearn)](https://towardsdatascience.com/feature-selection-with-genetic-algorithms-7dd7e02dd237)
    - RFE

- hybrid
## Correlation
![](https://i.imgur.com/AzR8ZdL.png)

https://machinelearningmastery.com/statistical-hypothesis-tests/
https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/

### Categorical vs. Categorical (Cramer's V) or (Theil's U/ uncertainty coefficient)
[Cramer's V & Theil's U](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9)
- Cramer's V: correction if the number of category or the data is large 
    - divided by min(row-1, col-1)
    - divided by n (the number of observatoin)
- Theil's U: asymmetric approach to compute the correlation between two categorical variables
    -  $U(X|Y) = \frac{H(X)-H(X|Y)}{H(X)} = I(X;Y)/H(X)$

![](https://i.imgur.com/LEiDa7Y.png)
![](https://i.imgur.com/5sgv1s2.png)

[](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9)

- Two variables with Gaussian distribution: **Pearson’s correlation**
- Variables that doesn't have Gaussian distribution: **Rank correlation** methods must be used
- Based on row, we can know how much about column 
<center> 

![](https://i.imgur.com/yND1E7O.png =400x)

</center>

```python=
def qual_corr_asymmetric_test(x, y, log_base = math.e):
    """
    Theil_u:  U(X|Y) = (H(X) - H(X|Y)) / H(X) = I(X;Y) / H(X)  (mutual information / X's entropy)
    pay attention to the H(X|Y) not H(Y|X) if denominator is H(X)
    asymmetric correlation between two categorical variables 
    U(x,y) not equal to U(y, x)                                                                                                
    """
    def conditional_entropy(x, y, log_base = math.e): # condition on x: H(Y|X)
        """
        compute the conditional entropy between two features
        Parameters
        ----------
        x : {list} a feature in a dataframe
        y : {list} a feature in a dataframe
        Returns
        cond_entropy: {float} the value of conditional entropy 
        -------
        """
        y_counter = Counter(y) # Counter: dict subclass for counting hashable objects  表示各事件發生次數
        xy_counter = Counter(tuple(zip(x, y))) # mutual information 
        total_event = sum(y_counter.values()) # total events used to calculate proportion
        cond_entropy = 0
        for xy in xy_counter.keys():
            py = y_counter[xy[1]] / total_event # condition on x, so xy[0]
            pxy = xy_counter[xy] / total_event
            cond_entropy += pxy * math.log(py / pxy, log_base) # -log(a/b) = log(b/a)
        return cond_entropy  

    h_xy = conditional_entropy(x, y, log_base)
    x_counter = Counter(x)
    total_event = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_event, x_counter.values())) # proportion calculation
    h_x = stats.entropy(p_x, base = log_base) # calcuate x's entropy
    if h_x == 0: # means all x are same (if we know y, then know all x)
        return 1
    else:
        return (h_x - h_xy) / h_x

def categorical_corr_mat(df, qual_attr, title = None, test_type = "Cramer", log_base = math.e):
    corr_mat = pd.DataFrame(columns = qual_attr, index = qual_attr)
    if (test_type == "Cramer"):
        for col_name in combinations(qual_attr, 2):
            print(col_name)
            corr_mat[col_name[0]][col_name[1]] =  cramer_v(df[col_name[0]], df[col_name[1]]) 
            corr_mat[col_name[1]][col_name[0]] =  corr_mat[col_name[0]][col_name[1]]    # symmetric    
    else: # Theil's U theorem
        for col_name in permutations(qual_attr, 2):
            corr_mat[col_name[0]][col_name[1]] =  qual_corr_asymmetric_test(df[col_name[0]], df[col_name[1]], log_base)       
    for col_name in qual_attr:
        corr_mat[col_name][col_name] = 1.0
    plt.figure(figsize = (15, 15))
    plt.title(title)
    corr_mat = corr_mat.astype(float)
    sns.heatmap(data = corr_mat, annot = True, annot_kws = {"size" : 25})  

```

### Rank correlation
- methods that quantify the association between variables using the ordinal relationship between the values rather than the specific values. Ordinal data is data that has label values and has an order or rank relationship; for example: **low**, **medium**, and **high**.
- quantify the association between the two ranked variables
- referred to as disribution-free correlation or nonparametric correlation
![](https://i.imgur.com/suxC12c.png)

- using pearson correlation and get highly correlated pair and plot
    - which will include spearman correlation (no need)
```python=
######### List the highest correlation pairs from a large correlation matrix in pandas ###################
def get_high_corr_feature_pair():
    corr = df.corr(method='pearson')
    corr_pair = corr.abs().unstack() # absolute value of correlation coefficient
    corr_pair = corr_pair.sort_values(ascending=False, kind='quicksort')

    # high correlated pair (filter the correlation that >=0.8 & <1)
    high_corr = corr_pair[(corr_pair >= 0.8) & (corr_pair < 1)]
    # convert to dataframe and set columns
    high_corr = pd.DataFrame(high_corr).reset_index()
    high_corr.columns = ['feature1', 'feature2', 'corr']
    return high_corr
```

### Nonlinear Correlation Coefficient (NCC)
- a method based on mutual information (a quantity measuring the relationship between two discrete random variables)
- [Intrusion detection method based on nonlinear correlation measure](https://opus.lib.uts.edu.au/bitstream/10453/33842/5/Intrusion%20detection%20method%20based%20on%20nonlinear%20correlation%20measure.pdf)
$$
I(X; Y) = H(X) + H(Y) - H(X,Y) \\
H(X,Y)=-\Sigma_{i=1}^n\Sigma_{j=1}^nP(x_i,y_j)lnP(x_i,y_j)
$$
- the disadvantage of MI is that is doesn't range in a definite closed interval [-1, 1] as the correlation coefficient. Thus,  Wang et al. (Wang et al., 2005) developed a revised version of the MI, named nonlinear correlation coefficient (NCC)
$$
H^r(X,Y)=-\Sigma_{i=1}^b\Sigma_{j=1}^b \frac{n_{ij}}{N}log_b\frac{n_{ij}}{N}
$$
- bxb rank grids are used to place the sample
$$
NCC(X;Y)=H^r(X) + H^r(Y) - H^r(X,Y) \\
H^r(X)=-\Sigma_{i=1}^b \frac{n_{ij}}{N}log_b\frac{n_{ij}}{N} \\
H^r(X)=-\Sigma_{j=1}^b \frac{n_{ij}}{N}log_b\frac{n_{ij}}{N} \\
NCC(X;Y)= 2+\Sigma_{i=1}^b\Sigma_{j=1}^b \frac{n_{ij}}{N}log_b\frac{n_{ij}}{N}
$$
- **-1: weakest relationships, 1: strongest relationships**
- also a symmetric matrix (diagonal are equal to one)
- Application for anomaly detection: $|\overline {NCC}^n| - |\overline {NCC}^{n, n+1}|>\sigma$

## Missing Value issue
### five measures to deal with missing value
- [five types of measure](https://analyticsindiamag.com/5-ways-handle-missing-values-machine-learning-datasets/)
1. delete rows
2. replace with mean, median, mode
3. assigned as an unique category 
4. predict missing value
5. use algorithms which support missing value
    - In scikit-learn library for the KNN in python doesn't support the presence of the missing values

### Iterative Imputer
- [Iterative Imputation for Missing Values in Machine Learning(using xgboost, bayesian ridge...)](https://machinelearningmastery.com/iterative-imputation-for-missing-values-in-machine-learning/)

### Using KNN, SoftImputer, IterativeSVD, MICE, MatrixFactorization, NuclearNormMinimization
- [fancyimpute](https://pypi.org/project/fancyimpute/)
- [github source](https://github.com/iskandr/fancyimpute)
- **KNN**: Nearest neighbor imputations which weights samples using the mean squared difference on features for which two rows both have observed data.
- **MICE**: Reimplementation of Multiple Imputation by Chained Equations.
- IterativeSVD: Matrix completion by iterative low-rank SVD decomposition. Should be similar to SVDimpute from Missing value estimation methods for DNA microarrays by Troyanskaya et. al.
- Matrix completion by iterative soft thresholding of SVD decompositions which is based on Spectral Regularization Algorithms for Learning Large Incomplete Matrices by Mazumder et. al.
- **BiScaler**: Iterative estimation of row/column means and standard deviations to get doubly normalized matrix. Not guaranteed to converge but works well in practice. Taken from Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares.

## Imbalanced data 
- [Imbalanced method API](https://imbalanced-learn.org/stable/references/index.html)
- [Data Sampling Methods for Imbalanced Classification](https://machinelearningmastery.com/data-sampling-methods-for-imbalanced-classification/)
- [5 SMOTE techniques](https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5)
- undersampling(T-link)
- downsampling and upweighted 
[tutorial from google](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data)

![](https://i.imgur.com/79qSjGk.png)
![](https://i.imgur.com/7bvhYcT.png)


## Columntransformer
```python=
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({'brand': ['aaaa', 'asdfasdf', 'sadfds', 'NaN'],
                   'category': ['asdf', 'asfa', 'asdfas', 'as'],
                   'num1': [1, 1, 0, 0],
                   'target': [0.2, 0.11, 1.34, 1.123]})

numeric_features = ['num1']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['brand', 'category']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor',  LinearRegression())])
clf.fit(df.drop('target', 1), df['target'])

clf.named_steps['preprocessor'].transformers_[1][1]\
   .named_steps['onehot'].get_feature_names(categorical_features)
```
## Pipeline
[Column Transformer with Mixed Types](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)

- remember to do train_test_split before preprocessing(data leakage problem)
[pipeline example](https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf)


https://en.wikipedia.org/wiki/Multidimensional_scaling

## Gower distance
- Gower distance
    - $\frac{1}{K}\sum_{n=1}^{K}\frac{|X_{ik}-X_{jk}|}{R_k}$
- qualitative variable:
    - ordinal variable
        - $1^{st}$ rank
        - $2^{nd}$ Manhatten distance
        - $R_k=range(trait_k)=max_k-min_k$
    - nominal variable
        - $|X_{ik}-X_{jk}|=I[X_{ik}=X_{jk}]$
        - if equal = 0, else = 1
        - $R_k=1$

- quantitative variable (Manhatten distance)
    - $\frac{|X_{ik}-X_{jk}|}{R_k}$
    - $R_k=range(trait_k)=max_k-min_k$

### The way to check the variable type of columns
```python=
for col in range(x_n_cols):
    # check the type of the column, if the type of column is np.number, then return False
    if not np.issubdtype(type(X[0, col]), np.number):
        cat_features[col] = True
    print("cat_features: {}".format(cat_features()))
```
### Code to compute Gower distance
```python=

import gower # can use the package directly
import pandas as pd
import numpy as np

df = pd.DataFrame({"gender": ["M", "F", "M", "F"],
                   "age": [20, 30, 50 ,40]})
# no need to transform the categorical variable into numerical value
df1 = pd.DataFrame({"gender": [0, 1, 0, 1],
                   "age": [20, 30, 50 ,40]})

         
def gower_get(xi_cat, xi_num, xj_cat, xj_num, feature_weight_cat,
              feature_weight_num, feature_weight_sum, categorical_features,
              ranges_of_numeric, max_of_numeric):
    
    # categorical columns
    sij_cat = np.where(xi_cat == xj_cat, np.zeros_like(xi_cat), np.ones_like(xi_cat))
    sum_cat = np.multiply(feature_weight_cat, sij_cat).sum(axis = 1) 

    # numerical columns
    abs_delta = np.absolute(xi_num - xj_num)
    sij_num = np.divide(abs_delta, ranges_of_numeric, out = np.zeros_like(abs_delta), where=ranges_of_numeric!=0)
    #print("sij_num: {}".format(sij_num))
    sum_num = np.multiply(feature_weight_num,sij_num).sum(axis = 1)
    sums = np.add(sum_cat,sum_num)
    sum_sij = np.divide(sums,feature_weight_sum)
    return sum_sij            

def gower_matrix(data_x, data_y = None, weight = None, cat_features = None):  
    # function checks
    X = data_x
    if data_y is None: Y = data_x 
    else: Y = data_y 
    if not isinstance(X, np.ndarray): 
        if not np.array_equal(X.columns, Y.columns): raise TypeError("X and Y must have same columns!")   
    else: 
         if not X.shape[1] == Y.shape[1]: raise TypeError("X and Y must have same y-dim!")  
                
    #if issparse(X) or issparse(Y): raise TypeError("Sparse matrices are not supported!")        
            
    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape 
    
    if cat_features is None:
        if not isinstance(X, np.ndarray):           
            is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
            print("is_number: {}".format(is_number))
            cat_features = is_number(X.dtypes)   
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            print("cat_features: {}".format(cat_features()))

            for col in range(x_n_cols):
                # check the type of the column, if the type of column is np.number, then return False
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col] = True
            print("cat_features: {}".format(cat_features()))

    else:          
        cat_features = np.array(cat_features)
    
    # print(cat_features)
    print("cat_features: {}".format(cat_features))
   
    if not isinstance(X, np.ndarray): X = np.asarray(X)
    if not isinstance(Y, np.ndarray): Y = np.asarray(Y)
    print("X: {}".format(X))
    Z = np.concatenate((X,Y))
    
    x_index = range(0, x_n_rows)
    y_index = range(x_n_rows, x_n_rows+y_n_rows)
    
    Z_num = Z[:,np.logical_not(cat_features)]
    print("Z_num: {}".format(Z_num))
    num_cols = Z_num.shape[1]
    print("Z_num.shape: {}".format(Z_num.shape))
    num_ranges = np.zeros(num_cols)
    num_max = np.zeros(num_cols)
    
    for col in range(num_cols):
        col_array = Z_num[:, col].astype(np.float32) 
        max = np.nanmax(col_array) # ignore nan find the maximum value of col
        min = np.nanmin(col_array)
     
        if np.isnan(max):
            max = 0.0
        if np.isnan(min):
            min = 0.0
        num_max[col] = max
        num_ranges[col] = (1 - min / max) if (max != 0) else 0.0
    print("num_range: {}".format(num_ranges))
    # This is to normalize the numeric values between 0 and 1.
    Z_num = np.divide(Z_num ,num_max,out = np.zeros_like(Z_num), where=num_max!=0)
    print("Z_num normalization: {}".format(Z_num))
    Z_cat = Z[:,cat_features]
    print("Z_cat: {}".format(Z_cat))
    # equal weight for quantitative variable
    if weight is None:
        weight = np.ones(Z.shape[1])
        
    #print(weight)    
    
    weight_cat = weight[cat_features]
    weight_num = weight[np.logical_not(cat_features)]   
    print("weight_categorical: {}".format(weight_cat))
    print("weight_num: {}".format(weight_num))
    out = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)
        
    weight_sum = weight.sum()
    
    X_cat = Z_cat[x_index,]
    X_num = Z_num[x_index,]
    Y_cat = Z_cat[y_index,]
    Y_num = Z_num[y_index,]
    
    # print(X_cat,X_num,Y_cat,Y_num)
    print("X_cat: {}".format(X_cat))
    # calculate the gower distance row by row 
    for i in range(x_n_rows):          
        j_start = i        
        if x_n_rows != y_n_rows:
            j_start = 0
        # call the main function
        print("X_cat[i,:]: {}".format(X_cat[i,:]))
        res = gower_get(X_cat[i,:], 
                        X_num[i,:],
                        Y_cat[j_start:y_n_rows,:],
                        Y_num[j_start:y_n_rows,:],
                        weight_cat,
                        weight_num,
                        weight_sum,
                        cat_features,
                        num_ranges,
                        num_max) 
        #print(res)
        print("res: {}".format(res))
        out[i, j_start:] = res
        if x_n_rows == y_n_rows: 
            out[i:,j_start] = res
    return out    
    

print(gower_matrix(df))
```





[mutual information feature selection](https://machinelearningmastery.com/feature-selection-for-regression-data/)

[paper for mutual information(numeric and discrete variable)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3929353/)



## Hypothesis test summary
![](https://i.imgur.com/pGaQ2O1.png)

![](https://i.imgur.com/WsGd89t.png)


- ANOVA vs. MANOVA
    - MANOVA: when there's more than two dependent variables
- ANOVA vs. T-test
    - when there's only two groups, then t-test will have the same p-value as ANOVA
    - when more than two groups, use ANOVA
- MANOVA vs. Hotelling's T squared
    - when there's only two groups(independent variable), then Hotelling's T squared is easier to calculate 

#### Hotelling's T2
- based on the Mahalanobis distance between the group centroids
    - ![](https://i.imgur.com/tzdRCTH.png)
    - takes into account the relationship between the dependent variables in its calculations
    - $T^2 = \frac{n_1n_2}{n_1+n_2}(MD)^2$
        - MD:Mahalanobis distance
    - $F=\frac{n_1+n_2-p-1}{p(n_1+n_2-2)}T^2$
- difficulty: calculate mahalanobis distance
- after doing Hotelling's T2, do t-test seperately