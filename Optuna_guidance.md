Optuna
===

## Comparison between ```Hyperopt``` and ```Optuna```
- Hyperopt should **predefine the search space** using **dictionary** to setup
- Optuna can use any python syntax to set up search space (e.g. ```.yml```, ```dict```)
<img src=https://user-images.githubusercontent.com/54303314/193978439-d7a86648-6c28-4e26-904b-2e207fc1162a.png width="800" height="450">

# Hyperparameter algorithm containes two parts of strategy 
1. Sampling Strategy 
2. Pruning Strategy

## Samplers
- [Overview](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#sphx-glr-tutorial-10-key-features-003-efficient-optimization-algorithms-py)
- Model-based
- Other method
### Model-based 
1. TPE: kernel fitting (tree-based estimator which can deal with both numerical and categorical features)
2. GP: Gaussian process (suitable for numerical features)
3. CMA-ES (covariant matrix adaptation-evolution strategy): meta-heuristics algorithm for continuous space (a kind of Genetic algorithm)
4. Simulated Annealing

### Other method
1. Random Search
2. Grid Search
3. User-defined algorithm

<img src=https://user-images.githubusercontent.com/54303314/193965369-15ef0332-a00b-41df-984c-7769f49f3a77.png width="500" height="270">
- A **hybrid sampler** largely improves optimization performance (```TPE``` then ```CMA-ES```)
    - Step 1: Global search with TPE
    - Step 2: Local search with CMA-ES

## Pruning Strategy (a.k.a. automated early stopping)
- Stop unpromising trials based on learning curve (can let computing resource dedicate to more promising trials)
- Median pruning (```Median Pruner```), non-pruning (```NonPruner```), asynchronous successive halving algorithm (```SuccessiveHalvingPruner```), hyberband, etc.
- default: ```Median Pruner``` although basically it is outperformed by ```SuccessiveHalvingPruner``` and ```HyberbandPruner```


## Template for ```Optuna``` hyperparameter
- [Examples for different models (LightGBM, pytorch, tensorflow)](https://github.com/optuna/optuna-examples)
```python=
import optuna

def objective(trial):
    # write your code
    
    return evaluation_score
   
study = optuna.create_study()
study.optimize(objective, n_trials=<number of trials>)
```
- If you want to do pruning
```python=
def objective(trial):
    # write your code

    trial.report(accuracy, epoch)
    # Handle pruning based on the intermediate value
    if trial.shoud_prune():
        raise optuna.exceptions.TrialPruned() # let the function be exited
    
    return accuracy
```

- Another way to set up hyperparameter (using dictionary)
```python=
# model can be seperated from objective using the example shown below
def return_score(param):
    model = xgb.XGBRegressor(**param)  
    rmse = -np.mean(model_selection.cross_val_score(model,X_train[:1000],y_train[:10000], cv = 4, n_jobs =-1,scoring='neg_root_mean_squared_error'))
    return rmse


def objective(trial):
    param = {
                "n_estimators" : trial.suggest_int('n_estimators', 0, 500),
                'max_depth':trial.suggest_int('max_depth', 3, 5),
                'reg_alpha':trial.suggest_uniform('reg_alpha',0,6),
                'reg_lambda':trial.suggest_uniform('reg_lambda',0,2),
                'min_child_weight':trial.suggest_int('min_child_weight',0,5),
                'gamma':trial.suggest_uniform('gamma', 0, 4),
                'learning_rate':trial.suggest_loguniform('learning_rate',0.05,0.5),
                'colsample_bytree':trial.suggest_uniform('colsample_bytree',0.4,0.9),
                'subsample':trial.suggest_uniform('subsample',0.4,0.9),
                'nthread' : -1
            }
    return(return_score(param)) # this will return the rmse score
```

## Which Sampler and Pruner should be used?
- For not deep learning tasks,
    - For ```RandomSampler```, ```MedianPruner``` is the best
    - For ```TPESampler```,  ```Hyberband``` is the best
    - [Ref](https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako)
- For deep learning tasks,
<img src=https://user-images.githubusercontent.com/54303314/193975227-5e4fc778-7c9b-4159-858d-a0a4a42919b3.png width="900" height="250">


Reference: [ref](https://optuna.readthedocs.io/en/stable/tutorial/index.html)

- Complementary: [Specify hyperparameter manually](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/008_specify_params.html)
    - Passing those sets of hyperparameters and let Optuna evaluate them - ```enqueue_trial()```
    - Adding the results of those sets as completed Trials - ```add_trial()```
