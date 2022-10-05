Optuna
===

## Comparison between ```Hyperopt``` and ```Optuna```
- Hyperopt should **predefine the search space** using **dictionary** to setup
- Optuna can use any python syntax to set up search space (e.g. ```.yml```, ```dict```)
<img src=https://user-images.githubusercontent.com/54303314/193978439-d7a86648-6c28-4e26-904b-2e207fc1162a.png width="800" height="450">


Samplers & pruning


## Samplers
- [Overview](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#sphx-glr-tutorial-10-key-features-003-efficient-optimization-algorithms-py)
- Model-based
- Other method
### Model-based 
1. TPE: kernel fitting (tree-based estimator which can deal with both numerical and categorical features)
2. GP: Gaussian process (suitable for numerical features)
3. CMA-ES (covariant matrix adaptation-evolution strategy): meta-heuristics algorithm for continuous space

### Other method
1. Random Search
2. Grid Search
3. User-defined algorithm

<img src=https://user-images.githubusercontent.com/54303314/193965369-15ef0332-a00b-41df-984c-7769f49f3a77.png width="500" height="270">


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

## Which Sampler and Pruner should be used?
- For not deep learning tasks,
    - For ```RandomSampler```, ```MedianPruner``` is the best
    - For ```TPESampler```,  ```Hyberband``` is the best
    - [Ref](https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako)
- For deep learning tasks,
<img src=https://user-images.githubusercontent.com/54303314/193975227-5e4fc778-7c9b-4159-858d-a0a4a42919b3.png width="900" height="250">


reference: [ref](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
