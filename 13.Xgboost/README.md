# Gradient Boosting XGBoosting

- One of the best methods for Kaggle competitions
- **Random Forest** is better than decision tree because averages predictions of many decision trees.
- Random Forest is an **ensemble** method,ensemble methods combine the predictions of several models (e.g., several trees, in the case of random forests).
  
  
**Gradient boosting** is a method that goes through cycles to iteratively add models into an ensemble.


- Starts initializing a niave model
- Adds improvement

Then, we start the cycle:

- First, we use the current ensemble to generate predictions for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
- These predictions are used to calculate a loss function (like mean squared error, for instance).
- Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss. (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.)
- Finally, we add the new model to ensemble, and ...
... repeat!

[modeldiagram](https://i.imgur.com/MvCGENh.png)
*All material from kaggle tutorial site linked at head of repo*
    

extreme gradient boosting, which is an implementation of gradient boosting with **several additional features** focused on **performance** and speed. (Scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages.)



```python
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
```
## output
```
Mean Absolute Error: 269553.99606958765
```


## Parameter Tuning  

XGBRegressor class has many tunable parameters

## n_estimators

`n_estimators` specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.

- Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data.
- Too high a value causes overfitting, which causes accurate predictions on training data, but inaccurate predictions on test data (which is what we care about).  

Typical values range from 100-1000, though this depends a lot on the learning_rate parameter discussed below.

```python
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)
```
  
## early_stopping_rounds  

`early_stopping_rounds` 

Early stopping causes the model to stop iterating when the validation score stops improving,

`early_stopping_rounds=5` is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.
Use early stopping to find out optimal value when training on all data.

```python
y_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
```

## learning_rate

Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in.

This means each tree we add to the ensemble helps us less. So, we can set a higher value for n_estimators without overfitting. If we use early stopping, the appropriate number of trees will be determined automatically.

In general, a **small learning rate and large number of estimators** will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets learning_rate=0.1.

```python
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```


## n_jobs

*(only useful for large compute, doesn't affect accuracy)*

On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter n_jobs equal to the number of cores on your machine. On smaller datasets, this won't help.

The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction. But, it's useful in large datasets where you would otherwise spend a long time waiting during the fit command.

```python
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```



