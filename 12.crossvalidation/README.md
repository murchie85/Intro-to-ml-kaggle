# CROSS VALIDATION

- [FULL-CODE](#FULL-CODE)

**CHALLENGE** Iteration of models is the core work done in data science, the challenge is when splitting train and validation, more validation means less to train. Also a certain validation set may be more effective than if you chose another portion.


**CROSS VALIDATION** is when modelling process is run on different subsets of the data.   
This gives multiple measures of model quality.

For example, we could begin by dividing the data into 5 pieces, each 20% of the full dataset. In this case, we say that we have broken the data into 5 "folds".  

![example](https://i.imgur.com/9k60cVA.png)
*thanks to kaggle for image and all kt resources*


Then, we run one experiment for each fold:

- In **Experiment 1**, we use the first fold as a validation (or holdout) set and everything else as training data. This gives us a measure of model quality based on a 20% holdout set.  

- In **Experiment 2**, we hold out data from the second fold (and use everything except the second fold for training the model). The holdout set is then used to get a second estimate of model quality.

- We repeat this process, using every fold once as the holdout set. Putting this together, 100% of the data is used as holdout at some point, and we end up with a measure of model quality that is based on all of the rows in the dataset (even if we don't use all rows simultaneously).

# WHEN TO USE

For small datasets, where extra computational burden isn't a big deal, you should run cross-validation.
I.E If it takes a few minutes or less to run.

# EXAMPLE
##Import

```python
import pandas as pd

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price
```

## Define pipeline

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])
```


## Score with scikit-learn cross_val_score

```python
rom sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
```
### output
```python
MAE scores:
 [301628.7893587  303164.4782723  287298.331666   236061.84754543
 260383.45111427]
 ```

 We chose negative mean absolute error as our scoring 
 scikit learn likes to make : higher the better for scoring.  
 But its rare elsewhere.  
 so we multiply by - to keep it familiar with what we do.   

## GET AVERAGE 

```python
print("Average MAE score (across experiments):")
print(scores.mean())
```
## OUTPUT
```
277707.3795913405
```



If you'd like to learn more about [hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization), you're encouraged to start with **grid search**, which is a straightforward method for determining the best _combination_ of parameters for a machine learning model.  Thankfully, scikit-learn also contains a built-in function [`GridSearchCV()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) that can make your grid search code very efficient!

# FULL CODE 

```python
#------------------------------------------------------------------------------------------
# PROGRAM DESCRIPTION 
#
# THIS PROGRAM DEALS PIPELINES
# 
#------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score


#------------------------------------------------------------------------------------------
# PREPROCESS
#------------------------------------------------------------------------------------------

# Read the data
train_data = pd.read_csv('train.csv', index_col='Id')
test_data = pd.read_csv('test.csv', index_col='Id')


# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)


# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()


print(X.head())

#------------------------------------------------------------------------------------------
# SCORING FUNCTION (PIPELINE/SCORE INCLUDED)
#------------------------------------------------------------------------------------------

def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()

#------------------------------------------------------------------------------------------
# TRY VARIOUS MULTIPLES OF 50
#------------------------------------------------------------------------------------------


results = {}
for i in range(1,9):
    results[50*i] = get_score(50*i)


#------------------------------------------------------------------------------------------
# PLOT VALUES
#------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(results.keys(), results.values())
plt.show()
```

