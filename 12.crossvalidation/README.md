# CROSS VALIDATION

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