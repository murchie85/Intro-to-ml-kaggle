#------------------------------------------------------------------------------------------
# PROGRAM DESCRIPTION 
#
# THIS PROGRAM DEALS PIPELINES
# 
#------------------------------------------------------------------------------------------



# Code you have previously used to load data
import pandas as pd
# FIT MODEL
from sklearn.ensemble import RandomForestRegressor
# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder




#------------------------------------------------------------------------------------------
# PREP
# 1. Import
# 2. Drop missing rows
# 3. set y
# 4. Drop target column from train
# 5. Split data
# 6. Ascertain catagorical column names & < 10 unique
# 7. Ascertain Numerical columns
# 8. Combine into feature columns
# 9. Set Xtrain,valid,test using features columns.
#------------------------------------------------------------------------------------------


# Read the data
X_full = pd.read_csv('../train.csv', index_col='Id')
X_test_full = pd.read_csv('../test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True) # DROP MISSING
y = X_full.SalePrice # SET Y
X_full.drop(['SalePrice'], axis=1, inplace=True) # DROP COLUMN (axis=1)



# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)


# Assign to cname if match (text)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols


X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()   # so we don't have to chop later


print('printing head')
print(X_train.head())

#------------------------------------------------------------------------------------------
# SET UP PIPELINE
# 1. Define NUMERICAL transformer/preprocessor
# 2. Define CATAGORICAL transformer/preprocessor
# 3. Bundle preprocessors
# 4. Define model
# 5. Bundle preprocessor & model
# 6. Fit & predict
#------------------------------------------------------------------------------------------


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')


# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# COMBINE Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)





# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))



#------------------------------------------------------------------------------------------
# TEST predict
#------------------------------------------------------------------------------------------

print('Predicting on test data set')
preds_test = clf.predict(X_test)

#------------------------------------------------------------------------------------------
# SAVE
#------------------------------------------------------------------------------------------
print('Writing submission file...')
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

print('complete')
