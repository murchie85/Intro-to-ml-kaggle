# NOTE THIS TIME USING ALL NUMERICAL VALUES, ACCURACY SHOULD INCREASE
#------------------------------------------------------------------------------------------
# PROGRAM DESCRIPTION 
# 
# THIS CODE IS FOR DEALING WITH MISSING VALUES 
# OPTION 1: JUST DROP THE COLUMNS
# OPTION 2: IMPUTE THE MISSING VALUES
#------------------------------------------------------------------------------------------



# Code you have previously used to load data
import pandas as pd
# FIT MODEL
from sklearn.ensemble import RandomForestRegressor
# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error





#------------------------------------------------------------------------------------------
# PREP: REMOVE ROWS MISSING TARGET, USE ONLY NUMBERS & SPLIT
#------------------------------------------------------------------------------------------


# Read the data
X_full = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# Remove rows with MISSING TARGET, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X =           X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])




# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

print('printing head')
print(X_train.head())
print('Remember all catagorical fields were ignored')
#------------------------------------------------------------------------------------------
# DEFINE OPTIMISER SCORING FUNCTION
#------------------------------------------------------------------------------------------



# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)




#------------------------------------------------------------------------------------------
# OPTION 1: DROP ALL COLUMNS WITH MISSING VALUES
#------------------------------------------------------------------------------------------


# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
#reduced_X_test = X_test.drop(cols_with_missing, axis=1) #uncomment if you want to chose drop

print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))


#------------------------------------------------------------------------------------------
# OPTION 2 IMPUTATION METHOD: FOR BOTH TRAIN & VALIDATION SETS
#------------------------------------------------------------------------------------------


from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer(strategy='median')
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns


print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

#------------------------------------------------------------------------------------------
# DEFINE FINAL MODEL AND REPORT
#------------------------------------------------------------------------------------------

final_imputer = my_imputer
final_X_train = imputed_X_train
final_X_valid = imputed_X_valid

# Define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your chosen approach):")
print(mean_absolute_error(y_valid, preds_valid))



#------------------------------------------------------------------------------------------
# TEST DATA PREP
#------------------------------------------------------------------------------------------

# MAKE SURE 
# the preprocessed test DataFrame has no missing values, and
# `final_X_test` has the same number of rows as `X_test`.

# PROCESS TEST INPUT DATA
final_X_test = pd.DataFrame(final_imputer.transform(X_test))

# Get test predictions
preds_test = model.predict(final_X_test)


print('PRINTING PREDICTIONS')
print(preds_test)
#------------------------------------------------------------------------------------------
# SAVE
#------------------------------------------------------------------------------------------

print('saving to submission file ')

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

