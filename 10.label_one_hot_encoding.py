# NOTE THIS TIME USING ALL NUMERICAL VALUES, ACCURACY SHOULD INCREASE
#------------------------------------------------------------------------------------------
# PROGRAM DESCRIPTION 
#
# THIS PROGRAM DEALS WITH CATAGORICAL VALUES I.E. TEXT
# 
# OPTION 1: DROP ALL COLUMNS WITH ['object']
# OPTION 2: LABEL ENCODING
# OPTION 3: ONE HOT ENCODING
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
X = pd.read_csv('train.csv', index_col='Id')
X_test = pd.read_csv('test.csv', index_col='Id')

# Remove rows with MISSING TARGET, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True) # removes sales price from training frame


# FULL DROP OUT BUT KEEPING MOST CATAGORICAL ROWS 
# DROP MISSING VALUES (NO IMPUTATION)
cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)          # DROP TRAIN
X_test.drop(cols_with_missing, axis=1, inplace=True)     # DROP TEST


# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

print('printing preview')
print(X_train.head())





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
# GET ALL CATAGORICAL COLUMSN
#------------------------------------------------------------------------------------------

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]


#------------------------------------------------------------------------------------------
# OPTION 1: DROP ALL CATAGORICAL 
#------------------------------------------------------------------------------------------


drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print('Final trainx shape is :' + str(drop_X_train.shape))
print('Final validx shape is :' + str(drop_X_valid.shape))
print('')
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))


#------------------------------------------------------------------------------------------
# OPTION 2: LABEL ENCODING
#------------------------------------------------------------------------------------------

print('first look at values in condition 2')
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())
print('')
print('we need to drop unmatching columns')



# GET GOOD COLUMN NAMES
good_label_cols = [col for col in object_cols if 
                   set(X_train[col]) == set(X_valid[col])]
        
# GET BAD COLUMN NAMES
bad_label_cols = list(set(object_cols)-set(good_label_cols))

      
print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)



from sklearn.preprocessing import LabelEncoder

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply label encoder
label_encoder = LabelEncoder()

# Transform each of the xtrain/xvalid columns that match the goodlabels column
for col in set(good_label_cols):
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
    
print('Final trainx shape is :' + str(label_X_train.shape))
print('Final validx shape is :' + str(label_X_valid.shape))
print('')
print("MAE from Approach 2 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))



#------------------------------------------------------------------------------------------
#  OPTION 3: ONE HOT ENCODING
#  Get unique value count for catagorical columsn
#  One hot encode all below <10
#  Drop any above >10 
#  Or label encode (best not to do this, incase of disparity)
#------------------------------------------------------------------------------------------

from sklearn.preprocessing import OneHotEncoder

# Investigating cardinality

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])


# ENCODE ONLY IF LESS THAN 10 UNIQUE VALUES I.E. CARDINALITY < 10
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# STORE THE REMAINING COLUMN NAMES TO BE LABEL ENCODED
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)


# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))



# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
# This also saves us the hassle of dropping columns 

num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)



print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))





#------------------------------------------------------------------------------------------
# TEST DATA PREP
#------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------
# SAVE
#------------------------------------------------------------------------------------------
"""
print('saving to submission file ')

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
"""
