#------------------------------------------------------------------------------------------
# PROGRAM DESCRIPTION 
# Random forrest implementation
# Similar to before, but optimiser has been removed
# 
# CODE BELOW SAVED FOR LATER
"""
# DECISION TREE - BEST NODE
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))
"""
#------------------------------------------------------------------------------------------



# Code you have previously used to load data
import pandas as pd
# FIT MODEL
from sklearn.ensemble import RandomForestRegressor
# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error




#------------------------------------------------------------------------------------------
# STANDARD PREP APPROACH
#------------------------------------------------------------------------------------------

# Path of the file to read
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
# print the list of columns (to find the target we want to predict) 
home_data.columns
# set target output
y = home_data.SalePrice
# SLICE THE HOME DATA INTO TARGETTED COLUMNS ONLY
features = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr']
# select data corresponding to features in features
X = home_data[features]
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)




#------------------------------------------------------------------------------------------
# TRAIN ON ALL DATA 
#------------------------------------------------------------------------------------------

# Accuracy should improve if used on all data
rf_model_on_full_data = RandomForestRegressor(n_estimators=100)
rf_model_on_full_data.fit(X,y) # ALL Data from training CSV

full_train_prediction = rf_model_on_full_data.predict(X)
full_train_mae = mean_absolute_error(full_train_prediction, y)
print("ALL DATA MAE: {:,.0f}".format(full_train_mae))



#------------------------------------------------------------------------------------------
# BUILD MODEL FOR TRAIN DATA FILE SPLITTING TRAIN/VALIDATE
#------------------------------------------------------------------------------------------

# RANDOM FOREST
model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(train_X, train_y)
rf_val_predictions = model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("VALIDATION TEST MAE: {:,.0f}".format(rf_val_mae))




#------------------------------------------------------------------------------------------
# PREDICT ON TEST COMPETITION DATA
#------------------------------------------------------------------------------------------
test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]

test_prediction = rf_model_on_full_data.predict(test_X)
print('')
print('Test csv does not have target sales price to compare MAE or accuracy')
print('Printing test predicted values')
print(test_prediction)




#------------------------------------------------------------------------------------------
# SAVE
#------------------------------------------------------------------------------------------
"""
print('saving to submission file ')

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_prediction})
output.to_csv('submission.csv', index=False)
"""

