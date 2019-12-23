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
# SLICE THE HOME DATA INTO TARGETTED COLUMNS ONLY
features = ['LotArea','OverallQual','OverallCond', 'TotRmsAbvGrd', 'GrLivArea', 'YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr']

X = home_data[features] # select data corresponding to features in features
y = home_data.SalePrice # set target output

# Split into validation and training data
#train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

print('printing preview')
print(train_X.head())
print(train_X.shape)
#------------------------------------------------------------------------------------------
# OPTIMISER
#------------------------------------------------------------------------------------------



# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
model_6 = RandomForestRegressor(n_estimators=100, random_state=1)

models = [model_1, model_2, model_3, model_4, model_5, model_6]



# Function for comparing different models
def score_model(model, X_t=train_X, X_v=val_X, y_t=train_y, y_v=val_y):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

mae = 9999999 # SET MAE HIGH 
mae_index = 0
for i in range(0, len(models)):
    mae_current = score_model(models[i])
    if mae_current < mae:
    	mae = mae_current
    	mae_index = i
    print("Model %d MAE: %d" % (i+1, mae_current))


print('Min MAE value is : ' + str(mae) + ' and the model index is ' + str(mae_index))
# models[mae_index] is now the ideal value



#------------------------------------------------------------------------------------------
# ALL DATA (used for test submission)
#------------------------------------------------------------------------------------------

# Accuracy should improve if used on all data
rf_model_on_full_data = models[mae_index]
rf_model_on_full_data.fit(X,y) # ALL Data from training CSV

full_train_prediction = rf_model_on_full_data.predict(X)
full_train_mae = mean_absolute_error(full_train_prediction, y)
print('')
print("Full Set MAE: {:,.0f}".format(full_train_mae))





#------------------------------------------------------------------------------------------
# TRAIN DATA FOR VALIDATION SET (a good benchmark to see if it will be better)
#------------------------------------------------------------------------------------------

# RANDOM FOREST
model = models[mae_index]
model.fit(train_X, train_y)
rf_val_predictions = model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(rf_val_mae))




#------------------------------------------------------------------------------------------
# ALL DATA FOR TEST COMPETITION CSV
#------------------------------------------------------------------------------------------
test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]

# Model is already build, we are just applying it to test data now
test_prediction = rf_model_on_full_data.predict(test_X)
print('Test csv does not have target sales price to compare MAE or accuracy')
print('Printing test predicted values')
print(test_prediction)




#------------------------------------------------------------------------------------------
# SAVE
#------------------------------------------------------------------------------------------

print('saving to submission file ')

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_prediction})
output.to_csv('submission.csv', index=False)


