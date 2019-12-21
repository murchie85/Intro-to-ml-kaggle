#------------------------------------------------------------------------------------------
# PROGRAM DESCRIPTION 
# The first part prepares the data and splits as normal
# The second part creates a function that can be called to define, fit, predict and give error
# The third iterates this with all leaf values, a lambda is easier but harder to understand
# 
#------------------------------------------------------------------------------------------



# Code you have previously used to load data
import pandas as pd
# FIT MODEL
from sklearn.tree import DecisionTreeRegressor
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
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr']
# select data corresponding to features in feature_names
X = home_data[feature_names]
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#------------------------------------------------------------------------------------------
# OPTIMISE FUNCTION
#------------------------------------------------------------------------------------------

# THIS DEFINES, FITS, PREDICTS AND ERRORS as normal in one bundle 
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

#------------------------------------------------------------------------------------------
# ITERATE OPTIMISE FUNCTION
#------------------------------------------------------------------------------------------


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
#scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
#best_tree_size = min(scores, key=scores.get)


leaf_dict = {}
for leaf_size in candidate_max_leaf_nodes:
    current_min = get_mae(leaf_size, train_X, val_X, train_y, val_y)
    leaf_dict[leaf_size] = current_min

best_tree_size = min(leaf_dict, key=leaf_dict.get) 


#------------------------------------------------------------------------------------------
# BUILD FINAL MODEL
#------------------------------------------------------------------------------------------


# final model on all data, not just train
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)

prediction = final_model.predict(val_X)
# ACCURACY
val_mae = mean_absolute_error(val_y, prediction)


#------------------------------------------------------------------------------------------
# PRINT 
#------------------------------------------------------------------------------------------

print('printing full leaf dict: ')
print(leaf_dict)
print('')
print('The best size is: ' + str(best_tree_size))

print('The predicted values on validation set are : ' )
print(str(prediction[0:5]))
print(' ')

print('The actual values are: ' + str(val_y[:5]))

print('The mean absolute error on validation set, is: ')
print("Validation MAE: {:.0f}".format(val_mae))
print('This is different than tree size value, as that was applied only to train data')

