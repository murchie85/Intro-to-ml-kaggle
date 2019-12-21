# Code you have previously used to load data
import pandas as pd
# FIT MODEL
from sklearn.tree import DecisionTreeRegressor
# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Path of the file to read
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
# print the list of columns (to find the target we want to predict) 
home_data.columns

# set target output
y = home_data.SalePrice
# SLICE THE HOME DATA INTO TARGETTED to train COLUMNS ONLY
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr']
# select data corresponding to features in feature_names
X = home_data[feature_names]
#  Split is random proportion
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# SPECIFY MODEL this time with train_X instead of just X
model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)
# USE VAL DATA
val_predictions = model.predict(val_X)
# ACCURACY
val_mae = mean_absolute_error(val_predictions, val_y)



#------------------------------------------------------------------------------------------
# PRINT
#------------------------------------------------------------------------------------------


print('The predicted values on validation set are :')
print(val_predictions[0:5])
print(' ')
# NOTE YOU HAVE TO USE VALIDATION NUMBERS to be consistent
print('The actual values are: ')
print(val_y[:5])





print('The mean absolute error on validation set, is: ')
print("Validation MAE: {:.0f}".format(val_mae))
