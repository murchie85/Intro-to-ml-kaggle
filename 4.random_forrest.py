#------------------------------------------------------------------------------------------
# PROGRAM DESCRIPTION 
# Random forrest implementation
# Similar to before, but optimiser has been removed
# 
#------------------------------------------------------------------------------------------



# Code you have previously used to load data
import pandas as pd
# FIT MODEL
#from sklearn.tree import DecisionTreeRegressor NOT NEEDED NOW
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
# BUILD  MODEL
#------------------------------------------------------------------------------------------

model = RandomForestRegressor(random_state=1)
model.fit(train_X, train_y)

prediction = model.predict(val_X)
MAE = mean_absolute_error(prediction, val_y)


#------------------------------------------------------------------------------------------
# PRINT 
#------------------------------------------------------------------------------------------

print("Validation MAE for Random Forest Model: {}".format(MAE))

