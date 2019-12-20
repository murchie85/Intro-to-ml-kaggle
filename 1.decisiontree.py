# Code you have previously used to load data
import pandas as pd
# FIT MODEL
from sklearn.tree import DecisionTreeRegressor

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

model = DecisionTreeRegressor(random_state=1)
model.fit(X,y)

# PREDICT

prediction = model.predict(X)

print('The predicted values are :')
print(prediction[0:5])
print(' ')
print('The actual values are: ')
print(y.head())