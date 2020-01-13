#------------------------------------------------------------------------------------------
# PROGRAM DESCRIPTION 
#
# THIS PROGRAM DEALS PIPELINES
# 
#------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#------------------------------------------------------------------------------------------
# PREPROCESS
#------------------------------------------------------------------------------------------

# Read the data
X = pd.read_csv('../train.csv', index_col='Id')
X_test_full = pd.read_csv('../test.csv', index_col='Id')


# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)



#------------------------------------------------------------------------------------------
# DEFINE AND PREDICT 
#------------------------------------------------------------------------------------------



from xgboost import XGBRegressor
  
# Define the models
model_1 = XGBRegressor(n_estimators=500, learning_rate=0.05)
model_2 = XGBRegressor(n_estimators=1500, learning_rate=0.05, early_stopping_rounds = 5)
model_3 = XGBRegressor(n_estimators=100)
model_4 = XGBRegressor(n_estimators=1000, learning_rate=0.08)
model_5 = XGBRegressor(n_estimators=500, learning_rate=0.03)


models = [model_1, model_2, model_3, model_4, model_5]



# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
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
# DEFINE AND PREDICT: nestimators and learning rate
#------------------------------------------------------------------------------------------

print('Chosen model is : ')
print(str(models[mae_index]))

# Define the model
final_model = models[mae_index]

# Fit the model
final_model.fit(X_train, y_train)

# Get predictions
predictions_2 = final_model.predict(X_valid)

# Calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid)
print("Mean Absolute Error:" , mae_2)


#------------------------------------------------------------------------------------------
# TEST SUBMISSIONS
#------------------------------------------------------------------------------------------
tes_pred = final_model.predict(X_test) 

#------------------------------------------------------------------------------------------
# SAVE
#------------------------------------------------------------------------------------------
print('Writing submission file...')
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': tes_pred})
output.to_csv('submission.csv', index=False)

print('complete')

