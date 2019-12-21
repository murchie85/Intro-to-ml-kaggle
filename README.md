# INTRO TO ML & HOUSING PRICE COMPETITION

![HOUSE](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)

Some starter modules, from the kaggle tutorial on learning ML, this repo is just a collection of the work i have done and summarisation of the litterature. Please use their official site for learning the content [here](https://www.kaggle.com/learn/intro-to-machine-learning).

Additionally this repository includes my submission for the competition on kaggle to guess housing prices, link [here](https://www.kaggle.com/c/home-data-for-ml-course)

# General steps

1. Import data 
2. Define y (your target output to measure/predict)
3. Define X (your full training set/table)
4. Split data like this  `train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)`
5. Specify your model i.e. `model = DecisionTreeRegressor(random_state=1)`
6. Fit your model, use training data such as `model.fit(train_X, train_y)`
7. Create predictions on validation set `val_predictions = model.predict(val_X)`
8. Predict accuracy on validation set `val_mae = mean_absolute_error(val_predictions, val_y)`
9. Optimisation Underfitting/overfitting control (build a function to itterate hyperparms and check error)


