# HOUSING PRICE KAGGLE COMPETITION

![HOUSE](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)

A combination of some starter modules, from the kaggle tutorial on learning ML, and my own work I have built on top.   
This repo is also collection of the work I have done and summarisation of the litterature.   
Please use Kaggles official site for learning the content [here](https://www.kaggle.com/learn/intro-to-machine-learning).

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
9. **(optional)** Optimisation Underfitting/overfitting control (build a function to itterate hyperparms and check error)
10. Define and fit model on all data `(X,y)` 
11. Predict accuracy and write to submission file




# Intermediate Machine Learning micro-course!

[Link](https://www.kaggle.com/alexisbcook/introduction)

- tackle data types often found in real-world datasets (missing values, categorical variables),
- design pipelines to improve the quality of your machine learning code,
- use advanced techniques for model validation (cross-validation),
- build state-of-the-art models that are widely used to win Kaggle competitions (XGBoost), and
- avoid common and important data science mistakes (leakage).
