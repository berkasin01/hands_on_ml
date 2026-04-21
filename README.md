# Hands-On Machine Learning
Working through exercises and practice code from Hands-On Machine Learning Book from Aurelien Geron with Scikit Learn, Keras and TensorFlow.
This repo tracks my progress as I work through the book, applying concepts hands-on with real code. Will be explaining each chapter to give insight.

## Chapter 2 - End to End Machine Learning Project
Looking at Regression models using the California Housing dataset. Practiced the full ML pipeline from data loading to model evaluation.
Data Exploration and Preparation

Loaded and explored the dataset using pandas (info, describe, histograms)
Created income categories using pd.cut for stratified sampling
Used StratifiedShuffleSplit to maintain income distribution across train/test sets
Visualised geographic data with scatter plots using population size and median house value
Analysed correlation matrices to find relationships between features
Engineered new features: rooms per household, bedrooms per room, population per household

Handling Missing Data and Encoding:
Used SimpleImputer with median strategy to fill missing values
Explored OrdinalEncoder vs OneHotEncoder for categorical columns (ocean_proximity)
Built a custom transformer (CombinedAtrributesAdder) to add engineered features

Feature Scaling:
Compared MinMaxScaler (normalisation) and StandardScaler (standardisation)

Pipelines:
Built a numerical Pipeline chaining SimpleImputer, custom transformer and StandardScaler
Combined numerical and categorical pipelines using ColumnTransformer

Model Training and Evaluation:
Trained and compared LinearRegression, DecisionTreeRegressor and RandomForestRegressor
Used cross_val_score with 10 fold CV to get reliable RMSE estimates
Applied GridSearchCV and RandomizedSearchCV for hyperparameter tuning
Analysed feature importances from the best model
Evaluated final model on test set with RMSE and 95% confidence interval using scipy stats