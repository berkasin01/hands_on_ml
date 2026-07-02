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

## Chapter 3 - Classification

Explored classification techniques using the MNIST handwritten digit dataset with 70,000 images.

Binary Classification:
Trained a SGDClassifier to detect a single digit (5 vs not-5)
Implemented manual cross-validation using StratifiedKFold and clone to understand how it works under the hood
Compared cross_val_score (returns scores) vs cross_val_predict (returns predictions for each fold)

Evaluation Metrics:
Built confusion matrices to visualise true/false positives and negatives
Explored Precision (how many positive predictions were correct) vs Recall (how many actual positives were found)
Combined Precision and Recall into a single F1 score for balanced evaluation
Used decision_function to access raw model scores and set custom thresholds using precision_recall_curve and np.argmax
Plotted ROC curves to visualise the trade-off between true positive rate and false positive rate at all thresholds
Calculated AUC (Area Under Curve) as a single number summary of model performance

Multiclass Classification:
Learned how binary classifiers extend to multiclass: OvO (One vs One) trains a model for every pair of classes, OvR (One vs Rest) trains one model per class against all others
Some models like SVM default to OvO, others like SGD default to OvR
Performed error analysis using confusion matrices to identify which digits the model confuses most

Multilabel and Multioutput Classification:
Explored multilabel classification where each input can have multiple labels
Built a multioutput classifier using KNN to denoise images, predicting clean pixel values from noisy inputs

Data Augmentation:
Applied transformations to training images to generate additional training data and improve model robustness

## Chapter 4 - Training Models
Linear Regression:
Explored the Normal Equation and SVD (via np.linalg.lstsq) as closed-form solutions to find optimal parameters directly without iteration
Understood computational cost of closed-form solutions with large feature sets

Gradient Descent:
Implemented Batch Gradient Descent using the full training set per step -- slow but stable
Mini-batch Gradient Descent uses random subsets per step, balancing speed and stability
Stochastic Gradient Descent uses one instance per step, fast but noisy -- can escape local minima

Polynomial Regression:
Extended linear models by adding polynomial features using PolynomialFeatures
Higher degree polynomials fit training data better but overfit -- bias-variance tradeoff
Used learning curves to diagnose underfitting vs overfitting

Regularisation:
Ridge Regression adds L2 penalty to shrink coefficients, keeps all features
Lasso Regression adds L1 penalty, drives some coefficients to zero -- automatic feature selection
Elastic Net combines L1 and L2 penalties, balancing both approaches
Early Stopping used as a form of regularisation by halting training when validation error stops improving

Logistic Regression:
Used for binary classification, outputs probability via sigmoid function
Trained using cross-entropy loss (log loss)
Extended to multiclass via Softmax Regression (multinomial logistic regression) using softmax function and cross-entropy

## Chapter 5 - SVM
im gonna fix it all, one of the worst days recently

## Chapter 6 - Decision Trees


## Chapter 7 - Ensemble and Random Forest 


## Chapter 8 - Dimensionality Reduction 


## Chapter 9 - Unsupervirsed Learning Techniques

