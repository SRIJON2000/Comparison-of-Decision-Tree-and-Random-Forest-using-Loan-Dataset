https://github.com/sonarsushant/Loan-Defaulter-Prediction

This problem is about loan defaulters prediction. For this, I have used Lending Club Loan Dataset avaiable on Kaggle. Lending Club is a peer-to-peer lending platform.

The dataset contains loan data for all loans issued through the 2007-2015, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. Dataset is highly skewed with around 900K rows and 74 columns. Using necessary data, I have developed a machine learning model to predict loan defaulters.

This ML model is able to predict 61% of the defaulters. It has also improved Return on Investment (ROI) of loans.

I have divided this project into two parts.

Exploratory Data Analysis and Data Cleaning:

Included in 'Data_Cleaning.ipynb'
Done following cleaning operations:
Selecting relevant features
Null value imputation
Creating dummy variables
Handling outliers
Multicolinearity Check
Training a machine learning model:

Included in 'Model_Training_and_Evaluation.ipynb'
Models used:
Logistic Regression
Random Forest
LightGBM
lending_club.zip: contains custom classes and functions