# Credit-card-fraud-detection-Model

# Project Report:
Introduction
Credit card fraud is a major concern for both consumers and financial institutions. Fraudulent transactions can lead to financial losses and damage to the reputation of financial institutions. Machine learning techniques have been used extensively to detect fraudulent transactions. In this project, we use logistic regression to classify transactions as either legitimate or fraudulent based on their features

# Data
The data used in this project is a CSV file containing credit card transaction data. The data has 31 columns and 5 rows. The "Class" column is the target variable, which indicates whether the transaction is legitimate (Class = 0) or fraudulent (Class = 1).

# Preprocessing
Before training the model, we first separate the legitimate and fraudulent transactions. Since the data is imbalanced, with significantly more legitimate transactions than fraudulent transactions, we undersample the legitimate transactions to balance the classes. We then split the data into training and testing sets using the train_test_split () function.

# Model
We use RandomForestClassifier to classify transactions as either legitimate or fraudulent based on their features. RandomForestClassifier is a widely used classification algorithm that models the probability of an event occurring based on input features. The  RandomForestClassifier model is trained on the training data using the RandomForestClassifier() function from scikit-learn. The trained model is then used to predict the target variable for the testing data.

# Evaluation
The performance of the model is evaluated using the accuracy metric, which is the fraction of correctly classified transactions. The accuracy on the training and testing data is calculated using the accuracy_score() function from scikit-learn.

# Streamlit Application
We use Streamlit to create a user interface for the credit card fraud detection project. The Streamlit application allows the user to upload a CSV file containing credit card transaction data, and the uploaded data is used to train the logistic regression model. The user can also input transaction features and get a prediction on whether the transaction is legitimate or fraudulent.

# Conclusion
In this project, we used logistic regression to detect fraudulent credit card transactions. We achieved a high accuracy on both the training and testing data, indicating that the model is effective at detecting fraudulent transactions. The Streamlit application provides an easy-to-use interface for detecting fraudulent transactions in real-time.
