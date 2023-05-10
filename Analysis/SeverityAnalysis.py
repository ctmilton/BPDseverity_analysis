import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

s_df = pd.read_csv(r'/Users/carol/Desktop/severity.csv')


# Performing Random Forest Classifier!

# Creating new input matrix X for predictor values
X = s_df.iloc[:, s_df.columns.get_loc('abandonment'):s_df.columns.get_loc(
    'dissociation_and_paranoid_ideation')+1].values
y = s_df.loc[:, 'BPDSIsumCat']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100)

# Fitting the random forest model to the training data
rf.fit(X_train, y_train)

# Evaluating the model performance
accuracy = rf.score(X_test, y_test)
print("\n\nAccuracy of Random Forest Classifier:", accuracy)
print("This is a pretty high accuracy!")

#-------------------------------------------------------------------------------------------------
# Creating Random Forest Regression Models for each predictor

# Model for 'abandonment' variable
print("\n\nCreating Random Forest Regression Model for Abandonment Variable:")
# Creating new input matrix X for predictor values
X = s_df.iloc[:, s_df.columns.get_loc('interpersonal_relationships'):s_df.columns.get_loc(
    'dissociation_and_paranoid_ideation')+1].values
y = s_df.loc[:, 'abandonment']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a random forest regression model with 100 trees
rf_regressor = RandomForestRegressor(n_estimators=100)

# Fitting the model on the training data
rf_regressor.fit(X_train, y_train)

# Predicting on the test data
y_predict = rf_regressor.predict(X_test)

# Evaluating the mean squared error of the model
mse = mean_squared_error(y_test, y_predict)
print("Mean Squared Error:", mse)

#-------------------------------------------------------------------------------------------------
# Model for 'interpersonal_relationships' variable
print("\n\nCreating Random Forest Regression Model for Interpersonal Relationships Variable:")
# Creating new input matrix X for predictor values
X = s_df[['abandonment','identity','impulsivity','parasuicidal_behavior',
                       'affective_instability','emptiness','outbursts_of_anger',
                       'dissociation_and_paranoid_ideation']].values
y = s_df.loc[:, 'interpersonal_relationships']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a random forest regression model with 100 trees
rf_regressor = RandomForestRegressor(n_estimators=100)

# Fitting the model on the training data
rf_regressor.fit(X_train, y_train)

# Predicting on the test data
y_predict = rf_regressor.predict(X_test)

# Evaluating the mean squared error of the model
mse = mean_squared_error(y_test, y_predict)
print("Mean Squared Error:", mse)

#-------------------------------------------------------------------------------------------------
# Model for 'identity' variable
print("\n\nCreating Random Forest Regression Model for Identity Variable:")
# Creating new input matrix X for predictor values
X = s_df[['abandonment','interpersonal_relationships','impulsivity','parasuicidal_behavior',
                       'affective_instability','emptiness','outbursts_of_anger',
                       'dissociation_and_paranoid_ideation']].values
y = s_df.loc[:, 'identity']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a random forest regression model with 100 trees
rf_regressor = RandomForestRegressor(n_estimators=100)

# Fitting the model on the training data
rf_regressor.fit(X_train, y_train)

# Predicting on the test data
y_predict = rf_regressor.predict(X_test)

# Evaluating the mean squared error of the model
mse = mean_squared_error(y_test, y_predict)
print("Mean Squared Error:", mse)

#-------------------------------------------------------------------------------------------------
# Model for 'impulsivity' variable
print("\n\nCreating Random Forest Regression Model for Impulsivity Variable:")
# Creating new input matrix X for predictor values
X = s_df[['abandonment','interpersonal_relationships','identity','parasuicidal_behavior',
                       'affective_instability','emptiness','outbursts_of_anger',
                       'dissociation_and_paranoid_ideation']].values
y = s_df.loc[:, 'impulsivity']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a random forest regression model with 100 trees
rf_regressor = RandomForestRegressor(n_estimators=100)

# Fitting the model on the training data
rf_regressor.fit(X_train, y_train)

# Predicting on the test data
y_predict = rf_regressor.predict(X_test)

# Evaluating the mean squared error of the model
mse = mean_squared_error(y_test, y_predict)
print("Mean Squared Error:", mse)

#-------------------------------------------------------------------------------------------------
# Model for 'parasuicidal_behavior' variable
print("\n\nCreating Random Forest Regression Model for Parasuicidal Behavior Variable:")
# Creating new input matrix X for predictor values
X = s_df[['abandonment','interpersonal_relationships','identity','impulsivity',
                       'affective_instability','emptiness','outbursts_of_anger',
                       'dissociation_and_paranoid_ideation']].values
y = s_df.loc[:, 'parasuicidal_behavior']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a random forest regression model with 100 trees
rf_regressor = RandomForestRegressor(n_estimators=100)

# Fitting the model on the training data
rf_regressor.fit(X_train, y_train)

# Predicting on the test data
y_predict = rf_regressor.predict(X_test)

# Evaluating the mean squared error of the model
mse = mean_squared_error(y_test, y_predict)
print("Mean Squared Error:", mse)

#-------------------------------------------------------------------------------------------------
# Model for 'affective_instability' variable
print("\n\nCreating Random Forest Regression Model for Affective Instability Variable:")
# Creating new input matrix X for predictor values
X = s_df[['abandonment','interpersonal_relationships','identity','impulsivity',
                       'parasuicidal_behavior','emptiness','outbursts_of_anger',
                       'dissociation_and_paranoid_ideation']].values
y = s_df.loc[:, 'affective_instability']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a random forest regression model with 100 trees
rf_regressor = RandomForestRegressor(n_estimators=100)

# Fitting the model on the training data
rf_regressor.fit(X_train, y_train)

# Predicting on the test data
y_predict = rf_regressor.predict(X_test)

# Evaluating the mean squared error of the model
mse = mean_squared_error(y_test, y_predict)
print("Mean Squared Error:", mse)

#-------------------------------------------------------------------------------------------------
# Model for 'emptiness' variable
print("\n\nCreating Random Forest Regression Model for Emptiness Variable:")
# Creating new input matrix X for predictor values
X = s_df[['abandonment','interpersonal_relationships','identity','impulsivity',
                       'parasuicidal_behavior','affective_instability','outbursts_of_anger',
                       'dissociation_and_paranoid_ideation']].values
y = s_df.loc[:, 'emptiness']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a random forest regression model with 100 trees
rf_regressor = RandomForestRegressor(n_estimators=100)

# Fitting the model on the training data
rf_regressor.fit(X_train, y_train)

# Predicting on the test data
y_predict = rf_regressor.predict(X_test)

# Evaluating the mean squared error of the model
mse = mean_squared_error(y_test, y_predict)
print("Mean Squared Error:", mse)

#-------------------------------------------------------------------------------------------------
# Model for 'outbursts_of_anger' variable
print("\n\nCreating Random Forest Regression Model for Outbursts of Anger Variable:")
# Creating new input matrix X for predictor values
X = s_df[['abandonment','interpersonal_relationships','identity','impulsivity',
                       'parasuicidal_behavior','affective_instability','emptiness',
                       'dissociation_and_paranoid_ideation']].values
y = s_df.loc[:, 'outbursts_of_anger']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a random forest regression model with 100 trees
rf_regressor = RandomForestRegressor(n_estimators=100)

# Fitting the model on the training data
rf_regressor.fit(X_train, y_train)

# Predicting on the test data
y_predict = rf_regressor.predict(X_test)

# Evaluating the mean squared error of the model
mse = mean_squared_error(y_test, y_predict)
print("Mean Squared Error:", mse)

#-------------------------------------------------------------------------------------------------
# Model for 'dissociation_and_paranoid_ideation' variable
print("\n\nCreating Random Forest Regression Model for Dissociation and Paranoid Ideation Variable:")
# Creating new input matrix X for predictor values
X = s_df[['abandonment','interpersonal_relationships','identity','impulsivity',
                       'parasuicidal_behavior','affective_instability','emptiness',
                       'outbursts_of_anger']].values
y = s_df.loc[:, 'dissociation_and_paranoid_ideation']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a random forest regression model with 100 trees
rf_regressor = RandomForestRegressor(n_estimators=100)

# Fitting the model on the training data
rf_regressor.fit(X_train, y_train)

# Predicting on the test data
y_predict = rf_regressor.predict(X_test)

# Evaluating the mean squared error of the model
mse = mean_squared_error(y_test, y_predict)
print("Mean Squared Error:", mse)


target_name = s_df.BPDSIsumCat.values

# Creating figure of the first decisions tree in random forest classifier
plt.close()
fig = plt.figure(figsize=(15, 10))
plot_tree(rf.estimators_[0],
          feature_names=X_train.dtype.names,
          class_names=target_name,
          filled=True, impurity=True,
          rounded=True)
plt.show()
