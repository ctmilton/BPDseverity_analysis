import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import make_scorer, brier_score_loss

plt.style.use('ggplot')
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Replace the file path
f_df = pd.read_csv(r'/Users/carol/Desktop/factors.csv')

# #print(f_df)
#
# print("Descriptive Statistics:")
# result = f_df.iloc[:, f_df.columns.get_loc('paranoiaM'):f_df.columns.get_loc('staitM') + 1].describe()
# print(result)
#
# # Condition Checking
# fig, ax = plt.subplots()
# ax.boxplot((f_df.paranoiaM, f_df.bdiM, f_df.staisM, f_df.staitM),
#            vert=False, showmeans=True, meanline=True,
#            labels=('Paranoia', 'BDI', 'Stais', 'Stait'),
#            patch_artist=True,
#            medianprops={'linewidth': 2, 'color': 'purple'},
#            meanprops={'linewidth': 2, 'color': 'red'})
# plt.title('Checking Distribution of Data')
# plt.show()
#
# print("\n\nThere are potential outliers that may affect our models later. However, they are not too far away so I will "
#       "keep them.")
#
# # Checking distributions for each factor
# plt.clf()
# fig, ax = plt.subplots()
# ax.hist(f_df.paranoiaM, 5, cumulative=False)
# ax.set_xlabel('Paranoia')
# ax.set_ylabel('Frequency')
# plt.show()
#
# plt.clf()
# fig, ax = plt.subplots()
# ax.hist(f_df.bdiM, 8, cumulative=False)
# ax.set_xlabel('BDI')
# ax.set_ylabel('Frequency')
# plt.show()
#
# plt.clf()
# fig, ax = plt.subplots()
# ax.hist(f_df.staisM, 8, cumulative=False)
# ax.set_xlabel('Stais')
# ax.set_ylabel('Frequency')
# plt.show()
#
# plt.clf()
# fig, ax = plt.subplots()
# ax.hist(f_df.staitM, 8, cumulative=False)
# ax.set_xlabel('Stait')
# ax.set_ylabel('Frequency')
# plt.show()
#
# print("\n\nNone of the distributions are normal.")
#
# # Correlations Among Factors
#
# print("Since the distributions are non-normal, we use the non-parametric test, Spearman's Rank Correlation Test.")
# corr = f_df.iloc[:,f_df.columns.get_loc('paranoiaM'):f_df.columns.get_loc('staitM')+1].corr(method='spearman')
# print("Spearman's Correlation Matrix")
# print(corr)
#
# print("As seen in the correlation matrix, all factors are very strongly correlated with each other. Therefore, "
#       "we will create logistic models for each individual factor later.")
#
# # Checking variable types
# print(f_df.dtypes)
#
# # Checking Linearity of Log Odds
# plt.clf()
# paranoia_check = sns.regplot(x='paranoiaM', y='groupn', data=f_df, logistic=True)
# paranoia_check.set_title("Paranoia Log Odds Linear Plot")
# plt.show()
# #paranoia_check.figure.savefig(r'/Users/carol/Desktop/LogOddsPlots/paranoia_log_lin.png')
# # Possible Perfect Separation Warning most probably occurring because of the small sample size (21)
# # This is similar in most other factor variables
#
# plt.clf()
# bdi_check = sns.regplot(x='bdiM', y='groupn', data=f_df, logistic=True)
# bdi_check.set_title("BDI Log Odds Linear Plot")
# plt.show()
# #bdi_check.figure.savefig("./LogOddsPlots/bdi_log_lin.png")
# # Perfect Separation Warning is likely because the outcome is perfectly consistent above and below a bdiM value
# # low bdiM values all have groupn of 0
# # higher bdiM values all have groupn of 1
#
#
# plt.clf()
# stais_check = sns.regplot(x='staisM', y='groupn', data=f_df,
#               logistic=True).set_title("State Anxiety Log Odds Linear Plot")
# plt.show()
# #stais_check.figure.savefig("./LogOddsPlots/stais_log_lin.png")
# # Possible Perfect Separation Warning most probably occurring because of the small sample size (21)
#
# plt.clf()
# stait_check = sns.regplot(x='staitM', y='groupn', data=f_df,
#               logistic=True).set_title("Trait Anxiety Log Odds Linear Plot")
# plt.show()
# #stait_check.figure.savefig("./LogOddsPlots/stait_log_lin.png")
# # Possible Perfect Separation Warning most probably occurring because of the small sample size (21)
#
# print("Overall, factors pass linearity of odds condition!")
#
#
# # Logistic Regressions!!
# print("We control for the confounding variable `gender`.")
#
# print("\nLogit Model for Paranoia")
# p_model= smf.logit(formula="groupn~paranoiaM+gendern", data=f_df).fit()
# print(p_model.summary())
# print("\nThe p-value of 0.039 indicates that paranoia is a significant predictor of BPD diagnosis.")
# print("However, it appears that gender does not affect BPD diagnosis in this model.")
# # Since the p-value for `paranoia` is less than alpha=0.05, we see that `paranoia` is a
# # significant predictor for BPD diagnosis. The p-value for `gender` is much greater than 0.05,
# # so it seems that gender does not have a significant influence on the diagnosis of BPD when
# # in conjunction with `paranoia`.
#
# print("\n\nLogit Model for Depression")
# b_model= smf.logit(formula="groupn~bdiM+gendern", data=f_df).fit()
# print(b_model.summary())
#
# print("\n\nLogit Model for State Anxiety")
# s_model= smf.logit(formula="groupn~staisM+gendern", data=f_df).fit()
# print(s_model.summary())
#
# print("\n\nLogit Model for Trait Anxiety")
# t_model= smf.logit(formula="groupn~staitM+gendern", data=f_df).fit()
# print(t_model.summary())
#
# # P-values for all other factors are not significant
#
# print("\n\nAccording to p-values, paranoia is the only significant predictor for BPD diagnosis.")

# Trying Dimensionality Reduction
print("\n\nLet's try using dimensionality reduction to create logit model with all factors.")

# Creating predictor table
X = f_df.iloc[:, f_df.columns.get_loc('paranoiaM'):f_df.columns.get_loc('staitM') + 1]
y = f_df.loc[:, 'groupn']

# Standardizing the predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Conducting PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Determining the number of components
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# # Plotting scree plot to visualize explained variance
# plt.clf()
# plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Scree Plot')
# plt.show()

# Using elbow rule to determine number of components
print("\nUsing elbow rule on Scree plot shows that the number of components should be 3.")
n_components = 3
# Retaining the selected number of components
X_pca_selected = X_pca[:, :n_components]

# Initializing logistic regression model
logit = LogisticRegression()

# Setting up k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Conducting k-fold cross-validation using accuracy as scoring metric
# Good for evaluating overall correctness of model's predictions
cv_results = cross_val_score(logit, X_pca_selected, y, cv=kfold, scoring='accuracy')

# Printing results
print("Cross-validation results: ", cv_results)
print("Mean Accuracy: ", cv_results.mean())
print("Standard Deviation: ", cv_results.std())

print("\nThe accuracy of 86% is decent.")
print("However, we would want better accuracy since we are diagnosing BPD in people.\n\n")

# Conducting k-fold cross-validation using recall as scoring metric
# Focuses on the proportion of correctly predicted positive instances out of all actual positive instances
# Measures the model's ability to identify positive cases and minimizes false negatives
cv_results_recall = cross_val_score(logit, X_pca_selected, y, cv=kfold, scoring='recall')

# Printing results
print("Cross-validation results: ", cv_results_recall)
print("Mean Recall: ", cv_results_recall.mean())
print("Standard Deviation: ", cv_results_recall.std())

print("\nThe mean when using the recall metric is 90% which is good.\n\n")


# Brier score measures the mean squared difference between the predicted probabilities and the actual binary outcomes

# Defining the Brier score as the scoring metric
brier_scorer = make_scorer(brier_score_loss, greater_is_better=False)

# Performing k-fold cross-validation and calculating the Brier score for each fold
scores = cross_val_score(logit, X_pca_selected, y, cv=kfold, scoring=brier_scorer)

# Printing the Brier scores for each fold
for fold, score in enumerate(scores):
    print(f"Fold {fold+1}: Brier Score = {score:.4f}")

# Calculating the average Brier score across all folds
average_brier_score = scores.mean()

print("Average Brier Score: ", average_brier_score)
print("The Brier score of 0.14, shows that the model's predicted outcomes are fairly close to the real outcomes.")

