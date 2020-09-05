# Machine Learning

In this assignment, I built and evaluated several machine-learning models to predict credit risk using free data from LendingClub. Credit risk is an inherently imbalanced classification problem (the number of good loans is much larger than the number of at-risk loans), so I employed different techniques for training and evaluating models with imbalanced classes. I used the imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques:

- Resampling
- Ensemble Learning 

## Resampling: 
I used the imbalanced learn library to resample the LendingClub data and build and evaluate logistic regression classifiers using the resampled data.

Steps:

Loaded the Lending Club data, split the data into training and testing sets, and scaled the features data.
Oversampled the data using the Naive Random Oversampler and SMOTE algorithms.
Undersampled the data using the Cluster Centroids algorithm.
Over- and under-sampled using a combination SMOTEENN algorithm.

For each of the above, I did:

Trained a logistic regression classifier from sklearn.linear_model using the resampled data.
Calculated the balanced accuracy score from sklearn.metrics.
Calculated the confusion matrix from sklearn.metrics.
Printed the imbalanced classification report from imblearn.metrics.

### Findings:

Which model had the best balanced accuracy score? - Naive Random Oversampling 
Which model had the best recall score? - Cluster Centroids
Which model had the best geometric mean score? - SMOTE


## Ensemble Learning:
In this section, I trained and compared two different ensemble classifiers to predict loan risk and evaluate each model. I used the Balanced Random Forest Classifier and the Easy Ensemble Classifier. Refer to the documentation for each of these to read about the models and see examples of the code.

For each of the above, I did:

Loaded the Lending Club data, split the data into training and testing sets, and scaled the features data.
Trained the model using the quarterly data from LendingClub provided in the Resource folder.
Calculated the balanced accuracy score from sklearn.metrics.
Printed the confusion matrix from sklearn.metrics.
Generated a classification report using the imbalanced_classification_report from imbalanced learn.
For the balanced random forest classifier only, printed the feature importance sorted in descending order (most important feature to least important) along with the feature score.

### Findings:

Which model had the best balanced accuracy score? - Easy Ensemble Classifier
Which model had the best recall score? - Easy Ensemble Classifier
Which model had the best geometric mean score? - Easy Ensemble Classifier
What are the top three features? - 0.07876809003486353, 'total_rec_prncp'), (0.05883806887524815, 'total_pymnt'), (0.05625613759225244, 'total_pymnt_inv')