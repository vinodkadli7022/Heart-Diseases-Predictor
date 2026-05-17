# Heart Disease Classification Using Machine Learning

**Abstract**

This project develops a binary classification model to predict heart disease from clinical patient attributes. The workflow includes data exploration, preprocessing, feature engineering, univariate feature selection, model comparison, hyperparameter tuning, and final evaluation. On the held-out test set, the final tuned Random Forest model achieved 0.8478 accuracy and 0.9143 ROC-AUC, showing strong discriminatory performance on a compact tabular medical dataset.

**Index Terms**

Heart disease prediction, classification, feature engineering, feature selection, Random Forest, ROC-AUC.

## I. Introduction

Heart disease is a high-impact clinical prediction problem because early risk identification can support preventive care and triage. In this assignment, a supervised learning approach was used to predict a binary target variable, `HeartDisease`, from patient-level clinical measurements. The main objective was not only predictive performance, but also a transparent end-to-end workflow with proper preprocessing, model comparison, and evaluation.

## II. Data and Preprocessing

The dataset contains 918 patient records and 12 columns, including age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, maximum heart rate, exercise-induced angina, oldpeak, ST slope, and the target class. Basic exploration included printing shape, dtypes, descriptive statistics, missing values, and duplicate counts. The class distribution was also inspected to understand balance.

Two preprocessing choices were important. First, `Cholesterol = 0` was treated as missing because a zero value is biologically invalid. These values were imputed using the median cholesterol for the same sex group, which is more clinically plausible than a global median. Second, all object-type categorical variables were encoded using `LabelEncoder` so they could be used by downstream statistical tests and models [1], [2].

## III. Feature Engineering and Selection

Three engineered features were created to better capture clinically meaningful relationships:

1. `HRtoAge_Ratio = MaxHR / Age`, representing cardiac response relative to age.
2. `STxAngina = Oldpeak × ExerciseAngina`, capturing the interaction between exercise-induced angina and ST depression.
3. `AgeBin`, an ordinal age group feature based on clinically meaningful age ranges.

Feature relevance was then assessed with `SelectKBest` using `f_classif`, an ANOVA F-test for classification [3]. The top 10 features were retained for model training. This step reduced dimensionality while keeping the strongest predictors from the training data only.

## IV. Model Selection and Justification

Three models were compared using 5-fold stratified cross-validation: Logistic Regression, Random Forest, and Gradient Boosting. Logistic Regression was used as the interpretable linear baseline [4]. Random Forest was selected because it handles nonlinear feature interactions, is robust on tabular data, and provides feature importance scores [5]. Gradient Boosting was also tested because it often performs strongly on structured data by fitting residual structure sequentially [6].

The final model choice was based on cross-validated ROC-AUC and stability, not only on raw accuracy. Random Forest gave the best overall validation performance in this run, so it was selected for tuning. This choice is reasonable for a small-to-medium tabular medical dataset with mixed feature types and nonlinear relationships.

## V. Training and Evaluation

The best model was tuned with `GridSearchCV` using 5-fold cross-validation and `roc_auc` as the scoring metric [7]. The final tuned model was trained on the full training set and evaluated on the held-out test set.

Final test results were:

- Accuracy: 0.8478
- Precision: 0.8627
- Recall: 0.8627
- F1-score: 0.8627
- ROC-AUC: 0.9143

The confusion matrix and classification report showed balanced performance across both classes, with no extreme bias toward a single label. The evaluation dashboard also included the ROC curve, precision-recall curve, metric bar chart, feature importance plot, and predicted probability distribution.

## VI. Conclusion

The project demonstrates a complete machine learning pipeline for heart disease classification. The final Random Forest model achieved strong test performance and benefited from careful preprocessing, clinically motivated feature engineering, and statistically grounded feature selection. The workflow is reproducible and can be reused for future patient data by applying the saved preprocessing artifacts and model bundle.

## References

[1] scikit-learn developers, “LogisticRegression,” *scikit-learn documentation*. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

[2] scikit-learn developers, “RandomForestClassifier,” *scikit-learn documentation*. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

[3] scikit-learn developers, “SelectKBest,” *scikit-learn documentation*. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

[4] scikit-learn developers, “GradientBoostingClassifier,” *scikit-learn documentation*. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

[5] scikit-learn developers, “GridSearchCV,” *scikit-learn documentation*. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

[6] Heart disease classification dataset, provided as `heart.csv` in the assignment workspace.

[7] W. McKinney, *Python for Data Analysis*. Sebastopol, CA, USA: O’Reilly Media, 2022.
