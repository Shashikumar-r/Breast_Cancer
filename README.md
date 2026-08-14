# Machine Learning Assignment 2: Classification Models & Streamlit Deployment

## a. Problem Statement
The objective of this project is to build, evaluate, and deploy an interactive Machine Learning web application for binary classification. The application allows users to upload test data, select from multiple pre-trained classification algorithms, and instantly view performance metrics and confusion matrices. This end-to-end project demonstrates the complete ML workflow from data preprocessing and model training to UI design and cloud deployment.

## b. Dataset Description
**Dataset Chosen:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Source:** UCI Machine Learning Repository / Scikit-Learn built-in datasets.
* **Instances:** 569 (Satisfies minimum 500 requirement)
* **Features:** 30 real-valued input features (Satisfies minimum 12 requirement).
* **Target Variable:** Binary classification (Malignant = 0, Benign = 1).
* **Reason for Choice:** It is a classic, highly relevant medical diagnostic problem with well-separated classes, making it ideal for demonstrating multiple classification algorithms.

## c. Github Repository Link
* [Insert Your GitHub Repository Link Here]

## d. Models Used
*(Note: The assignment mentions "6 ML models" in the text but lists 5 in the table. I have implemented the 5 listed models and added Support Vector Machine (SVM) as the 6th model to strictly fulfill the requirement).*

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9439 |
| Decision Tree | 0.9474 | 0.9440 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| kNN | 0.9474 | 0.9820 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| Naive Bayes | 0.9649 | 0.9974 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| Random Forest (Ensemble) | 0.9649 | 0.9953 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| Support Vector Machine | 0.9825 | 0.9974 | 0.9726 | 1.0000 | 0.9861 | 0.9630 |

### Observations about Model Performance
| ML Model Name | Observation about Model Performance |
| --- | --- |
| Logistic Regression | Very high performance across all metrics (Accuracy 97.37%, AUC 99.74%). Acts as an exceptionally strong linear baseline for this dataset. |
| Decision Tree | Slightly lower performance (Accuracy 94.74%) compared to others. This drop is likely due to overfitting on the training data or lack of pruning parameters. |
| kNN | Shares the same accuracy as the Decision Tree (94.74%), but boasts a much higher AUC (98.2%). Shows decent class separation but struggles slightly with the precision/recall balance compared to linear models. |
| Naive Bayes | Strong performance (Accuracy 96.49%, AUC 99.74%), demonstrating that the assumption of feature independence isn't highly detrimental to this specific dataset. |
| Random Forest (Ensemble) | Improved noticeably over the single Decision Tree (Accuracy 96.49%, AUC 99.53%), demonstrating the power of bagging/ensembling to reduce variance, though it slightly trails Logistic Regression. |
| Support Vector Machine | The absolute best performer. Achieved the highest Accuracy (98.25%), perfect Recall (100%), and the highest MCC (0.963). It effectively finds the optimal hyperplane in high-dimensional space. |
| Overall Winner for your dataset? | **Support Vector Machine (SVM)**. Because this is a medical diagnostic dataset, avoiding False Negatives (missing a malignant tumor) is critical. SVM achieves a perfect Recall (100%) alongside the highest Accuracy and MCC score, making it the safest and most reliable model. |