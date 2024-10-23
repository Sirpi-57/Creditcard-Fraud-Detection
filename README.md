# Creditcard-Fraud-Detection

**Objective**

The goal of this project is to build a robust machine learning model to detect fraudulent credit card transactions. Due to the highly imbalanced nature of the dataset, the project focuses on experimenting with various models and techniques such as data scaling, undersampling, oversampling, and fine-tuning of hyperparameters to achieve better detection of the minority class (fraudulent transactions).

**Explanations of Concepts:**

**Class Imbalance**

In fraud detection, we typically deal with imbalanced datasets where fraudulent transactions make up a tiny fraction of the total. For this project, fraud accounts for only 0.172% of the data. Various techniques like undersampling the majority class, oversampling the minority class, and using performance metrics such as Precision-Recall and Area Under the Curve (AUC) are necessary to properly handle this imbalance.

**Scaling Features**

To bring the data to a common scale, both Min-Max Scaling and Standard Scaling were applied to the dataset. This helps models such as Logistic Regression and XGBoost perform better since unscaled features might skew results.

**Handling Skewness**

Skewness in data can lead to models that underperform. Techniques such as logarithmic transformation or scaling were applied to mitigate the skewness of the data and improve model accuracy.

**Logistic Regression Model**

Logistic Regression is a benchmark model that is simple but can be highly effective in detecting patterns. It was used as a baseline model, and its performance was improved with techniques such as cross-validation and hyperparameter tuning.

**XGBoost Model**

XGBoost, a more advanced ensemble learning method, was used to improve detection results. With techniques like hyperparameter tuning (learning rate, max depth, and subsampling), it significantly improved precision and recall for fraud detection.

**Under and Oversampling**

Due to the severe class imbalance, RandomUnderSampler and RandomOverSampler were applied to adjust the dataset and better train the model. These techniques balance the number of instances of each class so that the model can learn to detect fraud more accurately.

**Project Deliverables**

**Data Preprocessing:**

Scaling of features such as 'Amount' and dropping unnecessary columns like 'Time.'
Handling skewness in data for better performance.
Techniques for mitigating class imbalance, such as RandomUnderSampler and RandomOverSampler.

**Modeling:**

Logistic Regression model as a baseline for comparison.
XGBoost model fine-tuned using cross-validation and hyperparameter tuning.
Performance metrics evaluated using Precision, Recall, F1-Score, and Accuracy.

**Performance Improvement:**

Benchmarking results with oversampling and undersampling.
Hyperparameter tuning for both Logistic Regression and XGBoost models.
Achieved precision and recall improvements through oversampling.

**Evaluation Metrics:**

Focus on metrics like Precision-Recall Curve, F1-Score, and Area Under Curve (AUC) due to class imbalance.
Confusion matrices were used to understand model performance on both the majority (non-fraudulent) and minority (fraudulent) classes.

**Results:**

**The Best Model : Hypertuned XGBoost Model with SMOTE Over Sampling**

![Screenshot (166)](https://github.com/user-attachments/assets/3ca8e173-2a78-4ae0-9ca7-d4ebb982c208)

![Screenshot (167)](https://github.com/user-attachments/assets/c91536e6-1983-4f77-818f-9ed9d22509cb)

**Future Scope**

**Model Improvement:**

Explore deep learning techniques such as Neural Networks and Autoencoders for better anomaly detection.
Experiment with other advanced algorithms such as LightGBM or CatBoost to improve fraud detection performance.

**Feature Engineering:**

Apply additional feature engineering techniques to extract more meaningful information from the data, potentially improving model accuracy.

**Real-time Fraud Detection:**

Implementing the model in a real-time detection system using techniques like sliding windows or stream processing.
Integration with an API for continuous model improvement based on new fraudulent transaction patterns.

**Cost-Sensitive Learning:**

Implement cost-sensitive learning to better handle the economic impact of misclassifying fraudulent transactions.

**Dataset Link:https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud**
