# SMS-Spam-Detection-Using-PySpark-ML-lib-on-Databricks
Developed a machine learning pipeline for SMS spam detection using Apache Spark's MLlib. The project involved processing a large SMS dataset to classify messages as spam or not spam. Key tasks included:

Data Preparation: Loaded and preprocessed a tab-delimited dataset containing SMS messages and their labels.

Pipeline Development: Built a machine learning pipeline leveraging Spark's StringIndexer, Tokenizer, and CountVectorizer for feature engineering.

Model Training & Tuning: Trained a Logistic Regression model and optimized hyperparameters using a cross-validation approach with a ParamGrid for elastic net regularization.

Evaluation: Assessed model performance with a BinaryClassificationEvaluator, achieving high accuracy and area under the ROC curve on test data.

Tools & Technologies: Apache Spark (PySpark, MLlib), Logistic Regression with Cross-Validation, Python


