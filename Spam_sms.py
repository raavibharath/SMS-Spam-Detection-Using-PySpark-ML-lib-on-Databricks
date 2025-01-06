# Databricks notebook source
# MAGIC %md
# MAGIC # DS-610 Week 6 Homework: Machine Learning on Apache Spark
# MAGIC  we will build a spam classifier on data stored on the Cloud.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Tokenizer
from platform import python_version
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading Data
# MAGIC Reminder: It is highly recommended that you try this homework on Saint Peters' Databricks. Depending on whether you are running on the cloud or locally, adjust the data source accordingly below.

# COMMAND ----------

#Data source
data_source = "/FileStore/shared_uploads/dlee5@saintpeters.edu/ds610/SMSSpamCollection"
# Load data and rename column. DO NOT MODIFY
df = spark.read.option("header", "false") \
    .option("delimiter", "\t") \
    .option("inferSchema", "true") \
    .csv(data_source) \
    .withColumnRenamed("_c0", "label_string") \
    .withColumnRenamed("_c1", "sms")
df.limit(10).show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Pipelines
# MAGIC First we will declare `pipeline_stages` which will hold the complete steps for getting our dataset into the format for model training.

# COMMAND ----------


pipeline_stages = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 1a
# MAGIC We note that the `label_string` column consists of the label set `{ ham, spam }` (`ham` means not spam). Your task is now to create a pipeline stage which takes the input column `label_string` and outputs into a new column `label` which performs the following mapping: `{ham -> 0, spam -> 1}`. This is necessary since we would like to train a classifier on the training data.
# MAGIC
# MAGIC You may find the following useful:
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html

# COMMAND ----------

# Convert the "label_string" colun to 0/1 using StringIndexer.
indexer = StringIndexer(inputCol="label_string", outputCol="label")
# Add to the pipeline stage. DO NOT MODIFY.
pipeline_stages.append(indexer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 1b
# MAGIC Now we take a look at the `sms` column which represents the raw SMS message. We have to turn this into a more useful form. First we have to *tokenize* the message. For example:
# MAGIC ```
# MAGIC The quick brown fox jumps over the lazy dog -> [ "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog" ]
# MAGIC  You may find the following useful:
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html

# COMMAND ----------

# Tokenize the "sms" column into the list of words under the column name "words" (inputCol="sms" and outputCol="words")
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html
tokenizer = Tokenizer(inputCol='sms',outputCol='words') 

# Add to the pipeline stage. DO NOT MODIFY.
pipeline_stages.append(tokenizer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 1c
# MAGIC Finally, we create a feature vector called count vectorizer. For example:
# MAGIC ```
# MAGIC The quick brown fox jumps over the lazy dog -> [ "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog" ] -> (1000, [8, 10, 23, 34, 58, 100, 110, 112, 140], [2, 1, 1, 1, 1, 1, 1, 1])
# MAGIC ```
# MAGIC where the first number denotes the total size of the vocabulary, the second list denotes the word index for each of the words in the original sentence, the third list denotes the corresponding count for each word in the original sentence.
# MAGIC
# MAGIC Your task is to create a count vector feature out of the `words` column under the column name `features`, using `(inputCol="words", outputCol="features")`.
# MAGIC
# MAGIC You may find this useful:
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.CountVectorizer.html

# COMMAND ----------

# Make a count vector feature out of the words column under the column name "features" (inputCol="words", outputCol="features")
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.CountVectorizer.html
cv = CountVectorizer(inputCol='words', outputCol='features',minDF=2.0)

# Add to the pipeline stage. DO NOT MODIFY.
pipeline_stages.append(cv)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interlude
# MAGIC If your Part 1 is correctly implemented, then the following code should transform the data for model training. Note that for modeling training, only two of the columns in the transformed data is used, namely `label` and `features` columns.

# COMMAND ----------

# DO NOT MODIFY.
pipeline = Pipeline(stages=pipeline_stages)
data=pipeline.fit(df).transform(df)

# COMMAND ----------

data.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Model Training
# MAGIC Now we are ready to train out model. There are two parts to this exercise.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 2a
# MAGIC Let us first divide the `data` into `train` and `test` in the ratio of 0.8 to 0.2. You may find the following useful:
# MAGIC https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html

# COMMAND ----------

# Divide into train and test.
train, test = data.randomSplit([0.8,0.2],seed=42)          

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interlude
# MAGIC The machine learning model we will be working with is called logistic regression. For your reference, the sample code for training is shown below.

# COMMAND ----------

# Sample logistic regression code.
lr = LogisticRegression()
lrModel = lr.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 2b
# MAGIC In machine learning, we have to do what is a called hyperparameter tuning on a combination of parameters. For logistic regression, we can tune two hyperparameters, namely the $L_1$ regularizer and the $L_2$ regularizer in its elastic net formulation. For details on elastic net regularization, please see:
# MAGIC https://en.wikipedia.org/wiki/Elastic_net_regularization
# MAGIC
# MAGIC The Wikipedia entry will just have a elastic net formulation of linear regression. For elastic net formulation of logistic regression, the cost function is a bit different and involves adding L1 and L2 regularization to cross entropy loss of the data. For more details, you may want to consult a machine learning textbook.
# MAGIC
# MAGIC Your task is here to finish the implementation that runs cross-validation on the set of elastic net regularization parameter below. Most of the skeleton code is provided for you, including setting up the parameter grid which can be plugged into the `CrossValidator`.
# MAGIC
# MAGIC For more details: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html

# COMMAND ----------

# Your code for Part 2b goes here.
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create ParamGrid and Evaluator for Cross Validation
paramGrid = ParamGridBuilder().baseOn({lr.maxIter: 100}).baseOn({lr.fitIntercept: False}).addGrid(lr.elasticNetParam, [0, 0.25, 0.5, 0.75, 1]).build()
cvEvaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")

# Run Cross-validation
cv = CrossValidator(estimator=lr, evaluator=cvEvaluator, estimatorParamMaps=paramGrid, numFolds=3)
cvModel = cv.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Model Evaluation
# MAGIC Evaluate the best model trained in Part 2 on the `test` set from Part 2a. You would need to have `cvModel` from the previous part perform `transform` the `test` set.
# MAGIC
# MAGIC For example:
# MAGIC https://spark.apache.org/docs/latest/ml-classification-regression.html#binomial-logistic-regression
# MAGIC
# MAGIC Search for `# Make predictions` in this page for an example.

# COMMAND ----------

# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)       
print(cvPredictions.take(10))

# Evaluate bestModel found from Cross Validation. DO NOT MODIFY.
print ("Test Area Under ROC: ", cvEvaluator.evaluate(cvPredictions))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC We are done! In order to see what parameter sets were explored during the grid search, we can run the command below.

# COMMAND ----------


cvModel.bestModel.extractParamMap()

# COMMAND ----------

