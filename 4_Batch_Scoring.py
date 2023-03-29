#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.mllib.stat import Statistics
from pyspark.ml.linalg import DenseVector
from pyspark.sql import functions as F


# In[4]:


import random
import numpy as np
from pyspark.sql import Row
from sklearn import neighbors
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.stat import Statistics


# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


from pyspark.sql import SparkSession


# In[7]:


#from LC_Helper import vectorizerFunction, SmoteSampling


# In[8]:


spark = SparkSession.builder.appName("LC_Baseline_Model")\
                        .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-2")\
                        .config("spark.yarn.access.hadoopFileSystems",os.environ["STORAGE"])\
                        .getOrCreate()

df = spark.sql("SELECT * FROM default.lc_smote_subset")

#Creating list of categorical and numeric features
num_cols = [item[0] for item in df.dtypes if item[1].startswith('in') or item[1].startswith('dou')]


# In[46]:


df = df.dropna()


# In[47]:


df = df.select(['acc_now_delinq', 'acc_open_past_24mths', 'annual_inc', 'avg_cur_bal', 'funded_amnt', 'is_default'])


# In[48]:


train = df.sampleBy("is_default", fractions={0: 0.8, 1: 0.8}, seed=10)


# In[49]:


test = df.subtract(train)


# Creating Model Pipeline

# In[50]:


#Creates a Pipeline Object including One Hot Encoding of Categorical Features
def make_pipeline(spark_df):

    for c in spark_df.columns:
        spark_df = spark_df.withColumn(c, spark_df[c].cast("float"))

    stages= []

    cols = ['acc_now_delinq', 'acc_open_past_24mths', 'annual_inc', 'avg_cur_bal', 'funded_amnt']

    #Assembling mixed data type transformations:
    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    stages += [assembler]

    #Scaling features
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    stages += [scaler]

    #Logistic Regression
    lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='is_default', maxIter=10, regParam=0.1, elasticNetParam=0.1)
    stages += [lr]

    #Creating and running the pipeline:
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(spark_df)

    return pipelineModel


# In[51]:


pipelineModel = make_pipeline(train)


# In[52]:


for c in test.columns:
    test = test.withColumn(c, test[c].cast("float"))


# In[53]:


df_model = pipelineModel.transform(test)


# In[57]:

batch_score = df_model.rdd.map(lambda x: (x["acc_now_delinq"],
                                         x["acc_open_past_24mths"],
                                         x["annual_inc"],
                                         x["avg_cur_bal"],
                                         x["funded_amnt"],
                                         x["is_default"],
                                         x["prediction"],
                                         float(x["probability"][1])))\
                                          .toDF(["acc_now_delinq",
                                                "acc_open_past_24mths",
                                                "annual_inc", 
                                                "avg_cur_bal", 
                                                "funded_amnt", 
                                                "is_default", 
                                                "prediction",
                                                "probability"])

batch_score.write.format("parquet")\
  .mode("overwrite")\
  .saveAsTable(
    'default.LC_Batch_Scored'
)

input_data = df_model.rdd.map(lambda x: (x["is_default"], x["prediction"], float(x["probability"][1])))

#Saving pipeline to S3:
#pipelineModel.write().overwrite().save("s3a://demo-aws-2/datalake/pdefusco/pipeline")
