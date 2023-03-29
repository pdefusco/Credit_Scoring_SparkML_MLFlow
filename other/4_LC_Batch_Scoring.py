#!/usr/bin/env python
# coding: utf-8


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


spark = SparkSession.builder.appName("LC_Job")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-1")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://demo-aws-2/")\
    .getOrCreate()


# In[9]:


def vectorizerFunction(dataInput, TargetFieldName):
    if(dataInput.select(TargetFieldName).distinct().count() != 2):
        raise ValueError("Target field must have only 2 distinct classes")
    columnNames = list(dataInput.columns)
    columnNames.remove(TargetFieldName)
    dataInput = dataInput.select((','.join(columnNames)+','+TargetFieldName).split(','))
    assembler=VectorAssembler(inputCols = columnNames, outputCol = 'features')
    pos_vectorized = assembler.transform(dataInput)
    vectorized = pos_vectorized.select('features',TargetFieldName).withColumn('label',pos_vectorized[TargetFieldName]).drop(TargetFieldName)
    return vectorized


# In[10]:


def SmoteSampling(vectorized, k = 5, minorityClass = 1, majorityClass = 0, percentageOver = 200, percentageUnder = 100):
    if(percentageUnder > 100|percentageUnder < 10):
        raise ValueError("Percentage Under must be in range 10 - 100");
    if(percentageOver < 100):
        raise ValueError("Percentage Over must be in at least 100");
    dataInput_min = vectorized[vectorized['label'] == minorityClass]
    dataInput_maj = vectorized[vectorized['label'] == majorityClass]
    feature = dataInput_min.select('features')
    feature = feature.rdd
    feature = feature.map(lambda x: x[0])
    feature = feature.collect()
    feature = np.asarray(feature)
    nbrs = neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto').fit(feature)
    neighbours =  nbrs.kneighbors(feature)
    gap = neighbours[0]
    neighbours = neighbours[1]
    min_rdd = dataInput_min.drop('label').rdd
    pos_rddArray = min_rdd.map(lambda x : list(x))
    pos_ListArray = pos_rddArray.collect()
    min_Array = list(pos_ListArray)
    newRows = []
    nt = len(min_Array)
    nexs = percentageOver//100
    for i in range(nt):
        for j in range(nexs):
            neigh = random.randint(1,k)
            difs = min_Array[neigh][0] - min_Array[i][0]
            newRec = (min_Array[i][0]+random.random()*difs)
            newRows.insert(0,(newRec))
    newData_rdd = spark.sparkContext.parallelize(newRows)
    newData_rdd_new = newData_rdd.map(lambda x: Row(features = x, label = 1))
    new_data = newData_rdd_new.toDF()
    new_data_minor = dataInput_min.unionAll(new_data)
    new_data_major = dataInput_maj.sample(False, (float(percentageUnder)/float(100)))
    return new_data_major.unionAll(new_data_minor)


# In[11]:


df = spark.sql("SELECT * FROM default.LC_Table")


# In[12]:


df = df.limit(10000)


#We will drop this feature based on its imbalance


#We remove categorical features that have too broad a set of values, or are highly imbalanced, or could cause data leakage.
#We can elaborate and use them for feature extraction later, but they are not needed for a baseline
remove = ['addr_state', 'earliest_cr_line', 'home_ownership', 'initial_list_status', 'issue_d', 'emp_length',
          'loan_status', 'purpose', 'sub_grade', 'term', 'title', 'zip_code', 'application_type', 'desc', 'issue_month',
         'id', 'emp_title', 'grade']
df = df.drop(*remove)


# Baseline Feature Exploration

# In[19]:


#Creating list of categorical and numeric features
cat_cols = [item[0] for item in df.dtypes if item[1].startswith('string')]
num_cols = [item[0] for item in df.dtypes if item[1].startswith('in') or item[1].startswith('dou')]

#We will choose these features for our baseline model:
num_features, cat_features = num_cols, cat_cols


# In[29]:


#Count number of nulls for each column:
nulls = df.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas()


# In[30]:


nulls.T[(nulls.T > 0).any(axis=1)].index


# In[31]:


impute_list = list(nulls.T[(nulls.T > 0).any(axis=1)].index)


# In[32]:


#Both attributes are continuous so we will impute


# In[33]:


from pyspark.ml.feature import Imputer


# In[34]:


imputer = Imputer(inputCols=impute_list, outputCols=[i+"_imp" for i in impute_list])


# In[35]:


model = imputer.fit(df)


# In[36]:


df = model.transform(df)


# In[37]:


num_features+=[i+"_imp" for i in impute_list]


# In[38]:


num_features = [i for i in num_features if i not in impute_list]


# In[39]:


num_features.remove("is_default")


# In[40]:


#Creates a Pipeline Object including One Hot Encoding of Categorical Features
def make_pipeline(spark_df, num_att, cat_att):
    stages= []

    for col in cat_att:

        stringIndexer = StringIndexer(inputCol = col , outputCol = col + '_StringIndex')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[col + '_ClassVect'])
        stages += [stringIndexer, encoder]

    #Assembling mixed data type transformations:
    assemblerInputs = [c + "_ClassVect" for c in cat_att] + num_att
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

    stages += [assembler]

    #Creating and running the pipeline:
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(spark_df)
    out_df = pipelineModel.transform(spark_df)

    return out_df


# In[41]:


df.dtypes


# In[42]:


df_model = make_pipeline(df, num_features, cat_features)


# In[43]:


input_data = df_model.rdd.map(lambda x: (x["is_default"], DenseVector(x["features"])))


# In[44]:


df_baseline = spark.createDataFrame(input_data, ["is_default", "features"])


# In[45]:


df_baseline.show(5)


# In[46]:


#Scaling Data prior to SMOTE


# In[47]:


scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)


# In[48]:


scalerModel = scaler.fit(df_baseline)


# In[49]:


scaledData = scalerModel.transform(df_baseline)


# In[50]:


scaledData = scaledData.drop("features")


# In[51]:


scaledData.show(5)


# In[52]:


#column_names
temp = scaledData.rdd.map(lambda x:[float(y) for y in x['scaledFeatures']]).toDF(num_features + cat_features)


# In[53]:


#Notice we now have 56 columns rather than 50 - we have one hot encoded our categorical features
len(temp.columns)


# In[54]:


cols = list(df.columns)
cols.remove("is_default")


# In[55]:


import pyspark.sql.functions as sparkf

# This will return a new DF with all the columns + id
df_join = df.withColumn('id', sparkf.monotonically_increasing_id())
temp = temp.withColumn('id', sparkf.monotonically_increasing_id())


# In[56]:


df_join = df_join.select('id', 'is_default')


# In[57]:


temp = temp.join(df_join, temp.id == df_join.id, 'inner').drop(df_join.id).drop(temp.id)


# In[58]:


#Smote


# In[59]:


df_smote = SmoteSampling(vectorizerFunction(temp, 'is_default'), k = 2, minorityClass = 1, majorityClass = 0, percentageOver = 400, percentageUnder = 100)

df_smote.sampleBy("label", fractions={0: 0.8, 1: 0.8}, seed=10).show(2)


# In[67]:


train = df_smote.sampleBy("label", fractions={0: 0.8, 1: 0.8}, seed=10)


# In[68]:


test = df_smote.subtract(train)


# In[70]:


#The label counts for the training set
train.groupBy("label").count().show()


# In[71]:


#The label counts for the test set
test.groupBy("label").count().show()


# Logistic Regression

# In[79]:


lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10, regParam=0.1, elasticNetParam=0.2)


# In[80]:


# Fit the model
lrModel = lr.fit(train)


# In[81]:


# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("\n")
print("Intercept: " + str(lrModel.intercept))


# In[82]:


# Make predictions on test data. model is the model with combination of parameters that performed best.
predictions = lrModel.transform(test)

pred_df = predictions.rdd.map(lambda x:[float(y) for y in x['features']]).toDF()
# This will return a new DF with all the columns + id
pred_df_id = pred_df.withColumn('id', sparkf.monotonically_increasing_id())
predictions_id = predictions.withColumn('id', sparkf.monotonically_increasing_id())
out_df = pred_df_id.join(predictions_id, predictions_id.id == pred_df_id.id, 'inner')                .drop(pred_df_id.id).drop(predictions_id.id)


# In[124]:


dropped = ['features', 'rawPrediction', 'probability']
out_df = out_df.drop(*dropped)


# In[125]:


out_df.show()


# In[126]:


out_df.write.format("parquet").mode("overwrite").saveAsTable(
    'default.LC_predictions_Latest'
)


# In[140]:


#saving the model as well:

#lrModel.write().overwrite().save("hdfs:///tmp/models")


# In[ ]:
