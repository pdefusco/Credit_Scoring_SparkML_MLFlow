import random, os
import numpy as np
from pyspark.sql import Row
from sklearn import neighbors
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.stat import Statistics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer
from pyspark.ml import Pipeline
from pyspark.mllib.stat import Statistics
from pyspark.ml.linalg import DenseVector
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class PySparkOversamplingExperiment:
    def __init__(self, mlflow_experiment_name, mlflow_exeperiment_tagsdict):
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_exeperiment_tags = mlflow_exeperiment_tagsdict

    def vectorizerFunction(dataInput, TargetFieldName):
        '''
        Function that vectorizes data and returns an rdd.
        '''

        if(dataInput.select(TargetFieldName).distinct().count() != 2):
            raise ValueError("Target field must have only 2 distinct classes")
        columnNames = list(dataInput.columns)
        columnNames.remove(TargetFieldName)
        dataInput = dataInput.select((','.join(columnNames)+','+TargetFieldName).split(','))
        assembler=VectorAssembler(inputCols = columnNames, outputCol = 'features')
        pos_vectorized = assembler.transform(dataInput)
        vectorized = pos_vectorized.select('features',TargetFieldName).withColumn('label',pos_vectorized[TargetFieldName]).drop(TargetFieldName)
        
        return vectorized

    def smotesampling(vectorized, k = 5, minorityClass = 1, majorityClass = 0, percentageOver = 200, percentageUnder = 100):
        '''
        Function that accepts oversampling hyperparameters and returns an oversampled dataset.
        '''
        
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

    #Creates a Pipeline Object
    def prepare_data(spark_df):
        '''
        Function that scales and transforms data so it can be oversampled.
        '''
        
        remove = ['addr_state', 'earliest_cr_line', 'home_ownership', 'initial_list_status', 'issue_d', 'emp_length',
                  'loan_status', 'purpose', 'sub_grade', 'term', 'title', 'zip_code', 'application_type']
        df = df.drop(*remove)

        #We will choose these features for our baseline model:
        #Creating list of categorical and numeric features
        cat_cols = [item[0] for item in df.dtypes if item[1].startswith('string')]
        num_cols = [item[0] for item in df.dtypes if item[1].startswith('in') or item[1].startswith('dou')]
        num_features, cat_features = num_cols, cat_cols
        df = df.dropna()
        df = df.select(num_features)
        
        stages= []
        scale_cols = df.columns
        scale_cols.remove('is_default')

        #Assembling mixed data type transformations:
        assembler = VectorAssembler(inputCols=scale_cols, outputCol="features")
        stages += [assembler]

        #Standard scaler
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=True)
        stages += [scaler]

        #Creating and running the pipeline:
        pipeline = Pipeline(stages=stages)
        pipelineModel = pipeline.fit(spark_df)
        out_df = pipelineModel.transform(spark_df)
        df_model.rdd.map(lambda x: (x["is_default"], DenseVector(x["scaledFeatures"])))
        df_pre_smote = spark.createDataFrame(input_data, ["is_default", "scaledFeatures"])
        
        return df_pre_smote

tags = {
    "engineering": "ML Platform",
    "release.candidate": "RC1",
    "release.version": "2.2.0",
}
    
PySparkOversamplingExperiment("oversampling-experiment", tags)
    
    
spark = SparkSession.builder.appName("PythonSQL")\
            .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-2")\
            .config("spark.yarn.access.hadoopFileSystems",os.environ["STORAGE"])\
            .getOrCreate()

#Step1: Load data
df = spark.sql("SELECT * FROM default.LC_Table")





#Step2: Transform and Scale
df_pre_smote = prepare_data(df)

#Step3: Oversample
df_post_smote = SmoteSampling(vectorizerFunction(df_pre_smote, 'is_default'), k = 3, minorityClass = 1, majorityClass = 0, percentageOver = 400, percentageUnder = 100)

def extract(row):
    return tuple(row.features.toArray().tolist()) + (row.label, )

df_smote_table = df_post_smote.rdd.map(extract).toDF(df.columns)

#Step4: Back up to Spark Table
createtable(df_pre_smote, df_post_smote)

#Additions for experiments:
import cdsw
cdsw.track_metric("SMOTE New Row Count", df_smote.count())
