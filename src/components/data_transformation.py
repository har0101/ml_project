import sys
from src.logger import logging
from src.exception import CustomException
# from src.components.data_ingestion import DataIngestion
# from src.components.data_ingestion import DataIngestionconfig
from src.utils import save_object
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder

from dataclasses import dataclass

@dataclass
class DataTransformationConfig():
    preprocessor_obr_file_path = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.Data_transformation_config = DataTransformationConfig
        
    def perform_data_transform(self):
        '''
        this function does data transformation
        '''
        try:
            df = pd.read_csv(self.ingestion_config.train_data_path)
            numeric_column_df = df.select_dtypes(exclude='object')
            numeric_column = numeric_column_df.columns

            df = pd.read_csv(self.ingestion_config.train_data_path)
            cat_column_df = df.select_dtypes(include='object')
            cat_column = cat_column_df.columns

            num_pipeline = Pipeline(
                steps=[
                ("inputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
                ]
            )
            logging.info("numerical_pipeline completed")

            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("encoder",OrdinalEncoder()),
                ("scaler",StandardScaler())
                ]
            )
            logging.info("categorical_pipeline completed")

            perprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numeric_column),
                ("cat_pipeline",cat_pipeline,cat_column)
                
                ]
            
            )
            logging.info("combined both pipeline using ColumnTransformer")
            return perprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("train and test data are loaded for transformation")

            preprocessor_obj = self.perform_data_transform()
            logging.info("fecting data_transformation object")

            target_column = ['Loan_Status']

            feature_train_df = train_df.drop([target_column],axis=1)
            target_test_df = test_df[target_column]

            logging.info("applying preprocessing to feature and target")

            feature_train_arr = preprocessor_obj.fit_transform(feature_train_df)
            target_test_arr = preprocessor_obj.transform(target_test_df)

            train_arr = np.c_[feature_train_arr,np.array(feature_train_df)]
            test_arr = np.c_[target_test_arr,np.array(target_test_df)]

            logging.info("preprocessor completed")

            save_object(
                file_path = self.Data_transformation_config.preprocessor_obr_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.Data_transformation_config.preprocessor_obr_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
                

