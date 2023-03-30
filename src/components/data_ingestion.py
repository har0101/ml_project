import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass



from src.components.data_transformation import DataTransformationConfig 
from src.components.data_transformation import DataTransformation



@dataclass
class DataIngestionconfig:
    raw_data_path: str=os.path.join('artifacts','data.csv')
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

        # pass
    def initiate_ingestion(self):
        logging.info("initiated data ingestion")

        try:
            df = pd.read_csv('notebook/loan_sanction_train.csv')
            logging.info("data has been loaded")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("raw data is saved as data.csv")

            train_data,test_data = train_test_split(df,test_size=0.25,random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("train data is saved as train.csv")

            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("test data is saved as train.csv")


            return(
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            CustomException(e,sys)
    

if __name__ == '__main__':
    obr = DataIngestion()
    obr.initiate_ingestion()


    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train,test)




