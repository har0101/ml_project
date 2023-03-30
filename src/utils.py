import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obr:
            dill.dump(obj, file_obr)

    except Exception as e:
        raise CustomException(e, sys)




