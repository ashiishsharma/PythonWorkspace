import math

import numpy
import pandas
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

def getFilePath():
    current_working_directory_path = os.getcwd()
    os.chdir('..\..')
    project_path = os.getcwd()

    file_path = project_path + '\\resources\california_housing_train.csv'

    os.chdir(current_working_directory_path)
    print(os.getcwd())
    print(file_path)
    return file_path


california_housing_dataframe = pandas.read_csv(getFilePath(), sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    numpy.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0
print(california_housing_dataframe)
print(california_housing_dataframe.describe())