import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot
import numpy
import pandas
import os
from sklearn import metrics
import tensorflow
from tensorflow.python.data import Dataset

tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
pandas.options.display.max_rows = 10
pandas.options.display.float_format = '{:.1f}'.format


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

# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]

# Configure a numeric feature column for total_rooms.
feature_columns = [tensorflow.feature_column.numeric_column("total_rooms")]

# Define the label.
targets = california_housing_dataframe["median_house_value"]

# Use gradient descent as the optimizer for training the model.
my_optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tensorflow.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.

linear_regressor = tensorflow.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

       Args:
         features: pandas DataFrame of features
         targets: pandas DataFrame of targets
         batch_size: Size of batches to be passed to the model
         shuffle: True or False. Whether to shuffle the data.
         num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
       Returns:
         Tuple of (features, labels) for next data batch
       """
    # Convert pandas data into a dict of numpy arrays.
    features = {key: numpy.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    dataSet = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    dataSet = dataSet.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        dataset = dataSet.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = dataset.make_one_shot_iterator().get_next()

    return features, labels


_ = linear_regressor.train(input_fn=lambda: my_input_fn(my_feature, targets),
                           steps=100
                           )
