import math

import numpy
import pandas
import tensorflow
from IPython import display
from matplotlib import cm, pyplot
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
    dataset = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    dataset = dataset.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = dataset.make_one_shot_iterator().get_next()

    return features, labels


def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    """Trains a linear regression model of one feature.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      input_feature: A `string` specifying a column from `california_housing_dataframe`
        to use as input feature.
    """

    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]

    # Create feature columns.
    feature_columns = [tensorflow.feature_column.numeric_column(my_feature)]

    # Create input functions.
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # Create a linear regressor object.
    my_optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tensorflow.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tensorflow.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Set up to plot the state of our model's line each period.
    pyplot.figure(figsize=(15, 6))
    pyplot.subplot(1, 2, 1)
    pyplot.title("Learned Line by Period")
    pyplot.ylabel(my_label)
    pyplot.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    pyplot.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in numpy.linspace(-1, 1, periods)]

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = numpy.array([item['predictions'][0] for item in predictions])

        # Compute loss.
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)
        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = numpy.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = numpy.maximum(numpy.minimum(x_extents,
                                                sample[my_feature].max()),
                                  sample[my_feature].min())
        y_extents = weight * x_extents + bias
        pyplot.plot(x_extents, y_extents, color=colors[period])
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    pyplot.subplot(1, 2, 2)
    pyplot.ylabel('RMSE')
    pyplot.xlabel('Periods')
    pyplot.title("Root Mean Squared Error vs. Periods")
    pyplot.tight_layout()
    pyplot.plot(root_mean_squared_errors)

    # Output a table with calibration data.
    calibration_data = pandas.DataFrame()
    calibration_data["predictions"] = pandas.Series(predictions)
    calibration_data["targets"] = pandas.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
    pyplot.show()
    return calibration_data


california_housing_dataframe["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"])

calibration_data = train_model(learning_rate=0.05, steps=500, batch_size=5, input_feature="rooms_per_person")

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(calibration_data["predictions"], calibration_data["targets"])

plt.subplot(1, 2, 2)
_ = california_housing_dataframe["rooms_per_person"].hist()

california_housing_dataframe["rooms_per_person"] = (
    california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))

_ = california_housing_dataframe["rooms_per_person"].hist()

calibration_data = train_model(
    learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person")

_ = plt.scatter(calibration_data["predictions"], calibration_data["targets"])

pyplot.show()
