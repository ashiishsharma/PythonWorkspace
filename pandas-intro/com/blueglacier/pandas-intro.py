import os
import pandas


def getFilePath():
    current_working_directory_path = os.getcwd()
    os.chdir('..\..')
    project_path = os.getcwd()

    file_path = project_path + '\\resources\california_housing_train.csv'

    os.chdir(current_working_directory_path)
    print(os.getcwd())
    print(file_path)
    return file_path


print(pandas.__version__)

city_name = pandas.Series(['Delhi', 'Mumbai', 'Kolkata'])
print(city_name)

population = pandas.Series([10, 9, 8])
print(population)

data_frame = pandas.DataFrame({'City-Names': city_name, 'Population': population})
print(data_frame)

california_housing_dataframe = pandas.read_csv(getFilePath(), sep=",")
# print(california_housing_dataframe)
print(california_housing_dataframe.head())
print(california_housing_dataframe.describe())
california_housing_dataframe.hist('housing_median_age')