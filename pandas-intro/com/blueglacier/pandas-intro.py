import os
import pandas
import numpy


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
# california_housing_dataframe.hist('housing_median_age')

cities = pandas.DataFrame({'City-Names': city_name, 'Population': population})
print(type(cities['City-Names']))
print(cities['City-Names'])

print(type(cities['City-Names'][1]))
print(cities['City-Names'][1])

print(type(cities[0:2]))
print(cities[0:2])

print(population / 1000)

print(numpy.log(population))

print(population.apply(lambda val: val > 2))

cities['Area-square-kilometers'] = pandas.Series([10,20,10])
cities['Population-Density'] = cities['Population']/cities['Area-square-kilometers']
print(cities)

cities["Is-wide-and-With-population-greater-than-9"] = (cities['Area-square-kilometers'] > 10) & cities['City-Names'].apply(lambda name: name.startswith('De'))
print(cities)

print(city_name.index)
print(cities.index)

print(cities.reindex([2,0,1]))

print(cities.reindex(numpy.random.permutation(cities.index)))

print(cities.reindex([0,4,5,2]))