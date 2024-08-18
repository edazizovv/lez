#


#
import pandas


#


#
data = pandas.read_csv('./raw/train-data.csv').iloc[:, 1:]

data.to_csv('./dataset.csv')
