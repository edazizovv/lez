#


#
import pandas


#


#
data = pandas.read_csv('./raw/housing.csv')

data.to_csv('./dataset.csv', index=False)
