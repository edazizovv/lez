#


#
import pandas


#


#
data = pandas.read_csv('./raw/insurance.csv')

data.to_csv('./dataset.csv', index=False)
