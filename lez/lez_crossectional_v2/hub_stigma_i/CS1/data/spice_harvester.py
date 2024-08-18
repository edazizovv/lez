#


#
import pandas


#


#
data = pandas.read_csv('./raw/Real estate.csv')
data = data.set_index('No')

data.to_csv('./dataset.csv')
