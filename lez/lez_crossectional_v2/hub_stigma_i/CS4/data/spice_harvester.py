#


#
import pandas


#


#
data = pandas.read_csv('./raw/nyc-rolling-sales.csv').iloc[:, 1:]
data['BBL'] = data['BOROUGH'].astype(dtype=str) + "-" + data['BLOCK'].astype(dtype=str) + "-" \
              + data['LOT'].astype(dtype=str)
data = data.set_index('BBL')

data.to_csv('./dataset.csv')
