#


#
import pandas


#


#
data = pandas.read_csv('./raw/houses_to_rent_v2.csv')

data.to_csv('./dataset.csv', index=False)
