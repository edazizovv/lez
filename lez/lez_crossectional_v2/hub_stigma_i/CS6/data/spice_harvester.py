#


#
import pandas


#


#
train = pandas.read_csv('./raw/train.csv')
test = pandas.read_csv('./raw/test.csv')
data = pandas.concat((train, test), axis=0, ignore_index=True)

data.to_csv('./dataset.csv', index=False)
