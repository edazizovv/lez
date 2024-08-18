
import os
import pandas
from matplotlib import pyplot
import plotly.express as px


d = './precomputed/hists/{0}_{1}.csv'
g = './precomputed/linkage_1/pre/{0}_small_nlags.csv'

samples = os.listdir('./hub/')

for sa in samples:

    dataset = pandas.read_csv(g.format(sa))
    n, bins, pt = pyplot.hist(dataset['y'].values, bins=100)
    pandas.Series(n).to_csv(d.format(sa, 'n'), index=False)
    pandas.Series(bins).to_csv(d.format(sa, 'bins'), index=False)
