import pandas

d = './precomputed/linkage_2/{0}/{1}.csv'


def get_clust(sample, method, n_clusters):
    if method == 'hdbscan':
        tb = pandas.read_csv(d.format(sample, method), index_col=0)
    else:
        tb = pandas.read_csv(d.format(sample, '{0}_{1}'.format(method, n_clusters)), index_col=0)
    return tb


# test = get_clust(sample='CS1', method='hdbscan', n_clusters='7')
