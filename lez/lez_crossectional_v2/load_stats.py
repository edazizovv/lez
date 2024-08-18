import numpy
import pandas


d = './precomputed/linkage_1/{0}/{1}.csv'

targets = {'CS1': 'Y house price of unit area',
           'CS2': 'G3',
           'CS3': 'Price',
           'CS4': 'SALE PRICE',
           'CS5': 'rent amount (R$)',
           'CS6': 'TARGET(PRICE_IN_LACS)',
           'CS7': 'median_house_value',
           'CS8': 'charges',
           'TS1': 'meantemp',
           'TS2': '徳川埋蔵金_ja.wikipedia.org_all-access_spider',
           'TS3': 'T_station_vitoria.csv',
           'TS4': 'population',
           'TS5': 'pytorch',
           'TS6': 'meter',
           'TS7': 'USAGE',
           'TS8': 'Apparent Temperature (C)'}


def get_corr(sample, measure, direct, treat=True):
    mx = pandas.read_csv(d.format(sample, 'corr_{0}_{1}'.format(measure, direct)), index_col=0)
    factors_mask = [x != targets[sample] for x in mx.columns.values]
    mx = mx.loc[factors_mask, factors_mask]
    mxx = mx.values
    if treat:
        mxx[numpy.triu_indices(n=mx.shape[0])] = numpy.nan
        mxx = numpy.abs(mxx)
    mx = pandas.DataFrame(data=mxx, columns=mx.columns, index=mx.index)
    if measure == 'mutual_info':
        mx = mx.fillna(1)
    return mx


def get_perf(sample, measure, direct):
    mx = pandas.read_csv(d.format(sample, 'corr_{0}_{1}'.format(measure, direct)), index_col=0)
    factors_mask = [x != targets[sample] for x in mx.columns.values]
    arr = mx.loc[factors_mask, targets[sample]]
    arr = pandas.Series(data=numpy.abs(arr.values), index=arr.index)
    return arr


def get_stated(sample):
    tb = pandas.read_csv(d.format(sample, 'stated'), index_col=0)
    tb = tb.loc[tb.index != targets[sample], :]
    return tb


def compute_valued_themselves(mx, name, grid):
    na = (~pandas.isna(mx)).sum().sum()
    ns = []
    rs = []
    for j in range(len(grid) - 1):
        mask = (grid[j] < mx) * (mx <= grid[j+1])
        n = mask.sum().sum()
        r = n / na
        ns.append(n)
        rs.append(r)
    result = {'{0}_n'.format(name): ns, '{0}_r'.format(name): rs}
    return result


def compute_arr_valued_themselves(array, name, grid):
    na = array.shape[0]
    ns = []
    rs = []
    for j in range(len(grid) - 1):
        mask = (grid[j] < array) * (array <= grid[j+1])
        n = mask.sum()
        r = n / na
        ns.append(n)
        rs.append(r)
    result = {'{0}_n'.format(name): ns, '{0}_r'.format(name): rs}
    return result


def compute_all_valued_themselves(sample):
    grid = [-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    stated_results = {}
    stated = get_stated(sample=sample)
    stated = stated.drop(columns=['vif'])
    for column in stated.columns.values:
        stated_results = {**stated_results, **compute_arr_valued_themselves(array=stated[column].values, name=column, grid=grid)}

    mx_pearson_direct = get_corr(sample=sample, measure='pearson', direct='direct')
    v_pearson_direct = compute_valued_themselves(mx=mx_pearson_direct, name='pearson_direct', grid=grid)

    mx_pearson_partial = get_corr(sample=sample, measure='pearson', direct='partial')
    v_pearson_partial = compute_valued_themselves(mx=mx_pearson_partial, name='pearson_partial', grid=grid)

    mx_kendall_direct = get_corr(sample=sample, measure='kendall', direct='direct')
    v_kendall_direct = compute_valued_themselves(mx=mx_kendall_direct, name='kendall_direct', grid=grid)

    mx_mutual_info_direct = get_corr(sample=sample, measure='mutual_info', direct='direct')
    v_mutual_info_direct = compute_valued_themselves(mx=mx_mutual_info_direct, name='mutual_info_direct', grid=grid)

    # mx_mutual_info_partial = get_corr(sample=sample, measure='mutual_info', direct='partial')
    # v_mutual_info_partial = compute_valued_themselves(mx=mx_mutual_info_partial, name='mutual_info_partial', grid=grid)

    results = pandas.DataFrame(
        data={'g': grid[1:], **stated_results, **v_pearson_direct, **v_pearson_partial, **v_kendall_direct,
              **v_mutual_info_direct} # , **v_mutual_info_partial}
    )
    return results


def compute_all_stated(sample):
    percentiles = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]

    stockpiled = {}

    stated = get_stated(sample=sample)
    for col in stated.columns:
        stockpiled[col] = stated[col].values

    results_vl = pandas.DataFrame(data=stockpiled).describe(percentiles=percentiles)

    stockpiled = {}

    mx_pearson_direct = get_corr(sample=sample, measure='pearson', direct='direct')
    stockpiled['pearson_direct'] = mx_pearson_direct.values.flatten()

    mx_pearson_partial = get_corr(sample=sample, measure='pearson', direct='partial')
    stockpiled['pearson_partial'] = mx_pearson_partial.values.flatten()

    mx_kendall_direct = get_corr(sample=sample, measure='kendall', direct='direct')
    stockpiled['kendall_direct'] = mx_kendall_direct.values.flatten()

    mx_mutual_info_direct = get_corr(sample=sample, measure='mutual_info', direct='direct')
    stockpiled['mutual_info_direct'] = mx_mutual_info_direct.values.flatten()

    # mx_mutual_info_partial = get_corr(sample=sample, measure='mutual_info', direct='partial')
    # stockpiled['mutual_info_partial'] = mx_mutual_info_partial.values.flatten()

    results_mx = pandas.DataFrame(data=stockpiled).describe(percentiles=percentiles)

    results = results_vl.merge(right=results_mx, left_index=True, right_index=True)
    return results


def performances_valued_themselves(array, name, grid):
    na = array.shape[0]
    ns = []
    rs = []
    for j in range(len(grid) - 1):
        mask = (grid[j] < array) * (array <= grid[j+1])
        n = mask.sum()
        r = n / na
        ns.append(n)
        rs.append(r)
    result = {'{0}_n'.format(name): ns, '{0}_r'.format(name): rs}
    return result


def performances_all_valued_themselves(sample):
    grid = [-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    arr_pearson_direct = get_perf(sample=sample, measure='pearson', direct='direct')
    v_pearson_direct = performances_valued_themselves(array=arr_pearson_direct, name='pearson_direct', grid=grid)

    arr_pearson_partial = get_perf(sample=sample, measure='pearson', direct='partial')
    v_pearson_partial = performances_valued_themselves(array=arr_pearson_partial, name='pearson_partial', grid=grid)

    arr_kendall_direct = get_perf(sample=sample, measure='kendall', direct='direct')
    v_kendall_direct = performances_valued_themselves(array=arr_kendall_direct, name='kendall_direct', grid=grid)

    arr_mutual_info_direct = get_perf(sample=sample, measure='mutual_info', direct='direct')
    v_mutual_info_direct = performances_valued_themselves(array=arr_mutual_info_direct, name='mutual_info_direct', grid=grid)

    # arr_mutual_info_partial = get_perf(sample=sample, measure='mutual_info', direct='partial')
    # v_mutual_info_partial = performances_valued_themselves(array=arr_mutual_info_partial, name='mutual_info_partial', grid=grid)

    results = pandas.DataFrame(
        data={'g': grid[1:], **v_pearson_direct, **v_pearson_partial, **v_kendall_direct,
              **v_mutual_info_direct} # , **v_mutual_info_partial}
    )
    return results


def stated_over_performance(mx, array, name, grid):
    mns = []
    mes = []
    for j in range(len(grid) - 1):
        mask = (grid[j] < array) * (array <= grid[j+1])
        if mask.sum() > 0:
            mn = mx.loc[:, mask].mean().values[0]
            me = mx.loc[:, mask].median().values[0]
        else:
            mn = numpy.nan
            me = numpy.nan
        mns.append(mn)
        mes.append(me)
    result = {'{0}_mean'.format(name): mns, '{0}_median'.format(name): mes}
    return result


def all_stated_over_performances(sample):
    grid = [-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    arr_performances = get_perf(sample=sample, measure='kendall', direct='direct')

    mx_pearson_direct = get_corr(sample=sample, measure='pearson', direct='direct')
    v_pearson_direct = stated_over_performance(mx=mx_pearson_direct, array=arr_performances, name='pearson_direct', grid=grid)

    mx_pearson_partial = get_corr(sample=sample, measure='pearson', direct='partial')
    v_pearson_partial = stated_over_performance(mx=mx_pearson_partial, array=arr_performances, name='pearson_partial', grid=grid)

    mx_kendall_direct = get_corr(sample=sample, measure='kendall', direct='direct')
    v_kendall_direct = stated_over_performance(mx=mx_kendall_direct, array=arr_performances, name='kendall_direct', grid=grid)

    mx_mutual_info_direct = get_corr(sample=sample, measure='mutual_info', direct='direct')
    v_mutual_info_direct = stated_over_performance(mx=mx_mutual_info_direct, array=arr_performances, name='mutual_info_direct', grid=grid)

    # mx_mutual_info_partial = get_corr(sample=sample, measure='mutual_info', direct='partial')
    # v_mutual_info_partial = stated_over_performance(mx=mx_mutual_info_partial, array=arr_performances, name='mutual_info_partial', grid=grid)

    results = pandas.DataFrame(
        data={'g': grid[1:], **v_pearson_direct, **v_pearson_partial, **v_kendall_direct,
              **v_mutual_info_direct} # , **v_mutual_info_partial}
    )
    return results


def get_static_measures(sample):
    tb = pandas.read_csv(d.format(sample, 'static'), index_col=0)
    return tb


def load_static_measures(sample):
    results = get_static_measures(sample=sample)
    return results


# https://github.com/paulbrodersen/entropy_estimators
# https://dit.readthedocs.io/en/latest/measures/multivariate/total_correlation.html
# https://en.wikipedia.org/wiki/Directed_information       # this one for TS!

# test_stats = get_corr(sample='CS1', measure='mutual_info', direct='direct')
# test_stats = get_perf(sample='CS1', measure='kendall', direct='partial')
# test_stats = get_stated(sample='CS1')
# test_stats = compute_all_valued_themselves(sample='CS1')
# test_stats = compute_all_stated(sample='CS1')
# test_stats = performances_all_valued_themselves(sample='CS1')
# test_stats = all_stated_over_performances(sample='CS1')
# test_stats = load_static_measures(sample='CS1')
