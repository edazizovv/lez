#


#
import pandas


#


#
dataset = pandas.read_csv('./dataset.csv')
reported = pandas.read_csv('./reported.csv')
results = pandas.read_csv('./results.csv')

reported = reported.set_index('#')
results = results.set_index('ex')

results['param'] = 'dimredu=' + results['dimredu'] + ' | ' + \
                   'pre=' + results['pre'] + ' | ' + \
                   'lag=' + results['lag'] + ' | ' + \
                   'model=' + results['model'] + ' | ' + \
                   'fs=' + results['fs']

reported_train = reported[reported['sample'] == 'train']
reported_test = reported[reported['sample'] == 'test']
reported_new = reported_train.merge(right=reported_test, left_index=True, right_index=True, how='outer',
                                    suffixes=('__train', '__test'))

joint = reported_new.merge(right=results, left_index=True, right_index=True, how='outer',
                           suffixes=('__P', '__R'))

joint.index.name = 'ex'

joint['dataset'] = 'CS1'
dataset['dataset'] = 'CS1'

joint.to_excel('./joint.xlsx')
dataset.to_excel('./dataset.xlsx', index=False)
