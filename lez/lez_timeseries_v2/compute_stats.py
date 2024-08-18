import time
import datetime
import os
import numpy
import pandas
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import mutual_info_regression
from entropy_estimators import continuous
from sklearn.preprocessing import StandardScaler
from scipy import stats

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

n_lags = 'big_nlags'

dd = './precomputed/linkage_1/{0}/'
d = './precomputed/linkage_1/{0}/{1}_{2}.csv'
g = './precomputed/linkage_1/pre/{0}_{1}.csv'

samples = [x for x in os.listdir('./hub/') if x != 'TS6']
datasets = {}
datasets_train = {}
datasets_test = {}
joints = []
for sa in samples:

    dataset_x_train = pandas.read_csv('./sets/{0}/{1}/xx_train.csv'.format(sa, n_lags))
    dataset_x_test = pandas.read_csv('./sets/{0}/{1}/xx_test.csv'.format(sa, n_lags))
    dataset_y_train = pandas.read_csv('./sets/{0}/{1}/yy_train.csv'.format(sa, n_lags))
    dataset_y_test = pandas.read_csv('./sets/{0}/{1}/yy_test.csv'.format(sa, n_lags))

    dataset_train = pandas.concat((dataset_x_train, dataset_y_train), axis=1, ignore_index=False)
    dataset_test = pandas.concat((dataset_x_test, dataset_y_test), axis=1, ignore_index=False)

    dataset = pandas.concat((dataset_train, dataset_test), axis=0, ignore_index=True)

    reported = pandas.read_csv('./hub/{0}/reported.csv'.format(sa))
    results = pandas.read_csv('./hub/{0}/results.csv'.format(sa))

    reported = reported.set_index('#')
    results = results.set_index('ex')

    results['param'] = 'dimredu=' + results['dimredu'] + ' | ' + \
                       'pre=' + results['pre'] + ' | ' + \
                       'lag=' + str(results['lag']) + ' | ' + \
                       'model=' + results['model'] + ' | ' + \
                       'fs=' + results['fs']

    reported_train = reported[reported['sample'] == 'train']
    reported_test = reported[reported['sample'] == 'test']
    reported_new = reported_train.merge(right=reported_test, left_index=True, right_index=True, how='outer',
                                        suffixes=('__train', '__test'))

    joint = reported_new.merge(right=results, left_index=True, right_index=True, how='outer',
                               suffixes=('__P', '__R'))

    joint.index.name = 'ex'

    joint['dataset'] = sa
    dataset['dataset'] = sa
    dataset_train['dataset'] = sa
    dataset_test['dataset'] = sa

    joint = joint.reset_index()
    joint = joint.dropna()

    datasets[sa] = dataset
    datasets_train[sa] = dataset_train
    datasets_test[sa] = dataset_test
    joints.append(joint)

joint = pandas.concat(joints, axis=0, ignore_index=True)

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

sample_name = 'dataset'
excludeds = {'CS1': [targets['CS1'], sample_name, 'No'],
             'CS2': [targets['CS1'], sample_name, 'No'],
             'CS3': [targets['CS1'], sample_name, 'No', 'New_Price'],
             'CS4': [targets['CS1'], sample_name, 'No',
                     'BLOCK', 'LOT', 'NEIGHBORHOOD', 'EASE-MENT', 'ADDRESS', 'APARTMENT NUMBER', 'ZIP CODE',
                     'TAX CLASS AT PRESENT', 'BUILDING CLASS AT PRESENT', 'BUILDING CLASS AT TIME OF SALE',
                     'SALE DATE'],
             'CS5': [targets['CS1'], sample_name, 'No',
                     'hoa (R$)', 'property tax (R$)', 'fire insurance (R$)', 'total (R$)'],
             'CS6': [targets['CS1'], sample_name, 'No', 'BHK_OR_RK'],
             'CS7': [targets['CS1'], sample_name, 'No'],
             'CS8': [targets['CS1'], sample_name, 'No'],
             'TS1': [targets['CS1'], sample_name, 'No'],
             'TS2': [targets['CS1'], sample_name, 'No'],
             'TS3': [targets['CS1'], sample_name, 'No'],
             'TS4': [targets['CS1'], sample_name, 'No'],
             'TS5': [targets['CS1'], sample_name, 'No'],
             'TS6': [targets['CS1'], sample_name, 'No'],
             'TS7': [targets['CS1'], sample_name, 'No'],
             'TS8': [targets['CS1'], sample_name, 'No', 'Daily Summary']}


keepers = {'CS1': None,
           'CS2': None,
           'CS3': None,
           'CS4': None,
           'CS5': None,
           'CS6': None,
           'CS7': None,
           'CS8': None,
           'TS1': None,
           'TS2': ['虚血性心疾患_ja.wikipedia.org_all-access_all-agents',
                   'Allerheiligen_de.wikipedia.org_all-access_spider',
                   'Rowan_Atkinson_de.wikipedia.org_mobile-web_all-agents',
                   'Category:Sexual_fetishism_commons.wikimedia.org_desktop_all-agents',
                   'File:Penile-vaginal_sexual_act.JPG_commons.wikimedia.org_mobile-web_all-agents',
                   'Liste_des_personnages_de_Star_Wars_fr.wikipedia.org_all-access_all-agents',
                   'Football_américain_fr.wikipedia.org_all-access_spider',
                   'Álgebra_es.wikipedia.org_desktop_all-agents',
                   '420_(cannabis_culture)_en.wikipedia.org_desktop_all-agents',
                   'Milo_Ventimiglia_en.wikipedia.org_desktop_all-agents',
                   'Dióxido_de_carbono_es.wikipedia.org_mobile-web_all-agents',
                   '花より男子の登場人物_ja.wikipedia.org_mobile-web_all-agents',
                   '美少女戰士_zh.wikipedia.org_desktop_all-agents',
                   'День_сурка_(фильм)_ru.wikipedia.org_desktop_all-agents',
                   '建国記念の日_ja.wikipedia.org_desktop_all-agents',
                   "Grades_de_l'Armée_française_fr.wikipedia.org_mobile-web_all-agents",
                   'El_lobo_de_Wall_Street_es.wikipedia.org_mobile-web_all-agents',
                   'Bradley_Cooper_es.wikipedia.org_all-access_spider',
                   'Ophélie_Meunier_fr.wikipedia.org_all-access_all-agents',
                   'ジョシュア_(ファッションモデル)_ja.wikipedia.org_desktop_all-agents',
                   'Base_de_datos_es.wikipedia.org_all-access_spider',
                   'Medaillenspiegel_der_Olympischen_Sommerspiele_1988_de.wikipedia.org_desktop_all-agents',
                   '爱丁堡公爵菲利普亲王_zh.wikipedia.org_mobile-web_all-agents',
                   '選抜高等学校野球大会_ja.wikipedia.org_all-access_all-agents',
                   'Ashley_Graham_(model)_en.wikipedia.org_desktop_all-agents',
                   'Der_Pferdeflüsterer_de.wikipedia.org_all-access_all-agents',
                   '徐自強案_zh.wikipedia.org_all-access_all-agents',
                   'Cristo_Redentor_es.wikipedia.org_mobile-web_all-agents',
                   'Wikipedia:Beteiligen_de.wikipedia.org_desktop_all-agents',
                   'Scorpiones_es.wikipedia.org_mobile-web_all-agents',
                   'NARUTO_-ナルト-_ja.wikipedia.org_mobile-web_all-agents',
                   'Handball-Europameisterschaft_2016_de.wikipedia.org_mobile-web_all-agents',
                   'Sophie_Dorothea_von_Braunschweig-Lüneburg_de.wikipedia.org_desktop_all-agents',
                   'The_Office_(U.S._TV_series)_en.wikipedia.org_mobile-web_all-agents',
                   '辣妈正传_zh.wikipedia.org_all-access_all-agents',
                   'User:Christian_Ferrer_commons.wikimedia.org_desktop_all-agents',
                   'Antiquité_fr.wikipedia.org_all-access_all-agents',
                   'ルウト_ja.wikipedia.org_all-access_spider',
                   'フランス_ja.wikipedia.org_desktop_all-agents',
                   'レキシ_ja.wikipedia.org_all-access_spider',
                   'Documentation_www.mediawiki.org_all-access_all-agents',
                   'Во_все_тяжкие_ru.wikipedia.org_mobile-web_all-agents',
                   'セフィロス_ja.wikipedia.org_mobile-web_all-agents',
                   '関ジャニ∞_ja.wikipedia.org_all-access_spider',
                   'Савченко,_Надежда_Викторовна_ru.wikipedia.org_desktop_all-agents',
                   'España_en_los_Juegos_Olímpicos_de_Barcelona_1992_es.wikipedia.org_all-access_spider',
                   'Christina_Ricci_en.wikipedia.org_mobile-web_all-agents',
                   'Снаткина,_Анна_Алексеевна_ru.wikipedia.org_mobile-web_all-agents',
                   'FBI_(名探偵コナン)_ja.wikipedia.org_all-access_spider',
                   'МиГ-35_ru.wikipedia.org_mobile-web_all-agents',
                   'Category:Anal_balls_commons.wikimedia.org_all-access_spider',
                   'NRG_Stadium_de.wikipedia.org_all-access_spider',
                   'Березовский,_Борис_Абрамович_ru.wikipedia.org_all-access_all-agents',
                   'Хокинг,_Стивен_ru.wikipedia.org_desktop_all-agents',
                   'Берлинская_стена_ru.wikipedia.org_all-access_all-agents',
                   'Francisco_de_Miranda_es.wikipedia.org_all-access_spider',
                   'Spam_fr.wikipedia.org_desktop_all-agents',
                   '新垣渚_ja.wikipedia.org_mobile-web_all-agents',
                   '黎明_zh.wikipedia.org_all-access_all-agents',
                   'Category:Tia_Ling_commons.wikimedia.org_all-access_all-agents',
                   'Olympische_Sommerspiele_2012_de.wikipedia.org_all-access_all-agents',
                   'Football_League_Championship_en.wikipedia.org_desktop_all-agents',
                   'Eine_wie_keine_(Film)_de.wikipedia.org_mobile-web_all-agents',
                   'David_Rachline_fr.wikipedia.org_desktop_all-agents',
                   'Bratwurst_de.wikipedia.org_mobile-web_all-agents',
                   'さかなクン_ja.wikipedia.org_all-access_all-agents',
                   'WikiLeaks_de.wikipedia.org_all-access_all-agents',
                   'Ататюрк,_Мустафа_Кемаль_ru.wikipedia.org_mobile-web_all-agents',
                   'Deutsche_Demokratische_Republik_de.wikipedia.org_all-access_spider',
                   'Monica_Lewinsky_fr.wikipedia.org_all-access_all-agents',
                   'Software_es.wikipedia.org_mobile-web_all-agents',
                   '頑童MJ116_zh.wikipedia.org_desktop_all-agents',
                   'Category:Nude_recumbent_women_(supine)_commons.wikimedia.org_all-access_spider',
                   'Месхи,_Гела_ru.wikipedia.org_all-access_spider',
                   '安宰賢_zh.wikipedia.org_desktop_all-agents',
                   'Tundra_es.wikipedia.org_all-access_spider',
                   'Коллегия_выборщиков_США_ru.wikipedia.org_desktop_all-agents',
                   'Лоуренс,_Дженнифер_ru.wikipedia.org_desktop_all-agents',
                   'トヨタ自動車_ja.wikipedia.org_all-access_spider',
                   'Absinth_de.wikipedia.org_mobile-web_all-agents',
                   'ちびまる子ちゃんの登場人物_ja.wikipedia.org_all-access_spider',
                   'MLEB_www.mediawiki.org_desktop_all-agents',
                   '田中淳子_ja.wikipedia.org_all-access_all-agents',
                   '劉德華_zh.wikipedia.org_all-access_spider',
                   'Friends_fr.wikipedia.org_all-access_all-agents',
                   'Olympische_Ringe_de.wikipedia.org_all-access_spider',
                   'Кунг-фу_панда_3_ru.wikipedia.org_all-access_all-agents',
                   '松井珠理奈_ja.wikipedia.org_mobile-web_all-agents',
                   'Abou_Bakr_al-Baghdadi_fr.wikipedia.org_mobile-web_all-agents',
                   '幽☆遊☆白書_(テレビアニメ)_ja.wikipedia.org_all-access_spider',
                   '沖縄戦_ja.wikipedia.org_desktop_all-agents',
                   'アルバス・ダンブルドア_ja.wikipedia.org_mobile-web_all-agents',
                   'Aly_Raisman_en.wikipedia.org_all-access_all-agents',
                   'Sommerzeit_de.wikipedia.org_all-access_all-agents',
                   'Vitesse_de_la_lumière_fr.wikipedia.org_desktop_all-agents',
                   'Gro_Swantje_Kohlhof_de.wikipedia.org_all-access_spider',
                   'Ходченкова,_Светлана_Викторовна_ru.wikipedia.org_all-access_all-agents',
                   'Vanessa_Paradis_en.wikipedia.org_all-access_spider',
                   'Robert_Louis_Stevenson_fr.wikipedia.org_all-access_all-agents',
                   '徳川埋蔵金_ja.wikipedia.org_all-access_spider'],
           'TS3': None,
           'TS4': None,
           'TS5': None,
           'TS6': None,
           'TS7': None,
           'TS8': None}


def through_keepers(dataset_sub, sample):
    return dataset_sub


#Interquartile range
def iqr(arr, l, p_low=25, p_upp=75):
    q3, q1 = numpy.percentile(arr, [p_upp,p_low])
    if l is None:
        return q3, q1
    else:
        iqr_value = q3 - q1
        upper_out = iqr_value * l + q3
        lower_out = q1 - (iqr_value * l)
        return upper_out, lower_out


def outliers_stats(data, l, sample):
    out_threshs = {col: iqr(arr=data[col].values, l=l) for col in through_keepers(data.columns, sample)
                   if col not in excludeds[sample]}
    data_ = through_keepers(data[[col for col in data.columns if col not in excludeds[sample]]], sample)
    result = [((data_[col] >= out_threshs[col][0]) | (data_[col] <= out_threshs[col][1])).sum() / data_.shape[0]
              for col in data_.columns]
    return result


total_time = time.time()
print('Computations started: {0}\n'.format(datetime.datetime.now().isoformat()))
for sample in samples:

    step_time = time.time()

    print('running {0}'.format(sample))
    print('started at {0}'.format(datetime.datetime.now().isoformat()))

    print('with size of {0:.2f} MB'.format(datasets[sample].memory_usage(index=True, deep=True).sum() / 2 ** 20))

    data = through_keepers(
        datasets[sample][datasets[sample]['dataset'] == sample][[x for x in datasets[sample].columns if x not in excludeds[sample]]],
        sample)

    data.to_csv(g.format(sample, n_lags), index=False)

    target = targets[sample]

    if sample != 'TS6':
        xx = add_constant(data)
        stated = pandas.DataFrame(data={'vif': [variance_inflation_factor(xx.values, i)
                            for i in range(xx.shape[1])]},
                           index=xx.columns)
        stated = stated[stated.index != 'const']
        stated['outliers_1.5'] = outliers_stats(data, 1.5, sample)
        stated['outliers_3.0'] = outliers_stats(data, 3.0, sample)

        if not os.path.isdir(dd.format(sample)):
            os.mkdir(dd.format(sample))
        stated.to_csv(d.format(sample, 'stated', n_lags))

        print('finished {0}'.format(sample))
        print('finished at {0}'.format(datetime.datetime.now().isoformat()))
        print('time spent for sample: {0}'.format(time.time() - step_time))
        print('total time spent: {0}'.format(time.time() - total_time))
        print()
    else:

        stateds = []
        for j in range(10):
            print(j)
            data_ = data.sample(frac=0.1)

            xx = add_constant(data_)
            stated = pandas.DataFrame(data={'vif': [variance_inflation_factor(xx.values, i)
                                                    for i in range(xx.shape[1])]},
                                      index=xx.columns)
            stated = stated[stated.index != 'const']
            stated['outliers_1.5'] = outliers_stats(data_, 1.5, sample)
            stated['outliers_3.0'] = outliers_stats(data_, 3.0, sample)

            stateds.append(stated)

        stated = sum(stateds) / 10

        if not os.path.isdir(dd.format(sample)):
            os.mkdir(dd.format(sample))
        stated.to_csv(d.format(sample, 'stated', n_lags))

        print('finished {0}'.format(sample))
        print('finished at {0}'.format(datetime.datetime.now().isoformat()))
        print('time spent for sample: {0}'.format(time.time() - step_time))
        print('total time spent: {0}'.format(time.time() - total_time))
        print()
