# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input, no_update, ctx
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import dash_daq as daq
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import seaborn as sns
from scipy.stats import kendalltau

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from load_cluster import get_clust
from load_stats import compute_all_valued_themselves, compute_all_stated, performances_all_valued_themselves, \
    all_stated_over_performances, load_static_measures

# Incorporate data
# joint = pd.read_excel('./joint.xlsx')
# dataset = pd.read_excel('./dataset.xlsx')
clusters = None
o_stats = None

samples = os.listdir('./hub/')
datasets = {}
datasets_train = {}
datasets_test = {}
joints = []
for sa in samples:
    dataset_x_train = pd.read_csv('./sets/{0}/x_train.csv'.format(sa))
    dataset_x_test = pd.read_csv('./sets/{0}/x_test.csv'.format(sa))
    dataset_y_train = pd.read_csv('./sets/{0}/y_train.csv'.format(sa))
    dataset_y_test = pd.read_csv('./sets/{0}/y_test.csv'.format(sa))

    dataset_train = pd.concat((dataset_x_train, dataset_y_train), axis=1, ignore_index=False)
    dataset_test = pd.concat((dataset_x_test, dataset_y_test), axis=1, ignore_index=False)

    dataset = pd.concat((dataset_train, dataset_test), axis=0, ignore_index=True)

    reported = pd.read_csv('./hub/{0}/reported.csv'.format(sa))
    results = pd.read_csv('./hub/{0}/results.csv'.format(sa))

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

    joint['dataset'] = sa
    dataset['dataset'] = sa
    dataset_train['dataset'] = sa
    dataset_test['dataset'] = sa

    joint = joint.reset_index()

    drop_mask = pd.isna(
        joint[['R-squared measure__train', 'R-squared measure__test', 'SMAPE measure__train', 'SMAPE measure__test']])
    joint = joint[~drop_mask.all(axis=1)].copy()

    datasets[sa] = dataset
    datasets_train[sa] = dataset_train
    datasets_test[sa] = dataset_test
    joints.append(joint)

joint = pd.concat(joints, axis=0, ignore_index=True)

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
    if keepers[sample]:
        return dataset_sub[keepers[sample]]
    else:
        return dataset_sub


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([

    dbc.Row([
        html.Div('Stigma Meta Monitor', className="text-primary text-center fs-3")
    ]),

    dbc.Row([

        dbc.RadioItems(options=[{"label": x, "value": x} for x in samples],
                       value=samples[0],
                       inline=True,
                       id='sample_radio_selector')

    ]),

    dbc.Row([

        daq.NumericInput(
            id='y_hist_nbins',
            value=20,
            max=1000, min=10
        ),
        html.Div(id='y_hist_nbins_div')

    ]),

    dbc.Row([

        dbc.Col([
            dash_table.DataTable(page_size=12, style_table={'overflowX': 'auto'},
                                 id='main_summary_table',
                                 sort_action='native')
        ], width=6),

        dbc.Col([
            dcc.Graph(figure={}, id='target_hist')
        ], width=6),

    ]),

    dbc.Row([

        html.Div([
            dcc.Dropdown(['MAXDEPTH', 'SLEAVES', 'MAXSAMPLES', 'CRITERION', 'MODEL', 'PRESELECT', 'NO'],
                         value='NO', id='scatter_colorize'),
            html.Div(id='scatter_colorize_div')
        ]),

        html.Div([
            dcc.Dropdown(['Anderson-Darling test', 'Augmented Dickey-Fuller test', 'Ljung-Box autocorrelation test',
                          'Ones'],
                         value='Ones', id='scatter_measure'),
            html.Div(id='scatter_measure_div')
        ]),

    ]),

    dbc.Row([

        dbc.Col([

            dcc.Graph(figure={}, id='scatter_measures_r2adj_sample')

        ], width=6),

        dbc.Col([

            dcc.Graph(figure={}, id='scatter_measures_r2adj_all')

        ], width=6),

    ]),

    dbc.Row([

        dbc.Col([

            dcc.Graph(figure={}, id='scatter_measures_smape_sample')

        ], width=6),

        dbc.Col([

            dcc.Graph(figure={}, id='scatter_measures_smape_all')

        ], width=6),

    ]),

    dbc.Row([

        dbc.Col([

            dash_table.DataTable(page_size=12, style_table={'overflowX': 'auto'},
                                 id='static_measures',
                                 sort_action='native')

        ], width=12),

    ]),

    dbc.Row([

        dbc.Col([

            dash_table.DataTable(page_size=12, style_table={'overflowX': 'auto'},
                                 id='correlations_over_values_themselves',
                                 sort_action='native')

        ], width=12),

    ]),

    dbc.Row([

        dbc.Col([

            dash_table.DataTable(page_size=12, style_table={'overflowX': 'auto'},
                                 id='correlations_stats',
                                 sort_action='native')

        ], width=12),

    ]),

    dbc.Row([

        dbc.Col([

            dash_table.DataTable(page_size=12, style_table={'overflowX': 'auto'},
                                 id='performances_over_values_themselves',
                                 sort_action='native')

        ], width=12),

    ]),

    dbc.Row([

        dbc.Col([

            dash_table.DataTable(page_size=12, style_table={'overflowX': 'auto'},
                                 id='performances_stats',
                                 sort_action='native')

        ], width=12),

    ]),

    dbc.Row([

        dbc.Col([

            dash_table.DataTable(page_size=12, style_table={'overflowX': 'auto'},
                                 id='correlations_stats_over_performances',
                                 sort_action='native')

        ], width=12),

    ]),

    dbc.Row([

        html.Div([

            dcc.Dropdown(['agglo_ward', 'kmeans', 'hdbscan'],
                         value='hdbscan', id='cluster_algo'),

            html.Div(id='cluster_algo_div')

        ]),

        html.Div([

            dcc.Dropdown(['2', '3', '4', '5', '6', '7'],
                         value='2', id='cluster_algo_param'),

            html.Div(id='cluster_algo_param_div'),
        ]),

    ]),

    dbc.Row([

        dcc.Graph(figure={}, id="clusters")

    ])

], fluid=True)


@callback(
    Output(component_id='main_summary_table', component_property='data'),
    Input(component_id='sample_radio_selector', component_property='value')
)
def update_main_table(sample):
    data_joint_umt = joint[joint['dataset'] == sample][['ex', 'param', 'model kwg',
                                                        'SMAPE measure__train', 'SMAPE measure__test',
                                                        'R-squared adjusted measure__train',
                                                        'R-squared adjusted measure__test']
    ].to_dict('records')
    return data_joint_umt


@callback(
    Output(component_id='static_measures', component_property='data'),
    Input(component_id='sample_radio_selector', component_property='value')
)
def update_static_measures(sample):
    data_static_measures = load_static_measures(sample=sample).to_dict('records')
    return data_static_measures


@callback(
    Output(component_id='correlations_over_values_themselves', component_property='data'),
    Input(component_id='sample_radio_selector', component_property='value')
)
def update_correlations_over_values_themselves(sample):
    data_correlations_over_values_themselves = compute_all_valued_themselves(sample=sample).to_dict('records')
    return data_correlations_over_values_themselves


@callback(
    Output(component_id='correlations_stats', component_property='data'),
    Input(component_id='sample_radio_selector', component_property='value')
)
def update_correlations_stats(sample):
    data_correlations_stats = compute_all_stated(sample=sample).reset_index().to_dict('records')
    return data_correlations_stats


@callback(
    Output(component_id='performances_over_values_themselves', component_property='data'),
    Input(component_id='sample_radio_selector', component_property='value')
)
def update_performances_over_values_themselves(sample):
    data_performances_over_values_themselves = performances_all_valued_themselves(sample=sample).to_dict('records')
    return data_performances_over_values_themselves


@callback(
    Output(component_id='correlations_stats_over_performances', component_property='data'),
    Input(component_id='sample_radio_selector', component_property='value')
)
def update_correlations_stats_over_performances(sample):
    data_correlations_stats_over_performances = all_stated_over_performances(sample=sample).to_dict('records')
    return data_correlations_stats_over_performances


@callback(
    Output(component_id='target_hist', component_property='figure'),
    Input(component_id='y_hist_nbins', component_property='value'),
    Input(component_id='sample_radio_selector', component_property='value')
)
def update_hist_target(n_bins, sample):
    if n_bins is None:
        n_bins = 20
    data_ds_uht = datasets[sample]
    fig = px.histogram(data_ds_uht[data_ds_uht['dataset'] == sample], x=targets[sample],
                       histnorm='probability density', nbins=n_bins)
    return fig


@callback(
    Output(component_id='scatter_measures_r2adj_sample', component_property='figure'),
    Input(component_id='sample_radio_selector', component_property='value'),
    Input(component_id='scatter_measure', component_property='value'),
    Input(component_id='scatter_colorize', component_property='value')
)
def update_scatter_measures_r2adj_sample(sample, scatter_measure, colorizer):
    jj_usmrs = joint[joint['dataset'] == sample].copy()

    smz = {'smape': {'train': 'SMAPE measure__train',
                     'test': 'SMAPE measure__test'},
           'rmse': {'train': 'R-squared adjusted measure__train',
                    'test': 'R-squared adjusted measure__test'}}

    jj_usmrs['measure_train'] = jj_usmrs[smz['rmse']['train']]
    jj_usmrs['measure_test'] = jj_usmrs[smz['rmse']['test']]
    if colorizer == 'MAXDEPTH':
        cozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 'max' if 'max_depth=100' in x else 'pruned').values
        vozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 1 if 'max_depth=100' in x else 0).values
    elif colorizer == 'SLEAVES':
        cozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 'single' if 'min_samples_leaf=1;' in x else 'some').values
        vozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 1 if 'min_samples_leaf=1;' in x else 0).values
    elif colorizer == 'MAXSAMPLES':
        cozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 'full' if 'max_samples=1.0' in x else 'rate').values
        vozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 1 if 'max_samples=1.0' in x else 0).values
    elif colorizer == 'CRITERION':
        cozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 'sqrt' if 'criterion=squared_error' in x else 'abs').values
        vozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 1 if 'criterion=squared_error' in x else 0).values
    elif colorizer == 'MODEL':
        cozy_usmrs = jj_usmrs['model'].apply(func=lambda x: 'RF' if 'RF' in x else 'ET').values
        vozy_usmrs = jj_usmrs['model'].apply(func=lambda x: 1 if 'RF' in x else 0).values
    elif colorizer == 'PRESELECT':
        cozy_usmrs = jj_usmrs['param'].apply(func=lambda x: 'with pre' if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 'no pre').values
        vozy_usmrs = jj_usmrs['param'].apply(func=lambda x: 1 if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 0).values
    else:
        cozy_usmrs = np.array(['No'] * jj_usmrs.shape[0])
        vozy_usmrs = np.zeros(shape=jj_usmrs.shape[0])

    if scatter_measure != "Ones":
        scaler_usmrs = MinMaxScaler(feature_range=(1, 5))
        sized_usmrs = scaler_usmrs.fit_transform(X=jj_usmrs[scatter_measure + '__test'].values.reshape(-1, 1)).flatten()
        texted = jj_usmrs[scatter_measure + '__test'].round(decimals=4).astype(dtype=str).values
    else:
        sized_usmrs = np.ones(shape=(jj_usmrs.shape[0],)) * 5
        texted = np.array(['Ones'] * jj_usmrs.shape[0])

    trend0_estimator_usmrs = LinearRegression()
    trend0_estimator_usmrs.fit(X=jj_usmrs["measure_train"].values[vozy_usmrs == 0].reshape(-1, 1),
                               y=jj_usmrs["measure_test"].values[vozy_usmrs == 0])
    trend0_x_usmrs = np.linspace(start=jj_usmrs["measure_train"].min(), stop=jj_usmrs["measure_train"].max(),
                                 num=10_000)
    trend0_y_usmrs = trend0_estimator_usmrs.predict(X=trend0_x_usmrs.reshape(-1, 1))

    if 1 in vozy_usmrs:
        trend1_estimator_usmrs = LinearRegression()
        trend1_estimator_usmrs.fit(X=jj_usmrs["measure_train"].values[vozy_usmrs == 1].reshape(-1, 1),
                                   y=jj_usmrs["measure_test"].values[vozy_usmrs == 1])
        trend1_x_usmrs = np.linspace(start=jj_usmrs["measure_train"].min(), stop=jj_usmrs["measure_train"].max(),
                                     num=10_000)
        trend1_y_usmrs = trend1_estimator_usmrs.predict(X=trend1_x_usmrs.reshape(-1, 1))

    fig_usmrs = go.Figure()
    fig_usmrs.add_trace(go.Histogram2dContour(
        x=jj_usmrs["measure_train"].values,
        y=jj_usmrs["measure_test"].values,
        # colorscale='Blues',
        # reversescale=True,
        xaxis='x',
        yaxis='y',
        # contours=dict(coloring='lines')
        line=dict(color='red'),
        contours=dict(coloring='none'),
        # colorscale=None,
    ))
    fig_usmrs.add_trace(go.Scatter(
        x=trend0_x_usmrs,
        y=trend0_y_usmrs,
        line=dict(color='blue'),
        hovertext='intercept={0:.4f} | slope={1:.4f}'.format(
            trend0_estimator_usmrs.intercept_, trend0_estimator_usmrs.coef_[0])
    ))
    if 1 in vozy_usmrs:
        fig_usmrs.add_trace(go.Scatter(
            x=trend1_x_usmrs,
            y=trend1_y_usmrs,
            line=dict(color='orange'),
            hovertext='intercept={0:.4f} | slope={1:.4f}'.format(
                trend1_estimator_usmrs.intercept_, trend1_estimator_usmrs.coef_[0])
        ))
    fig_usmrs.add_trace(go.Scatter(
        x=jj_usmrs["measure_train"].values,
        y=jj_usmrs["measure_test"].values,
        xaxis='x',
        yaxis='y',
        mode='markers',
        line=dict(color='black'),
        marker=dict(
            color=vozy_usmrs.astype(dtype=int),
            size=sized_usmrs,
        ),
        text=np.array([cozy_usmrs[j] + '|' + texted[j]
                       for j in range(cozy_usmrs.shape[0])]),
    ))
    fig_usmrs.update_layout(title='R2 Adj Sample')
    return fig_usmrs


@callback(
    Output(component_id='scatter_measures_r2adj_all', component_property='figure'),
    Input(component_id='sample_radio_selector', component_property='value'),
    Input(component_id='scatter_measure', component_property='value'),
    Input(component_id='scatter_colorize', component_property='value')
)
def update_scatter_measures_r2adj_all(sample, scatter_measure, colorizer):
    jj_usmra = joint.copy()

    smz = {'smape': {'train': 'SMAPE measure__train',
                     'test': 'SMAPE measure__test'},
           'rmse': {'train': 'R-squared adjusted measure__train',
                    'test': 'R-squared adjusted measure__test'}}

    jj_usmra['measure_train'] = jj_usmra[smz['rmse']['train']]
    jj_usmra['measure_test'] = jj_usmra[smz['rmse']['test']]
    if colorizer == 'MAXDEPTH':
        cozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 'max' if 'max_depth=100' in x else 'pruned').values
        vozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 1 if 'max_depth=100' in x else 0).values
    elif colorizer == 'SLEAVES':
        cozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 'single' if 'min_samples_leaf=1;' in x else 'some').values
        vozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 1 if 'min_samples_leaf=1;' in x else 0).values
    elif colorizer == 'MAXSAMPLES':
        cozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 'full' if 'max_samples=1.0' in x else 'rate').values
        vozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 1 if 'max_samples=1.0' in x else 0).values
    elif colorizer == 'CRITERION':
        cozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 'sqrt' if 'criterion=squared_error' in x else 'abs').values
        vozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 1 if 'criterion=squared_error' in x else 0).values
    elif colorizer == 'MODEL':
        cozy_usmra = jj_usmra['model'].apply(func=lambda x: 'RF' if 'RF' in x else 'ET').values
        vozy_usmra = jj_usmra['model'].apply(func=lambda x: 1 if 'RF' in x else 0).values
    elif colorizer == 'PRESELECT':
        cozy_usmra = jj_usmra['param'].apply(func=lambda x: 'with pre' if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 'no pre').values
        vozy_usmra = jj_usmra['param'].apply(func=lambda x: 1 if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 0).values
    else:
        cozy_usmra = np.array(['No'] * jj_usmra.shape[0])
        vozy_usmra = np.zeros(shape=jj_usmra.shape[0])

    if scatter_measure != "Ones":
        scaler_usmra = MinMaxScaler(feature_range=(1, 5))
        sized_usmra = scaler_usmra.fit_transform(X=jj_usmra[scatter_measure + '__test'].values.reshape(-1, 1)).flatten()
        texted = jj_usmra[scatter_measure + '__test'].round(decimals=4).astype(dtype=str).values
    else:
        sized_usmra = np.ones(shape=(jj_usmra.shape[0],)) * 5
        texted = np.array(['Ones'] * jj_usmra.shape[0])

    trend0_estimator_usmra = LinearRegression()
    trend0_estimator_usmra.fit(X=jj_usmra["measure_train"].values[vozy_usmra == 0].reshape(-1, 1),
                               y=jj_usmra["measure_test"].values[vozy_usmra == 0])
    trend0_x_usmra = np.linspace(start=jj_usmra["measure_train"].min(), stop=jj_usmra["measure_train"].max(),
                                 num=10_000)
    trend0_y_usmra = trend0_estimator_usmra.predict(X=trend0_x_usmra.reshape(-1, 1))

    if 1 in vozy_usmra:
        trend1_estimator_usmra = LinearRegression()
        trend1_estimator_usmra.fit(X=jj_usmra["measure_train"].values[vozy_usmra == 1].reshape(-1, 1),
                                   y=jj_usmra["measure_test"].values[vozy_usmra == 1])
        trend1_x_usmra = np.linspace(start=jj_usmra["measure_train"].min(), stop=jj_usmra["measure_train"].max(),
                                     num=10_000)
        trend1_y_usmra = trend1_estimator_usmra.predict(X=trend1_x_usmra.reshape(-1, 1))

    fig_usmra = go.Figure()
    fig_usmra.add_trace(go.Histogram2dContour(
        x=jj_usmra["measure_train"].values[joint['dataset'] == sample],
        y=jj_usmra["measure_test"].values[joint['dataset'] == sample],
        # colorscale='Blues',
        # reversescale=True,
        xaxis='x',
        yaxis='y',
        # contours=dict(coloring='lines')
        line=dict(color='red'),
        contours=dict(coloring='none'),
        # colorscale=None,
    ))
    if (joint['dataset'] != sample).sum() > 0:
        fig_usmra.add_trace(go.Histogram2dContour(
            x=jj_usmra["measure_train"].values[joint['dataset'] != sample],
            y=jj_usmra["measure_test"].values[joint['dataset'] != sample],
            # colorscale='Blues',
            # reversescale=True,
            xaxis='x',
            yaxis='y',
            # contours=dict(coloring='lines')
            line=dict(color='violet'),
            contours=dict(coloring='none'),
            # colorscale=None,
        ))
    fig_usmra.add_trace(go.Scatter(
        x=trend0_x_usmra,
        y=trend0_y_usmra,
        line=dict(color='blue'),
        hovertext='intercept={0:.4f} | slope={1:.4f}'.format(
            trend0_estimator_usmra.intercept_, trend0_estimator_usmra.coef_[0])
    ))
    if 1 in vozy_usmra:
        fig_usmra.add_trace(go.Scatter(
            x=trend1_x_usmra,
            y=trend1_y_usmra,
            line=dict(color='orange'),
            hovertext='intercept={0:.4f} | slope={1:.4f}'.format(
                trend1_estimator_usmra.intercept_, trend1_estimator_usmra.coef_[0])
        ))
    fig_usmra.add_trace(go.Scatter(
        x=jj_usmra["measure_train"].values,
        y=jj_usmra["measure_test"].values,
        xaxis='x',
        yaxis='y',
        mode='markers',
        line=dict(color='black'),
        marker=dict(
            color=vozy_usmra.astype(dtype=int),
            size=sized_usmra,
        ),
        text=np.array([cozy_usmra[j] + '|' + texted[j]
                       for j in range(cozy_usmra.shape[0])]),
    ))
    fig_usmra.update_layout(title='R2 Adj All')
    return fig_usmra


@callback(
    Output(component_id='scatter_measures_smape_sample', component_property='figure'),
    Input(component_id='sample_radio_selector', component_property='value'),
    Input(component_id='scatter_measure', component_property='value'),
    Input(component_id='scatter_colorize', component_property='value')
)
def update_scatter_measures_smape_sample(sample, scatter_measure, colorizer):
    jj_usmss = joint[joint['dataset'] == sample].copy()

    smz = {'smape': {'train': 'SMAPE measure__train',
                     'test': 'SMAPE measure__test'},
           'rmse': {'train': 'R-squared adjusted measure__train',
                    'test': 'R-squared adjusted measure__test'}}

    jj_usmss['measure_train'] = jj_usmss[smz['smape']['train']]
    jj_usmss['measure_test'] = jj_usmss[smz['smape']['test']]
    if colorizer == 'MAXDEPTH':
        cozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 'max' if 'max_depth=100' in x else 'pruned').values
        vozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 1 if 'max_depth=100' in x else 0).values
    elif colorizer == 'SLEAVES':
        cozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 'single' if 'min_samples_leaf=1;' in x else 'some').values
        vozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 1 if 'min_samples_leaf=1;' in x else 0).values
    elif colorizer == 'MAXSAMPLES':
        cozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 'full' if 'max_samples=1.0' in x else 'rate').values
        vozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 1 if 'max_samples=1.0' in x else 0).values
    elif colorizer == 'CRITERION':
        cozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 'sqrt' if 'criterion=squared_error' in x else 'abs').values
        vozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 1 if 'criterion=squared_error' in x else 0).values
    elif colorizer == 'MODEL':
        cozy_usmss = jj_usmss['model'].apply(func=lambda x: 'RF' if 'RF' in x else 'ET').values
        vozy_usmss = jj_usmss['model'].apply(func=lambda x: 1 if 'RF' in x else 0).values
    elif colorizer == 'PRESELECT':
        cozy_usmss = jj_usmss['param'].apply(func=lambda x: 'with pre' if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 'no pre').values
        vozy_usmss = jj_usmss['param'].apply(func=lambda x: 1 if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 0).values
    else:
        cozy_usmss = np.array(['No'] * jj_usmss.shape[0])
        vozy_usmss = np.zeros(shape=jj_usmss.shape[0])

    if scatter_measure != "Ones":
        scaler_usmss = MinMaxScaler(feature_range=(1, 5))
        sized_usmss = scaler_usmss.fit_transform(X=jj_usmss[scatter_measure + '__test'].values.reshape(-1, 1)).flatten()
        texted = jj_usmss[scatter_measure + '__test'].round(decimals=4).astype(dtype=str).values
    else:
        sized_usmss = np.ones(shape=(jj_usmss.shape[0],)) * 5
        texted = np.array(['Ones'] * jj_usmss.shape[0])

    trend0_estimator_usmss = LinearRegression()
    trend0_estimator_usmss.fit(X=jj_usmss["measure_train"].values[vozy_usmss == 0].reshape(-1, 1),
                               y=jj_usmss["measure_test"].values[vozy_usmss == 0])
    trend0_x_usmss = np.linspace(start=jj_usmss["measure_train"].min(), stop=jj_usmss["measure_train"].max(),
                                 num=10_000)
    trend0_y_usmss = trend0_estimator_usmss.predict(X=trend0_x_usmss.reshape(-1, 1))

    if 1 in vozy_usmss:
        trend1_estimator_usmss = LinearRegression()
        trend1_estimator_usmss.fit(X=jj_usmss["measure_train"].values[vozy_usmss == 1].reshape(-1, 1),
                                   y=jj_usmss["measure_test"].values[vozy_usmss == 1])
        trend1_x_usmss = np.linspace(start=jj_usmss["measure_train"].min(), stop=jj_usmss["measure_train"].max(),
                                     num=10_000)
        trend1_y_usmss = trend1_estimator_usmss.predict(X=trend1_x_usmss.reshape(-1, 1))

    fig_usmss = go.Figure()
    fig_usmss.add_trace(go.Histogram2dContour(
        x=jj_usmss["measure_train"].values,
        y=jj_usmss["measure_test"].values,
        # colorscale='Blues',
        # reversescale=True,
        xaxis='x',
        yaxis='y',
        # contours=dict(coloring='lines')
        line=dict(color='red'),
        contours=dict(coloring='none'),
        # colorscale=None,
    ))
    fig_usmss.add_trace(go.Scatter(
        x=trend0_x_usmss,
        y=trend0_y_usmss,
        line=dict(color='blue'),
        hovertext='intercept={0:.4f} | slope={1:.4f}'.format(
            trend0_estimator_usmss.intercept_, trend0_estimator_usmss.coef_[0])
    ))
    if 1 in vozy_usmss:
        fig_usmss.add_trace(go.Scatter(
            x=trend1_x_usmss,
            y=trend1_y_usmss,
            line=dict(color='orange'),
            hovertext='intercept={0:.4f} | slope={1:.4f}'.format(
                trend1_estimator_usmss.intercept_, trend1_estimator_usmss.coef_[0])
        ))
    fig_usmss.add_trace(go.Scatter(
        x=jj_usmss["measure_train"].values,
        y=jj_usmss["measure_test"].values,
        xaxis='x',
        yaxis='y',
        mode='markers',
        line=dict(color='black'),
        marker=dict(
            color=vozy_usmss.astype(dtype=int),
            size=sized_usmss,
        ),
        text=np.array([cozy_usmss[j] + '|' + texted[j]
                       for j in range(cozy_usmss.shape[0])]),
    ))
    fig_usmss.update_layout(title='SMAPE Sample')
    return fig_usmss


@callback(
    Output(component_id='scatter_measures_smape_all', component_property='figure'),
    Input(component_id='sample_radio_selector', component_property='value'),
    Input(component_id='scatter_measure', component_property='value'),
    Input(component_id='scatter_colorize', component_property='value')
)
def update_scatter_measures_smape_all(sample, scatter_measure, colorizer):
    jj_usmsa = joint.copy()

    smz = {'smape': {'train': 'SMAPE measure__train',
                     'test': 'SMAPE measure__test'},
           'rmse': {'train': 'R-squared adjusted measure__train',
                    'test': 'R-squared adjusted measure__test'}}

    jj_usmsa['measure_train'] = jj_usmsa[smz['smape']['train']]
    jj_usmsa['measure_test'] = jj_usmsa[smz['smape']['test']]
    if colorizer == 'MAXDEPTH':
        cozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 'max' if 'max_depth=100' in x else 'pruned').values
        vozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 1 if 'max_depth=100' in x else 0).values
    elif colorizer == 'SLEAVES':
        cozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 'single' if 'min_samples_leaf=1;' in x else 'some').values
        vozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 1 if 'min_samples_leaf=1;' in x else 0).values
    elif colorizer == 'MAXSAMPLES':
        cozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 'full' if 'max_samples=1.0' in x else 'rate').values
        vozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 1 if 'max_samples=1.0' in x else 0).values
    elif colorizer == 'CRITERION':
        cozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 'sqrt' if 'criterion=squared_error' in x else 'abs').values
        vozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 1 if 'criterion=squared_error' in x else 0).values
    elif colorizer == 'MODEL':
        cozy_usmsa = jj_usmsa['model'].apply(func=lambda x: 'RF' if 'RF' in x else 'ET').values
        vozy_usmsa = jj_usmsa['model'].apply(func=lambda x: 1 if 'RF' in x else 0).values
    elif colorizer == 'PRESELECT':
        cozy_usmsa = jj_usmsa['param'].apply(func=lambda x: 'with pre' if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 'no pre').values
        vozy_usmsa = jj_usmsa['param'].apply(func=lambda x: 1 if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 0).values
    else:
        cozy_usmsa = np.array(['No'] * jj_usmsa.shape[0])
        vozy_usmsa = np.zeros(shape=jj_usmsa.shape[0])

    if scatter_measure != "Ones":
        scaler_usmsa = MinMaxScaler(feature_range=(1, 5))
        sized_usmsa = scaler_usmsa.fit_transform(X=jj_usmsa[scatter_measure + '__test'].values.reshape(-1, 1)).flatten()
        texted = jj_usmsa[scatter_measure + '__test'].round(decimals=4).astype(dtype=str).values
    else:
        sized_usmsa = np.ones(shape=(jj_usmsa.shape[0],)) * 5
        texted = np.array(['Ones'] * jj_usmsa.shape[0])

    trend0_estimator_usmsa = LinearRegression()
    trend0_estimator_usmsa.fit(X=jj_usmsa["measure_train"].values[vozy_usmsa == 0].reshape(-1, 1),
                               y=jj_usmsa["measure_test"].values[vozy_usmsa == 0])
    trend0_x_usmsa = np.linspace(start=jj_usmsa["measure_train"].min(), stop=jj_usmsa["measure_train"].max(),
                                 num=10_000)
    trend0_y_usmsa = trend0_estimator_usmsa.predict(X=trend0_x_usmsa.reshape(-1, 1))

    if 1 in vozy_usmsa:
        trend1_estimator_usmsa = LinearRegression()
        trend1_estimator_usmsa.fit(X=jj_usmsa["measure_train"].values[vozy_usmsa == 1].reshape(-1, 1),
                                   y=jj_usmsa["measure_test"].values[vozy_usmsa == 1])
        trend1_x_usmsa = np.linspace(start=jj_usmsa["measure_train"].min(), stop=jj_usmsa["measure_train"].max(),
                                     num=10_000)
        trend1_y_usmsa = trend1_estimator_usmsa.predict(X=trend1_x_usmsa.reshape(-1, 1))

    fig_usmsa = go.Figure()
    fig_usmsa.add_trace(go.Histogram2dContour(
        x=jj_usmsa["measure_train"].values[joint['dataset'] == sample],
        y=jj_usmsa["measure_test"].values[joint['dataset'] == sample],
        # colorscale='Blues',
        # reversescale=True,
        xaxis='x',
        yaxis='y',
        # contours=dict(coloring='lines')
        line=dict(color='red'),
        contours=dict(coloring='none'),
        # colorscale=None,
    ))
    if (joint['dataset'] != sample).sum() > 0:
        fig_usmsa.add_trace(go.Histogram2dContour(
            x=jj_usmsa["measure_train"].values[joint['dataset'] != sample],
            y=jj_usmsa["measure_test"].values[joint['dataset'] != sample],
            # colorscale='Blues',
            # reversescale=True,
            xaxis='x',
            yaxis='y',
            # contours=dict(coloring='lines')
            line=dict(color='violet'),
            contours=dict(coloring='none'),
            # colorscale=None,
        ))
    fig_usmsa.add_trace(go.Scatter(
        x=trend0_x_usmsa,
        y=trend0_y_usmsa,
        line=dict(color='blue'),
        hovertext='intercept={0:.4f} | slope={1:.4f}'.format(
            trend0_estimator_usmsa.intercept_, trend0_estimator_usmsa.coef_[0])
    ))
    if 1 in vozy_usmsa:
        fig_usmsa.add_trace(go.Scatter(
            x=trend1_x_usmsa,
            y=trend1_y_usmsa,
            line=dict(color='orange'),
            hovertext='intercept={0:.4f} | slope={1:.4f}'.format(
                trend1_estimator_usmsa.intercept_, trend1_estimator_usmsa.coef_[0])
        ))
    fig_usmsa.add_trace(go.Scatter(
        x=jj_usmsa["measure_train"].values,
        y=jj_usmsa["measure_test"].values,
        xaxis='x',
        yaxis='y',
        mode='markers',
        line=dict(color='black'),
        marker=dict(
            color=vozy_usmsa.astype(dtype=int),
            size=sized_usmsa,
        ),
        text=np.array([cozy_usmsa[j] + '|' + texted[j]
                       for j in range(cozy_usmsa.shape[0])]),
    ))
    fig_usmsa.update_layout(title='SMAPE All')
    return fig_usmsa


# Add controls to build the interaction
@callback(
    Output(component_id='clusters', component_property='figure'),
    Input(component_id='sample_radio_selector', component_property='value'),
    Input(component_id='cluster_algo', component_property='value'),
    Input(component_id='cluster_algo_param', component_property='value')
)
def update_graph5(sample, cluster_algo, cluster_algo_param):
    cluster_data = get_clust(sample=sample, method=cluster_algo, n_clusters=cluster_algo_param)

    fig = px.scatter(x=cluster_data['x'].values, y=cluster_data['y'].values, color=cluster_data['c'].values)
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
