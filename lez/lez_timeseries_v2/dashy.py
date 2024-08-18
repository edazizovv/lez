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
from load_stats import compute_all_valued_themselves, compute_all_stated, performances_all_valued_themselves, all_stated_over_performances, load_static_measures, get_hist

# Incorporate data
# joint = pd.read_excel('./joint.xlsx')
# dataset = pd.read_excel('./dataset.xlsx')
clusters = None
o_stats = None

samples = os.listdir('./hub/')
joints = []
for sa in samples:

    reported = pd.read_csv('./hub/{0}/reported.csv'.format(sa))
    results = pd.read_csv('./hub/{0}/results.csv'.format(sa))

    reported = reported.set_index('#')
    results = results.set_index('ex')

    results['param'] = 'dimredu=' + results['dimredu'] + ' | ' + \
                       'pre=' + results['pre'] + ' | ' + \
                       'lag=' + results['lag'].astype(dtype=str) + ' | ' + \
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

    joint = joint.reset_index()

    drop_mask = pd.isna(joint[['R-squared measure__train', 'R-squared measure__test', 'SMAPE measure__train', 'SMAPE measure__test']])
    joint = joint[~drop_mask.all(axis=1)].copy()
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
           'TS1': 'y',
           'TS2': 'y',
           'TS3': 'y',
           'TS4': 'y',
           'TS5': 'y',
           'TS6': 'y',
           'TS7': 'y',
           'TS8': 'y'}

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
            dcc.Dropdown(['SMALL', 'WD', 'WEAKDROPS', 'WEAKBATCHES', 'L1LOSS', 'ADAMW', 'PRESELECT', 'NO'],
                         value='NO', id='scatter_colorize'),
            html.Div(id='scatter_colorize_div')
        ]),

        html.Div([
            dcc.Dropdown(['Anderson-Darling test', 'Augmented Dickey-Fuller test', 'Ljung-Box autocorrelation test',
                          'Ones'],
                         value='Anderson-Darling test', id='scatter_measure'),
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
        html.Div([
            dcc.Dropdown(['small_nlags', 'big_nlags'],
                         value='small_nlags', id='n_lags'),
            html.Div(id='n_lags_div')
        ]),
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
                                              'R-squared adjusted measure__train', 'R-squared adjusted measure__test']
                                             ].to_dict('records')
    return data_joint_umt


@callback(
    Output(component_id='static_measures', component_property='data'),
    Input(component_id='sample_radio_selector', component_property='value'),
    Input(component_id='n_lags', component_property='value')
)
def update_static_measures(sample, n_lags):
    data_static_measures = load_static_measures(sample=sample, n_lags=n_lags).to_dict('records')
    return data_static_measures


@callback(
    Output(component_id='correlations_over_values_themselves', component_property='data'),
    Input(component_id='sample_radio_selector', component_property='value'),
    Input(component_id='n_lags', component_property='value')
)
def update_correlations_over_values_themselves(sample, n_lags):
    data_correlations_over_values_themselves = compute_all_valued_themselves(sample=sample, n_lags=n_lags).to_dict('records')
    return data_correlations_over_values_themselves


@callback(
    Output(component_id='correlations_stats', component_property='data'),
    Input(component_id='sample_radio_selector', component_property='value'),
    Input(component_id='n_lags', component_property='value')
)
def update_correlations_stats(sample, n_lags):
    data_correlations_stats = compute_all_stated(sample=sample, n_lags=n_lags).reset_index().to_dict('records')
    return data_correlations_stats


@callback(
    Output(component_id='performances_over_values_themselves', component_property='data'),
    Input(component_id='sample_radio_selector', component_property='value'),
    Input(component_id='n_lags', component_property='value')
)
def update_performances_over_values_themselves(sample, n_lags):
    data_performances_over_values_themselves = performances_all_valued_themselves(sample=sample, n_lags=n_lags).to_dict('records')
    return data_performances_over_values_themselves


@callback(
    Output(component_id='correlations_stats_over_performances', component_property='data'),
    Input(component_id='sample_radio_selector', component_property='value'),
    Input(component_id='n_lags', component_property='value')
)
def update_correlations_stats_over_performances(sample, n_lags):
    data_correlations_stats_over_performances = all_stated_over_performances(sample=sample, n_lags=n_lags).to_dict('records')
    return data_correlations_stats_over_performances


@callback(
    Output(component_id='target_hist', component_property='figure'),
    Input(component_id='y_hist_nbins', component_property='value'),
    Input(component_id='sample_radio_selector', component_property='value')
)
def update_hist_target(n_bins, sample):
    stats = get_hist(sample=sample)
    fig = px.bar(stats, x='bins_middles', y='n')
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
    if colorizer == 'SMALL':
        cozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 'big' if ';L=20_2_1' not in x else 'small').values
    elif colorizer == 'WD':
        cozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 'no wd' if ';wd=' not in x else 'with wd').values
    elif colorizer == 'WEAKDROPS':
        cozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 'drop<0.5'
                                                                if ';DROP=0.5' not in x else 'drop=0.5').values
    elif colorizer == 'WEAKBATCHES':
        cozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 'no batchnorm'
                                                                if 'Batchnorm1D' not in x else 'batchnorm').values
    elif colorizer == 'L1LOSS':
        cozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 'mseloss' if 'L1Loss' not in x else 'l1loss').values
    elif colorizer == 'ADAMW':
        cozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 'adamax' if 'AdamW' not in x else 'adamw').values
    elif colorizer == 'PRESELECT':
        cozy_usmrs = jj_usmrs['param'].apply(func=lambda x: 'with pre' if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 'no pre').values
    else:
        cozy_usmrs = np.array(['No'] * jj_usmrs.shape[0])

    if colorizer == 'SMALL':
        vozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 0 if ';L=20_2_1' not in x else 1).values
    elif colorizer == 'WD':
        vozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 0 if ';wd=' not in x else 1).values
    elif colorizer == 'WEAKDROPS':
        vozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 0 if ';DROP=0.5' not in x else 1).values
    elif colorizer == 'WEAKBATCHES':
        vozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 0 if 'Batchnorm1D' not in x else 1).values
    elif colorizer == 'L1LOSS':
        vozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 0 if 'L1Loss' not in x else 1).values
    elif colorizer == 'ADAMW':
        vozy_usmrs = jj_usmrs['model kwg'].apply(func=lambda x: 0 if 'AdamW' not in x else 1).values
    elif colorizer == 'PRESELECT':
        vozy_usmrs = jj_usmrs['param'].apply(func=lambda x: 0 if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 1).values
    else:
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
    if colorizer == 'SMALL':
        cozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 'big' if ';L=20_2_1' not in x else 'small').values
    elif colorizer == 'WD':
        cozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 'no wd' if ';wd=' not in x else 'with wd').values
    elif colorizer == 'WEAKDROPS':
        cozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 'drop<0.5'
                                                                if ';DROP=0.5' not in x else 'drop=0.5').values
    elif colorizer == 'WEAKBATCHES':
        cozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 'no batchnorm'
                                                                if 'Batchnorm1D' not in x else 'batchnorm').values
    elif colorizer == 'L1LOSS':
        cozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 'mseloss' if 'L1Loss' not in x else 'l1loss').values
    elif colorizer == 'ADAMW':
        cozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 'adamax' if 'AdamW' not in x else 'adamw').values
    elif colorizer == 'PRESELECT':
        cozy_usmra = jj_usmra['param'].apply(func=lambda x: 'with pre' if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 'no pre').values
    else:
        cozy_usmra = np.array(['No'] * jj_usmra.shape[0])

    if colorizer == 'SMALL':
        vozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 0 if ';L=20_2_1' not in x else 1).values
    elif colorizer == 'WD':
        vozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 0 if ';wd=' not in x else 1).values
    elif colorizer == 'WEAKDROPS':
        vozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 0 if ';DROP=0.5' not in x else 1).values
    elif colorizer == 'WEAKBATCHES':
        vozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 0 if 'Batchnorm1D' not in x else 1).values
    elif colorizer == 'L1LOSS':
        vozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 0 if 'L1Loss' not in x else 1).values
    elif colorizer == 'ADAMW':
        vozy_usmra = jj_usmra['model kwg'].apply(func=lambda x: 0 if 'AdamW' not in x else 1).values
    elif colorizer == 'PRESELECT':
        vozy_usmra = jj_usmra['param'].apply(func=lambda x: 0 if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 1).values
    else:
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
    if colorizer == 'SMALL':
        cozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 'big' if ';L=20_2_1' not in x else 'small').values
    elif colorizer == 'WD':
        cozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 'no wd' if ';wd=' not in x else 'with wd').values
    elif colorizer == 'WEAKDROPS':
        cozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 'drop<0.5'
                                                                if ';DROP=0.5' not in x else 'drop=0.5').values
    elif colorizer == 'WEAKBATCHES':
        cozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 'no batchnorm'
                                                                if 'Batchnorm1D' not in x else 'batchnorm').values
    elif colorizer == 'L1LOSS':
        cozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 'mseloss' if 'L1Loss' not in x else 'l1loss').values
    elif colorizer == 'ADAMW':
        cozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 'adamax' if 'AdamW' not in x else 'adamw').values
    elif colorizer == 'PRESELECT':
        cozy_usmss = jj_usmss['param'].apply(func=lambda x: 'with pre' if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 'no pre').values
    else:
        cozy_usmss = np.array(['No'] * jj_usmss.shape[0])

    if colorizer == 'SMALL':
        vozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 0 if ';L=20_2_1' not in x else 1).values
    elif colorizer == 'WD':
        vozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 0 if ';wd=' not in x else 1).values
    elif colorizer == 'WEAKDROPS':
        vozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 0 if ';DROP=0.5' not in x else 1).values
    elif colorizer == 'WEAKBATCHES':
        vozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 0 if 'Batchnorm1D' not in x else 1).values
    elif colorizer == 'L1LOSS':
        vozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 0 if 'L1Loss' not in x else 1).values
    elif colorizer == 'ADAMW':
        vozy_usmss = jj_usmss['model kwg'].apply(func=lambda x: 0 if 'AdamW' not in x else 1).values
    elif colorizer == 'PRESELECT':
        vozy_usmss = jj_usmss['param'].apply(func=lambda x: 0 if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 1).values
    else:
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
    if colorizer == 'SMALL':
        cozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 'big' if ';L=20_2_1' not in x else 'small').values
    elif colorizer == 'WD':
        cozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 'no wd' if ';wd=' not in x else 'with wd').values
    elif colorizer == 'WEAKDROPS':
        cozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 'drop<0.5'
                                                                if ';DROP=0.5' not in x else 'drop=0.5').values
    elif colorizer == 'WEAKBATCHES':
        cozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 'no batchnorm'
                                                                if 'Batchnorm1D' not in x else 'batchnorm').values
    elif colorizer == 'L1LOSS':
        cozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 'mseloss' if 'L1Loss' not in x else 'l1loss').values
    elif colorizer == 'ADAMW':
        cozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 'adamax' if 'AdamW' not in x else 'adamw').values
    elif colorizer == 'PRESELECT':
        cozy_usmsa = jj_usmsa['param'].apply(func=lambda x: 'with pre' if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 'no pre').values
    else:
        cozy_usmsa = np.array(['No'] * jj_usmsa.shape[0])

    if colorizer == 'SMALL':
        vozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 0 if ';L=20_2_1' not in x else 1).values
    elif colorizer == 'WD':
        vozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 0 if ';wd=' not in x else 1).values
    elif colorizer == 'WEAKDROPS':
        vozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 0 if ';DROP=0.5' not in x else 1).values
    elif colorizer == 'WEAKBATCHES':
        vozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 0 if 'Batchnorm1D' not in x else 1).values
    elif colorizer == 'L1LOSS':
        vozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 0 if 'L1Loss' not in x else 1).values
    elif colorizer == 'ADAMW':
        vozy_usmsa = jj_usmsa['model kwg'].apply(func=lambda x: 0 if 'AdamW' not in x else 1).values
    elif colorizer == 'PRESELECT':
        vozy_usmsa = jj_usmsa['param'].apply(func=lambda x: 0 if
        ('dimredu=No' not in x) | ('pre=No' not in x) | ('fs=No' not in x) else 1).values
    else:
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
    Input(component_id='cluster_algo_param', component_property='value'),
    Input(component_id='n_lags', component_property='value')
)
def update_graph5(sample, cluster_algo, cluster_algo_param, n_lags):

    cluster_data = get_clust(sample=sample, method=cluster_algo, n_clusters=cluster_algo_param, n_lags=n_lags)

    fig = px.scatter(x=cluster_data['x'].values, y=cluster_data['y'].values, color=cluster_data['c'].values)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8060)
