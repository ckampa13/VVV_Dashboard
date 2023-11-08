from dash import Dash, html, dcc, Input, Output, State, ctx
#import plotly.express as px
#import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import numpy as np
import uproot
from copy import deepcopy

# "local" imports
from CONFIG import EFTAnalysisDir
from UTILS import classify_histogram_keys
#from plotting import config_plots, get_label, ticks_in, CMSify_title
#config_plots()
# from EFTAnalysis
import sys
sys.path.append(EFTAnalysisDir+'EFTAnalysisFitting/scripts/')
from DATACARD_DICT import datacard_dict
from CONFIG_VERSIONS import versions_dict, WC_ALL
from MISC_CONFIGS import template_filename_yields
from tools.plotting import config_plots, get_label, ticks_in, CMSify_title
config_plots()

# some format variables
desc_style = {'fontSize': 'large',
              }

# load yields for each channel
uproot_dict = {}
for channel in datacard_dict.keys():
    v = versions_dict[channel]['v']
    version = f'v{v}'
    sname_ch = datacard_dict[channel]['info']['short_name']
    subchs = datacard_dict[channel]['subchannels'].keys()
    for subch in subchs:
        sname_sch = datacard_dict[channel]['subchannels'][subch]['info']['short_name']
        if versions_dict[channel]['lumi'] == '2018':
            sname_sch += '_2018_scaled'
        filedir = EFTAnalysisDir+f'EFTAnalysisFitting/{channel}/{version}/'
        filename_dim6 = filedir+template_filename_yields.substitute(channel=sname_ch, subchannel=sname_sch, purpose='DataCard_Yields', proc='_MultiDimCleaned', version=version, file_type='root')
        filename_dim8 = filedir+template_filename_yields.substitute(channel=sname_ch, subchannel=sname_sch, purpose='DataCard_Yields', proc='_MultiDimCleaned_dim8', version=version, file_type='root')
        newkey = sname_ch + sname_sch
        uproot_dict[newkey] = {'dim6': {'fname': filename_dim6, 'ur': None, 'key_class_df': None}, 'dim8': {'fname': filename_dim8, 'ur': None, 'key_class_df': None}}
        uproot_dict[newkey]['dim6']['ur'] = uproot.open(filename_dim6)
        uproot_dict[newkey]['dim8']['ur'] = uproot.open(filename_dim8)
        keys_dim6 = [k.split(';')[0] for k in uproot_dict[newkey]['dim6']['ur']]
        keys_dim8 = [k.split(';')[0] for k in uproot_dict[newkey]['dim8']['ur']]
        bkgs_dim6, syst_dim6, datas_dim6 = classify_histogram_keys(keys_dim6)
        bkgs_dim8, syst_dim8, datas_dim8 = classify_histogram_keys(keys_dim8)
        uproot_dict[newkey]['dim6']['key_class_df'] = pd.DataFrame({'key': keys_dim6, 'bkg': bkgs_dim6, 'syst': syst_dim6, 'data': datas_dim6})
        uproot_dict[newkey]['dim8']['key_class_df'] = pd.DataFrame({'key': keys_dim8, 'bkg': bkgs_dim8, 'syst': syst_dim8, 'data': datas_dim8})
        # fname = template_filename_yields.substitute()
        # uproot_dict[newkey]['dim6'] = uproot.open()
        # add var of interest once (applies to dim6 and dim8)
        uproot_dict[newkey]['var_of_choice'] = datacard_dict[channel]['info']['variable_of_choice']
# sorted channel list
channels = list(sorted(uproot_dict.keys()))
#print(f'channnels: {channels}')
# for first test, pick a channel and explicitly list bkg
#ur = uproot_dict['0L_2FJ']['dim6']['ur']
#bkgs = ['WW', 'WZ', 'ZZ', 'ttV', 'QCD', 'WJets', 'DY', 'TTbar']
bkgs = uproot_dict['0L_2FJ']['dim6']['key_class_df'].query('(bkg) and (not syst) and (not data)')['key'].values
bkgs = [b.replace('h_', '') for b in bkgs]
#print(f'bkgs: {bkgs}')
systs = uproot_dict['0L_2FJ']['dim6']['key_class_df'].query('(bkg) and (syst) and (not data)')['key'].values
for bkg in bkgs:
    systs = [s.replace(f'h_{bkg}_', '') for s in systs]
for i in ['Up', 'Down']:
    systs = [s.replace(i, '') for s in systs]
#print(f'syst: {systs}')
#var_of_choice = r'$H_T$'
bin_unit = ' [GeV]'

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H1("VVV Yield Plotter", className='mb-2', style={'textAlign':'center'}),

    html.H2("Channel Selection", className='mb-2', style={'textAlign':'center'}),
    dbc.Row([
        dbc.Col([
            html.Label('Channel / Subchannel', style=desc_style),
            dcc.Dropdown(
                id='channel',
                value='0L_2FJ',
                clearable=False,
                options=channels,
            ),
        ], md=6),
        #], width=4, align='center'),
        dbc.Col([
            html.Label('EFT Dimension', style=desc_style),
            dcc.Dropdown(
                id='dim',
                value='dim6',
                clearable=False,
                options=['dim6', 'dim8'],
            ),
        ], md=6),
        #], width=4, align='center')
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Yield Origin:', style=desc_style),
            html.P('', style=desc_style, id='fname'),
        ], md=12)
    ]),

    html.H2("Backgrounds", className='mb-2', style={'textAlign':'center'}),
    html.H3("Individual Background & Systematic", className='mb-2', style={'textAlign':'center'}),
    dbc.Row([
        dbc.Col([
            html.Label('Background Process', style=desc_style),
            # dcc.Checklist(
            #     id='bkg_select_all',
            #     options=[{'label': 'Select All', 'value': 1}],
            #     value=[],
            # ),
            # html.Br(),
            # html.Button(
            #     children='Select All',
            #     id='bkg_select_all',
            #     n_clicks=0,
            # ),
            # html.Button(
            #     children='Select None',
            #     id='bkg_select_none',
            #     n_clicks=0,
            # ),
            dcc.Dropdown(
                id='bkg_ind',
                value='WW',
                clearable=True,
                options=bkgs,
                multi=False,
            ),
        ], md=6),
        #], width=4)
        dbc.Col([
            html.Label('Systematic Uncertainty', style=desc_style),
            # dcc.Checklist(
            #     id='bkg_select_all',
            #     options=[{'label': 'Select All', 'value': 1}],
            #     value=[],
            # ),
            # html.Br(),
            # html.Button(
            #     children='Select All',
            #     id='bkgsyst_select_all',
            #     n_clicks=0,
            # ),
            # html.Button(
            #     children='Select None',
            #     id='bkgsyst_select_none',
            #     n_clicks=0,
            # ),
            dcc.Dropdown(
                id='bkgsyst_ind',
                # value=[],
                # clearable=False,
                # options=systs,
                # multi=True,
                value=None,
                clearable=True,
                options=systs,
                multi=False,
            ),
        ], md=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label('y-axis Scale', style=desc_style),
            dcc.RadioItems(
                id='y_axis_scale_ind',
                options=[{'label': i, 'value': i} for i in ['linear', 'log']],
                value='linear',
                labelStyle={'display': 'inline-block'},
                style=desc_style,
            ),
        ],),
    ]),

    dbc.Row([
        dbc.Col([
            html.Img(id='bkg_hist_mpl_ind'),#, width='100%'),
        ], width=12, align='center')
    ]),

    dbc.Row([
        dbc.Col([
            html.Img(id='bkg_ratio_mpl_ind'),#, width='100%')
        ], width=12, align='center')
    ]),

    # too small to have on the same row
    # dbc.Row([
    #     dbc.Col([
    #         html.Img(id='bkg_hist_mpl', width='100%')
    #     ], md=6),
    #     dbc.Col([
    #         html.Img(id='bkg_ratio_mpl', width='100%')
    #     ], md=6)
    # ]),
    '''
    html.H3("Summed Backgrounds & Systematics Summed in Quadrature", className='mb-2', style={'textAlign':'center'}),
    dbc.Row([
        dbc.Col([
            html.Label('Background Process', style=desc_style),
            # dcc.Checklist(
            #     id='bkg_select_all',
            #     options=[{'label': 'Select All', 'value': 1}],
            #     value=[],
            # ),
            html.Br(),
            html.Button(
                children='Select All',
                id='bkg_select_all',
                n_clicks=0,
            ),
            html.Button(
                children='Select None',
                id='bkg_select_none',
                n_clicks=0,
            ),
            dcc.Dropdown(
                id='bkg',
                value=['WW'],
                clearable=False,
                options=bkgs,
                multi=True,
            ),
        ], md=6),
        #], width=4)
        dbc.Col([
            html.Label('Systematic Uncertainties', style=desc_style),
            # dcc.Checklist(
            #     id='bkg_select_all',
            #     options=[{'label': 'Select All', 'value': 1}],
            #     value=[],
            # ),
            html.Br(),
            html.Button(
                children='Select All',
                id='bkgsyst_select_all',
                n_clicks=0,
            ),
            html.Button(
                children='Select None',
                id='bkgsyst_select_none',
                n_clicks=0,
            ),
            dcc.Dropdown(
                id='bkgsyst',
                # value=[],
                # clearable=False,
                # options=systs,
                # multi=True,
                value=None,
                clearable=True,
                options=systs,
                multi=False,
            ),
        ], md=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label('y-axis Scale', style=desc_style),
            dcc.RadioItems(
                id='y_axis_scale',
                options=[{'label': i, 'value': i} for i in ['linear', 'log']],
                value='linear',
                labelStyle={'display': 'inline-block'},
                style=desc_style,
            ),
        ],),
    ]),

    dbc.Row([
        dbc.Col([
            html.Img(id='bkg_hist_mpl'),#, width='100%'),
        ], width=12, align='center')
    ]),

    dbc.Row([
        dbc.Col([
            html.Img(id='bkg_ratio_mpl'),#, width='100%')
        ], width=12, align='center')
    ]),
    '''

    # too small to have on the same row
    # dbc.Row([
    #     dbc.Col([
    #         html.Img(id='bkg_hist_mpl', width='100%')
    #     ], md=6),
    #     dbc.Col([
    #         html.Img(id='bkg_ratio_mpl', width='100%')
    #     ], md=6)
    # ]),

    # dbc.Row([
    #     dbc.Col([
    #         dcc.Graph(id='bar-graph-plotly', figure={})
    #     ], width=12, md=6),
    #     dbc.Col([
    #         dag.AgGrid(
    #             id='grid',
    #             rowData=df.to_dict("records"),
    #             columnDefs=[{"field": i} for i in df.columns],
    #             columnSize="sizeToFit",
    #         )
    #     ], width=12, md=6),
    # ], className='mt-4'),

])

# callbacks

###### INDIVIDUAL BACKGROUNDS
# available backgrounds dropdown
@app.callback(
    Output('bkg_ind', 'options'),
    Output('bkg_ind', 'value'),
    Output('fname', 'children'),
    Input('channel', 'value'),
    Input('dim', 'value'),
    State('bkg_ind', 'value'),
    #Input('bkg_select_all', 'n_clicks'),
    #Input('bkg_select_none', 'n_clicks'),
)
def update_bkg_ind_dropdown(channel, dim, bkg):
    # here bkg is None, or a string
    bkgs = uproot_dict[channel][dim]['key_class_df'].query('(bkg) and (not syst) and (not data)')['key'].values
    bkgs = [b.replace('h_', '') for b in bkgs]
    print(f'bkgs ind: {bkgs}')
    #bkg_val = None
    if bkg is None:
        bkg_val = None
    else:
        if bkg in bkgs:
            bkg_val = bkg
        else:
            #bkg_val = None
            bkg_val = bkgs[0]
    fname = uproot_dict[channel][dim]['fname']
    return bkgs, bkg_val, fname

# available background systematics dropdown
@app.callback(
    Output('bkgsyst_ind', 'options'),
    Output('bkgsyst_ind', 'value'),
    Input('bkg_ind', 'value'),
    State('bkgsyst_ind', 'value'),
    # Input('bkgsyst_select_all', 'n_clicks'),
    # Input('bkgsyst_select_none', 'n_clicks'),
    State('channel', 'value'),
    State('dim', 'value'),
)
def update_bkgsyst_ind_dropdown(bkg, bkgsyst, channel, dim):
    print(f'bkg ind (for update syst dropdow): {bkg}')
    if bkg is None:
        return None, None
    all_systs = uproot_dict[channel][dim]['key_class_df'].query('(bkg) and (syst) and (not data)')['key'].values
    systs = []
    for s in all_systs:
        has_bkg = False
        #for b in bkg:
        if f"h_{bkg}" in s:
            has_bkg = True
            #break
        if has_bkg:
            systs.append(s)
    # systs = list(systs)
    # print(f'set of systs: {systs}')
    print(f'systs ind (full key): {systs}')
    # update the syst name
    # print('looping over bkgs....')
    # for b in bkg:
        # print(b)
    systs = [s.replace(f'h_{bkg}_', '') for s in systs]
    for i in ['Up', 'Down']:
        systs = [s.replace(i, '') for s in systs]
    systs = list(set(systs))
    print(f'syst ind (cleaned): {systs}')
    if bkgsyst is None:
        syst_val = None
    else:
        if bkgsyst in systs:
            syst_val = bkgsyst
        else:
            #syst_val = None
            if len(systs) > 0:
                syst_val = systs[0]
            else:
                syst_val = None
    print(f'systs ind (just before return): {systs}')
    return systs, syst_val

# background plot
@app.callback(
    Output(component_id='bkg_hist_mpl_ind', component_property='src'),
    Output(component_id='bkg_ratio_mpl_ind', component_property='src'),
    Input('bkg_ind', 'value'),
    Input('bkgsyst_ind', 'value'),
    Input('channel', 'value'),
    Input('dim', 'value'),
    Input('y_axis_scale_ind', 'value'),
)
def plot_bkg_hist_ind(selected_bkg, selected_bkgsyst, channel, dim, yscale):
    # print(selected_bkg)
    # if type(selected_bkg) is str:
    #     selected_bkg = [selected_bkg]
    # print(selected_bkg)
    # if selected_bkgsyst is None:
    # # if selected_bkgsyst == '':
    #     selected_bkgsyst = []
    # else:
    #     selected_bkgsyst = [selected_bkgsyst]
    # print(selected_bkgsyst)
    # update the uproot file to grab from
    ur = uproot_dict[channel][dim]['ur']
    var_of_choice = uproot_dict[channel]['var_of_choice']

    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes((0.2, 0.2, 0.7, 0.65))
    ax = ticks_in(ax)
    CMSify_title(ax, lumi='138', lumi_unit='fb', energy='13 TeV', prelim=True)
    fig_r = plt.figure(figsize=(12, 10))
    ax_r = fig_r.add_axes((0.2, 0.2, 0.7, 0.65))
    ax_r = ticks_in(ax_r)
    CMSify_title(ax_r, lumi='138', lumi_unit='fb', energy='13 TeV', prelim=True)
    #if len(selected_bkg) > 0:
    if not selected_bkg is None:
        bkg = selected_bkg
        hist_lab = f'h_{bkg}'
        n, bins = ur[hist_lab].to_numpy()
        bins_c = bins[:-1] + np.diff(bins)/2.
        n_tot = n
        # for bkg in selected_bkg:
        #     hist_lab = f'h_{bkg}'
        #     n_, _ = ur[hist_lab].to_numpy()
        #     n_tot += n_
        ax.hist(bins_c, bins=bins, weights=n_tot, histtype='step', color='black', label='nominal')
        ax_r.hist(bins_c, bins=bins, weights=n_tot / n_tot, histtype='step', color='black', label='nominal')
    #if len(selected_bkgsyst) > 0:
    if not selected_bkgsyst is None:
        # first grab all systematic keys to see which are to be added
        all_systs = uproot_dict[channel][dim]['key_class_df'].query('(bkg) and (syst) and (not data)')['key'].values
        # keys_Up = []
        # keys_Down = []
        keys_Up = None
        keys_Down = None
        # loop all bkg and systs
        #for b_ in selected_bkg:
        b_ = selected_bkg
        n_bkg_list_Up = []
        n_bkg_list_Down = []
        n_bkg, _ = ur[f'h_{b_}'].to_numpy()
        #for s_ in selected_bkgsyst:
        s_ = selected_bkgsyst
        trialUp = f'h_{b_}_{s_}Up'
        trialDown = f'h_{b_}_{s_}Down'
        if trialUp in all_systs:
            #keys_Up.append(trialUp)
            #n_bkg_list_Up.append(n_bkg)
            keys_Up = trialUp
        if trialDown in all_systs:
            #keys_Down.append(trialDown)
            #n_bkg_list_Down.append(n_bkg)
            keys_Down = trialDown
        # add Up and Down contributions and plot
        #if len(keys_Up) > 0:
        if not keys_Up is None:
            #n_tot_Up = deepcopy(n_tot)
            #syst_Up2 = np.zeros_like(n_tot)
            #for i, tup in enumerate(zip(keys_Up, n_bkg_list_Up)):
            #k, n_bkg = tup
            #n_, _ = ur[k].to_numpy()
            n_tot_Up, _ = ur[keys_Up].to_numpy()
            # if i == 0:
            #     n_tot_Up += n_
            # else:
            #n_tot_Up += (n_ - n_bkg)
            ###syst_Up2 += (n_ - n_bkg)**2
            ax.hist(bins_c, bins=bins, weights=n_tot_Up, histtype='step', color='red', label=r'Up ($+1\sigma$)')
            ax_r.hist(bins_c, bins=bins, weights=n_tot_Up/n_tot, histtype='step', color='red', label=r'Up ($+1\sigma$)')
        if not keys_Down is None:
            #n_tot_Up = deepcopy(n_tot)
            #syst_Up2 = np.zeros_like(n_tot)
            #for i, tup in enumerate(zip(keys_Up, n_bkg_list_Up)):
            #k, n_bkg = tup
            #n_, _ = ur[k].to_numpy()
            n_tot_Down, _ = ur[keys_Down].to_numpy()
            # if i == 0:
            #     n_tot_Up += n_
            # else:
            #n_tot_Up += (n_ - n_bkg)
            ###syst_Up2 += (n_ - n_bkg)**2
            #     ax.hist(bins_c, bins=bins, weights=n_tot_Up, histtype='step', color='red', label=r'Up ($+1\sigma$)')
            # if len(keys_Down) > 0:
            #     n_tot_Down = deepcopy(n_tot)
            #     for i, tup in enumerate(zip(keys_Down, n_bkg_list_Down)):
            #         k, n_bkg = tup
            #         n_, _ = ur[k].to_numpy()
            #         # if i == 0:
            #         #     n_tot_Down += n_
            #         # else:
            #         n_tot_Down += (n_ - n_bkg)
            ax.hist(bins_c, bins=bins, weights=n_tot_Down, histtype='step', color='blue', label=r'Down ($-1\sigma$)')
            ax_r.hist(bins_c, bins=bins, weights=n_tot_Down/n_tot, histtype='step', color='blue', label=r'Down ($-1\sigma$)')
    #if (len(selected_bkg) > 0) or (len(selected_bkgsyst) > 0):
    if (not selected_bkg is None):
        # set xticks only to bin edges?
        for ax_ in [ax, ax_r]:
            ax_.set_xticks(bins)
            ax_.set_xticklabels([f'{b:0.0f}' for b in bins])
            ax_.legend()
    # formatting
    # labels
    for ax_ in [ax, ax_r]:
        ax_.set_yscale(yscale)
        ax_.set_xlabel(var_of_choice+bin_unit)
    ax.set_ylabel('yield')
    ax_r.set_ylabel('yield / nominal')
    #ax.set_title(f'{channel} Backgrounds')
    title_str = f'{channel}\nBkg: {selected_bkg}'
    # add systematics
    if not selected_bkgsyst is None:
        title_str += f'\nSyst: {selected_bkgsyst}'
    fig.suptitle(title_str)
    fig_r.suptitle(title_str)
    # Save it to a temporary buffer.
    fig_returns = []
    for f in [fig, fig_r]:
        buf = BytesIO()
        f.savefig(buf, format="png")
        # Embed the result in the html output.
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_return = f'data:image/png;base64,{fig_data}'
        fig_returns.append(fig_return)
    return fig_returns

###### MULTIPLE BACKGROUNDS
'''
# available backgrounds dropdown
@app.callback(
    Output('bkg', 'options'),
    Output('bkg', 'value'),
    Output('fname', 'children'),
    Input('channel', 'value'),
    Input('dim', 'value'),
    State('bkg', 'value'),
    Input('bkg_select_all', 'n_clicks'),
    Input('bkg_select_none', 'n_clicks'),
)
def update_bkg_dropdown(channel, dim, bkg, all_selected_n, none_selected_n):
    #ur = uproot_dict[channel][dim]['ur']
    bkgs = uproot_dict[channel][dim]['key_class_df'].query('(bkg) and (not syst) and (not data)')['key'].values
    bkgs = [b.replace('h_', '') for b in bkgs]
    bkg_val = []
    # if type(bkg) is str:
    #     bkg = [bkg]
    for bkg_ in bkg:
        if bkg_ in bkgs:
            bkg_val.append(bkg_)
    if len(bkg_val) < 1:
        bkg_val = [bkgs[0]]
    fname = uproot_dict[channel][dim]['fname']
    # if all selected, grab all
    # print(all_selected)
    # if len(all_selected) > 0:
    #     if all_selected[0] == 1:
    #         bkg_val = bkgs
    # toggle all on or off
    if 'bkg_select_all' == ctx.triggered_id:
        bkg_val = bkgs
    if 'bkg_select_none' == ctx.triggered_id:
        bkg_val = []
    return bkgs, bkg_val, fname

# available background systematics dropdown
@app.callback(
    Output('bkgsyst', 'options'),
    Output('bkgsyst', 'value'),
    Input('bkg', 'value'),
    State('bkgsyst', 'value'),
    Input('bkgsyst_select_all', 'n_clicks'),
    Input('bkgsyst_select_none', 'n_clicks'),
    State('channel', 'value'),
    State('dim', 'value'),
)
def update_bkgsyst_dropdown(bkg, bkgsyst, all_selected_n, none_selected_n, channel, dim):
    if len(bkg) < 1:
        return [], []
    all_systs = uproot_dict[channel][dim]['key_class_df'].query('(bkg) and (syst) and (not data)')['key'].values
    systs = []
    for s in all_systs:
        has_bkg = False
        for b in bkg:
            if f"h_{b}" in s:
                has_bkg = True
                break
        if has_bkg:
            systs.append(s)
    # systs = list(systs)
    # print(f'set of systs: {systs}')
    print(f'systs (full key): {systs}')
    # update the syst name
    print('looping over bkgs....')
    for b in bkg:
        print(b)
        systs = [s.replace(f'h_{b}_', '') for s in systs]
    for i in ['Up', 'Down']:
        systs = [s.replace(i, '') for s in systs]
    systs = list(set(systs))
    print(f'syst (cleaned): {systs}')
    syst_val = []
    #for s in bkgsyst:
    for s in [bkgsyst]:
        if s in systs:
            syst_val.append(s)
    if len(syst_val) < 1:
        if len(systs) > 0:
            syst_val = [systs[0]]
        else:
            syst_val = []
    # toggle all on or off
    if 'bkgsyst_select_all' == ctx.triggered_id:
        syst_val = systs
    if 'bkgsyst_select_none' == ctx.triggered_id:
        syst_val = []
    #return systs, syst_val
    if len(syst_val) < 1:
        return systs, None
    else:
        return systs, syst_val[0]

# background plot
@app.callback(
    Output(component_id='bkg_hist_mpl', component_property='src'),
    Output(component_id='bkg_ratio_mpl', component_property='src'),
    Input('bkg', 'value'),
    Input('bkgsyst', 'value'),
    Input('channel', 'value'),
    Input('dim', 'value'),
    Input('y_axis_scale', 'value'),
)
def plot_bkg_hist(selected_bkg, selected_bkgsyst, channel, dim, yscale):
    # print(selected_bkg)
    # if type(selected_bkg) is str:
    #     selected_bkg = [selected_bkg]
    print(selected_bkg)
    if selected_bkgsyst is None:
    # if selected_bkgsyst == '':
        selected_bkgsyst = []
    else:
        selected_bkgsyst = [selected_bkgsyst]
    print(selected_bkgsyst)
    # update the uproot file to grab from
    ur = uproot_dict[channel][dim]['ur']
    var_of_choice = uproot_dict[channel]['var_of_choice']

    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes((0.2, 0.2, 0.7, 0.65))
    ax = ticks_in(ax)
    CMSify_title(ax, lumi='138', lumi_unit='fb', energy='13 TeV', prelim=True)
    fig_r = plt.figure(figsize=(12, 10))
    ax_r = fig_r.add_axes((0.2, 0.2, 0.7, 0.65))
    ax_r = ticks_in(ax_r)
    CMSify_title(ax_r, lumi='138', lumi_unit='fb', energy='13 TeV', prelim=True)
    if len(selected_bkg) > 0:
        bkg0 = selected_bkg[0]
        hist_lab = f'h_{bkg0}'
        n, bins = ur[hist_lab].to_numpy()
        bins_c = bins[:-1] + np.diff(bins)/2.
        n_tot = np.zeros_like(n)
        for bkg in selected_bkg:
            hist_lab = f'h_{bkg}'
            n_, _ = ur[hist_lab].to_numpy()
            n_tot += n_
        ax.hist(bins_c, bins=bins, weights=n_tot, histtype='step', color='black', label='nominal')
    if len(selected_bkgsyst) > 0:
        # first grab all systematic keys to see which are to be added
        all_systs = uproot_dict[channel][dim]['key_class_df'].query('(bkg) and (syst) and (not data)')['key'].values
        keys_Up = []
        keys_Down = []
        # loop all bkg and systs
        for b_ in selected_bkg:
            n_bkg_list_Up = []
            n_bkg_list_Down = []
            n_bkg, _ = ur[f'h_{b_}'].to_numpy()
            for s_ in selected_bkgsyst:
                trialUp = f'h_{b_}_{s_}Up'
                trialDown = f'h_{b_}_{s_}Down'
                if trialUp in all_systs:
                    keys_Up.append(trialUp)
                    n_bkg_list_Up.append(n_bkg)
                if trialDown in all_systs:
                    keys_Down.append(trialDown)
                    n_bkg_list_Down.append(n_bkg)
        # add Up and Down contributions and plot
        if len(keys_Up) > 0:
            n_tot_Up = deepcopy(n_tot)
            #syst_Up2 = np.zeros_like(n_tot)
            for i, tup in enumerate(zip(keys_Up, n_bkg_list_Up)):
                k, n_bkg = tup
                n_, _ = ur[k].to_numpy()
                # if i == 0:
                #     n_tot_Up += n_
                # else:
                n_tot_Up += (n_ - n_bkg)
                ###syst_Up2 += (n_ - n_bkg)**2
            ax.hist(bins_c, bins=bins, weights=n_tot_Up, histtype='step', color='red', label=r'Up ($+1\sigma$)')
        if len(keys_Down) > 0:
            n_tot_Down = deepcopy(n_tot)
            for i, tup in enumerate(zip(keys_Down, n_bkg_list_Down)):
                k, n_bkg = tup
                n_, _ = ur[k].to_numpy()
                # if i == 0:
                #     n_tot_Down += n_
                # else:
                n_tot_Down += (n_ - n_bkg)
            ax.hist(bins_c, bins=bins, weights=n_tot_Down, histtype='step', color='blue', label=r'Down ($-1\sigma$)')
    if (len(selected_bkg) > 0) or (len(selected_bkgsyst) > 0):
        # set xticks only to bin edges?
        ax.set_xticks(bins)
        ax.set_xticklabels([f'{b:0.0f}' for b in bins])
        ax.legend()
    # formatting
    # labels
    ax.set_yscale(yscale)
    ax.set_xlabel(var_of_choice+bin_unit)
    ax.set_ylabel('yield')
    #ax.set_title(f'{channel} Backgrounds')
    title_str = f'{channel} Summed Backgrounds\nBkg: '+', '.join(selected_bkg)
    # add systematics
    #title_str +=
    fig.suptitle(title_str)
    # Save it to a temporary buffer.
    fig_returns = []
    for f in [fig, fig_r]:
        buf = BytesIO()
        f.savefig(buf, format="png")
        # Embed the result in the html output.
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_return = f'data:image/png;base64,{fig_data}'
        fig_returns.append(fig_return)
    return fig_returns
'''

'''
# Create interactivity between dropdown component and graph
@app.callback(
    Output(component_id='bar-graph-matplotlib', component_property='src'),
    Output('bar-graph-plotly', 'figure'),
    Output('grid', 'defaultColDef'),
    Input('category', 'value'),
)
def plot_data(selected_yaxis):

    # Build the matplotlib figure
    fig = plt.figure(figsize=(14, 5))
    plt.bar(df['State'], df[selected_yaxis])
    plt.ylabel(selected_yaxis)
    plt.xticks(rotation=30)

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'

    # Build the Plotly figure
    fig_bar_plotly = px.bar(df, x='State', y=selected_yaxis).update_xaxes(tickangle=330)

    my_cellStyle = {
        "styleConditions": [
            {
                "condition": f"params.colDef.field == '{selected_yaxis}'",
                "style": {"backgroundColor": "#d3d3d3"},
            },
            {   "condition": f"params.colDef.field != '{selected_yaxis}'",
                "style": {"color": "black"}
            },
        ]
    }

    return fig_bar_matplotlib, fig_bar_plotly, {'cellStyle': my_cellStyle}
'''


if __name__ == '__main__':
    #app.run_server(debug=False, port=8002)
    app.run_server(debug=True, port=8002)
