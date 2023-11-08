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
from MISC_CONFIGS import template_filename_yields, dim6_ops, WC_pretty_print_dict
from tools.plotting import config_plots, get_label, ticks_in, CMSify_title
config_plots()

# some format variables
desc_style = {'fontSize': 'large',
              }

desc_style_flag = {'fontSize': 'large',
                   'color': 'red',
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
        # add var of interest once (applies to dim6 and dim8)
        uproot_dict[newkey]['var_of_choice'] = datacard_dict[channel]['info']['variable_of_choice']
# sorted channel list
channels = list(sorted(uproot_dict.keys()))
#print(f'channnels: {channels}')
# for first test, pick a channel and explicitly list bkg
#ur = uproot_dict['0L_2FJ']['dim6']['ur']
#bkgs = ['WW', 'WZ', 'ZZ', 'ttV', 'QCD', 'WJets', 'DY', 'TTbar']
# c0 = '0L_2FJ'
c0 = '2L_SS_1FJ'
bkgs = uproot_dict[c0]['dim6']['key_class_df'].query('(bkg) and (not syst) and (not data)')['key'].values
bkgs = [b.replace('h_', '') for b in bkgs]
#print(f'bkgs: {bkgs}')
systs = uproot_dict[c0]['dim6']['key_class_df'].query('(bkg) and (syst) and (not data)')['key'].values
for bkg in bkgs:
    systs = [s.replace(f'h_{bkg}_', '') for s in systs]
for i in ['Up', 'Down']:
    systs = [s.replace(i, '') for s in systs]
WCs = [k.replace('h_quad_', '') for k in uproot_dict[c0]['dim6']['key_class_df'].query('(not bkg) and (not syst) and (not data)')['key'].values if 'h_quad' in k]
WCs = [WC for WC in WCs if WC in dim6_ops]
WC2s = deepcopy(WCs)
WC2s.remove('cW')
systs_WC = [k for k in uproot_dict[c0]['dim6']['key_class_df'].query('(not bkg) and (syst) and (not data)')['key'].values if ('h_quad' in k) and ('cW' in k)]
for WC in ['cW']:
    systs_WC = [s.replace(f'h_quad_{WC}_', '') for s in systs_WC]
for i in ['Up', 'Down']:
    systs_WC = [s.replace(i, '') for s in systs_WC]
systs_WC = list(set(systs_WC))
print(f'systs_WC (init): {systs_WC}')
#print(f'WCs (init): {WCs}')
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
                value=c0,
                clearable=False,
                options=channels,
            ),
        ], md=6),
        dbc.Col([
            html.Label('EFT Dimension', style=desc_style),
            dcc.Dropdown(
                id='dim',
                value='dim6',
                clearable=False,
                options=['dim6', 'dim8'],
            ),
        ], md=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Yield Origin:', style=desc_style),
            html.P('', style=desc_style, id='fname'),
        ], md=12)
    ]),

    html.H2("Backgrounds", className='mb-2', style={'textAlign':'center'}),
    dbc.Button(
        "Show / Collapse Bkg Plots",
        id="collapse_bkg_button",
        n_clicks=0,
    ),
    dbc.Collapse(
        [
        html.H3("Individual Background & Systematic", className='mb-2', style={'textAlign':'center'}),
        dbc.Row([
            dbc.Col([
                html.Label('Background Process', style=desc_style),
                dcc.Dropdown(
                    id='bkg_ind',
                    value='WW',
                    clearable=True,
                    options=bkgs,
                    multi=False,
                ),
            ], md=6),
            dbc.Col([
                html.Label('Systematic Uncertainty', style=desc_style),
                dcc.Dropdown(
                    id='bkgsyst_ind',
                    value=systs[0],
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
        ],
        id="collapse_bkg",
        #is_open=True,
        is_open=False,
    ),

    html.H2("EFT Signals", className='mb-2', style={'textAlign':'center'}),
    dbc.Button(
        "Show / Collapse EFT Params Plots",
        id="collapse_WC_params_button",
        n_clicks=0,
    ),
    dbc.Collapse(
        [
        html.H3("Parabola Parameters & Systematic", className='mb-2', style={'textAlign':'center'}),
        dbc.Row([
            dbc.Col([
                html.Label('Wilson Coefficient', style=desc_style),
                dcc.Dropdown(
                    id='WC_ind',
                    value='cW',
                    clearable=False,
                    options=WCs,
                    multi=False,
                ),
            ], md=6),
            dbc.Col([
                html.Label('Systematic Uncertainty', style=desc_style),
                dcc.Dropdown(
                    id='WCsyst_ind',
                    value=systs_WC[0],
                    clearable=True,
                    options=systs_WC,
                    multi=False,
                ),
            ], md=6),
        ]),

        dbc.Row([
            dbc.Col([
                html.Label('y-axis Scale', style=desc_style),
                dcc.RadioItems(
                    id='y_axis_scale_ind_WC',
                    options=[{'label': i, 'value': i} for i in ['linear', 'log']],
                    value='linear',
                    labelStyle={'display': 'inline-block'},
                    style=desc_style,
                ),
            ],),
        ]),

        dbc.Row([
            html.Label('SM Systematic Histogram Name:', style=desc_style),
            html.P('', style=desc_style, id='sm_syst_name'),
        ]),

        dbc.Button(
            "Show / Collapse Quad cW",
            id="collapse_quadWC_button",
            n_clicks=0,
        ),
        dbc.Collapse(
            [
            dbc.Row([
                dbc.Col([
                    html.Img(id='quadWC_hist_mpl_ind'),#, width='100%'),
                ], width=12, align='center')
            ]),

            dbc.Row([
                dbc.Col([
                    html.Img(id='quadWC_ratio_mpl_ind'),#, width='100%')
                ], width=12, align='center')
            ]),
            ],
            id='collapse_quadWC',
            is_open=False,
        ),

        html.Br(),
        dbc.Button(
            "Show / Collapse Lin cW",
            id="collapse_linWC_button",
            n_clicks=0,
        ),
        dbc.Collapse(
            [
            dbc.Row([
                dbc.Col([
                    html.Img(id='linWC_hist_mpl_ind'),#, width='100%'),
                ], width=12, align='center')
            ]),

            dbc.Row([
                dbc.Col([
                    html.Img(id='linWC_ratio_mpl_ind'),#, width='100%')
                ], width=12, align='center')
            ]),
            ],
            id='collapse_linWC',
            is_open=False,
        ),

        html.Br(),
        dbc.Button(
            "Show / Collapse SM",
            id="collapse_SM_button",
            n_clicks=0,
        ),
        dbc.Collapse(
            [
            dbc.Row([
                dbc.Col([
                    html.Img(id='SM_hist_mpl_ind'),#, width='100%'),
                ], width=12, align='center')
            ]),

            dbc.Row([
                dbc.Col([
                    html.Img(id='SM_ratio_mpl_ind'),#, width='100%')
                ], width=12, align='center')
            ]),
            ],
            id='collapse_SM',
            is_open=False,
        ),

        html.Br(),
        dbc.Button(
            "Show / Collapse Mixed",
            id="collapse_mixWC_button",
            n_clicks=0,
        ),
        dbc.Collapse(
            [
            dbc.Row([
                dbc.Col([
                    html.Label('Wilson Coefficient 2', style=desc_style),
                    dcc.Dropdown(
                        id='WC2_ind',
                        value='cHq3',
                        clearable=False,
                        options=WC2s,
                        multi=False,
                    ),
                ], md=6),
            ]),

            dbc.Row([
                html.Label('Mixed Systematic Histogram Name & Status:', style=desc_style),
                html.P('', style=desc_style, id='mix_syst_name'),
            ]),

            dbc.Row([
                dbc.Col([
                    html.Img(id='mixWC_hist_mpl_ind'),#, width='100%'),
                ], width=12, align='center')
            ]),

            dbc.Row([
                dbc.Col([
                    html.Img(id='mixWC_ratio_mpl_ind'),#, width='100%')
                ], width=12, align='center')
            ]),
            ],
            id='collapse_mixWC',
            is_open=False,
        ),

        # too small to have on the same row
        # dbc.Row([
        #     dbc.Col([
        #         html.Img(id='bkg_hist_mpl', width='100%')
        #     ], md=6),
        #     dbc.Col([
        #         html.Img(id='bkg_ratio_mpl', width='100%')
        #     ], md=6)
        # ]),
        ],
        id="collapse_WC_params",
        is_open=True,
    ),


])

# callbacks

###### INDIVIDUAL BACKGROUNDS
# collapse bkg
@app.callback(
    Output("collapse_bkg", "is_open"),
    Input("collapse_bkg_button", "n_clicks"),
    State("collapse_bkg", "is_open"),
)
def toggle_collapse_bkg(n, is_open):
    if n:
        return not is_open
    return is_open

# available backgrounds dropdown
@app.callback(
    Output('bkg_ind', 'options'),
    Output('bkg_ind', 'value'),
    Output('fname', 'children'),
    Input('channel', 'value'),
    Input('dim', 'value'),
    State('bkg_ind', 'value'),
)
def update_bkg_ind_dropdown(channel, dim, bkg):
    # here bkg is None, or a string
    bkgs = uproot_dict[channel][dim]['key_class_df'].query('(bkg) and (not syst) and (not data)')['key'].values
    bkgs = [b.replace('h_', '') for b in bkgs]
    #print(f'bkgs ind: {bkgs}')
    if bkg is None:
        #bkg_val = None
        bkg_val = bkgs[0]
    else:
        if bkg in bkgs:
            bkg_val = bkg
        else:
            bkg_val = bkgs[0]
    fname = uproot_dict[channel][dim]['fname']
    return bkgs, bkg_val, fname

# available background systematics dropdown
@app.callback(
    Output('bkgsyst_ind', 'options'),
    Output('bkgsyst_ind', 'value'),
    Input('bkg_ind', 'value'),
    State('bkgsyst_ind', 'value'),
    State('channel', 'value'),
    State('dim', 'value'),
)
def update_bkgsyst_ind_dropdown(bkg, bkgsyst, channel, dim):
    #print(f'bkg ind (for update syst dropdow): {bkg}')
    if bkg is None:
        return None, None
    all_systs = uproot_dict[channel][dim]['key_class_df'].query('(bkg) and (syst) and (not data)')['key'].values
    systs = []
    for s in all_systs:
        has_bkg = False
        if f"h_{bkg}" in s:
            has_bkg = True
        if has_bkg:
            systs.append(s)
    #print(f'systs ind (full key): {systs}')
    # update the syst name
    systs = [s.replace(f'h_{bkg}_', '') for s in systs]
    for i in ['Up', 'Down']:
        systs = [s.replace(i, '') for s in systs]
    systs = list(set(systs))
    #print(f'syst ind (cleaned): {systs}')
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
    #print(f'systs ind (just before return): {systs}')
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
        ax.hist(bins_c, bins=bins, weights=n_tot, histtype='step', color='black', label='nominal')
        ax_r.hist(bins_c, bins=bins, weights=n_tot / n_tot, histtype='step', color='black', label='nominal')
    #if len(selected_bkgsyst) > 0:
    if not selected_bkgsyst is None:
        # first grab all systematic keys to see which are to be added
        all_systs = uproot_dict[channel][dim]['key_class_df'].query('(bkg) and (syst) and (not data)')['key'].values
        keys_Up = None
        keys_Down = None
        b_ = selected_bkg
        s_ = selected_bkgsyst
        trialUp = f'h_{b_}_{s_}Up'
        trialDown = f'h_{b_}_{s_}Down'
        if trialUp in all_systs:
            keys_Up = trialUp
        if trialDown in all_systs:
            keys_Down = trialDown
        # add Up and Down contributions and plot
        if not keys_Up is None:
            n_tot_Up, _ = ur[keys_Up].to_numpy()
            ax.hist(bins_c, bins=bins, weights=n_tot_Up, histtype='step', color='red', label=r'Up ($+1\sigma$)')
            ax_r.hist(bins_c, bins=bins, weights=n_tot_Up/n_tot, histtype='step', color='red', label=r'Up ($+1\sigma$)')
        if not keys_Down is None:
            n_tot_Down, _ = ur[keys_Down].to_numpy()
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

###### SIGNAL PARAMS
# collapse signal
@app.callback(
    Output("collapse_WC_params", "is_open"),
    Input("collapse_WC_params_button", "n_clicks"),
    State("collapse_WC_params", "is_open"),
)
def toggle_collapse_WC_params(n, is_open):
    if n:
        return not is_open
    return is_open

# available WCs dropdown
@app.callback(
    Output('WC_ind', 'options'),
    Output('WC_ind', 'value'),
    Input('channel', 'value'),
    Input('dim', 'value'),
    State('WC_ind', 'value'),
)
def update_WC_ind_dropdown(channel, dim, WC):
    # here WC is None, or a string
    WCs = [k.replace('h_quad_', '') for k in uproot_dict[channel][dim]['key_class_df'].query('(not bkg) and (not syst) and (not data)')['key'].values if 'h_quad' in k]
    # not necessary, it's already split by file
    #WCs = [WC for WC in WCs if WC in dim6_ops]
    if WC is None:
        # WC_val = None
        WC_val = WCs[0]
    else:
        if WC in WCs:
            WC_val = WC
        else:
            WC_val = WCs[0]
    return WCs, WC_val

# available WC systematics dropdown
@app.callback(
    Output('WCsyst_ind', 'options'),
    Output('WCsyst_ind', 'value'),
    Input('WC_ind', 'value'),
    State('WCsyst_ind', 'value'),
    State('channel', 'value'),
    State('dim', 'value'),
)
def update_WCsyst_ind_dropdown(WC, WCsyst, channel, dim):
    #print(f'bkg ind (for update syst dropdow): {bkg}')
    if WC is None:
        return None, None
    all_systs = uproot_dict[channel][dim]['key_class_df'].query('(not bkg) and (syst) and (not data)')['key'].values
    systs = []
    for s in all_systs:
        if (WC in s) and ("h_quad" in s):
            systs.append(s)
    # update the syst name
    systs = [s.replace(f'h_quad_{WC}_', '') for s in systs]
    for i in ['Up', 'Down']:
        systs = [s.replace(i, '') for s in systs]
    systs = list(set(systs))
    if WCsyst is None:
        syst_val = None
    else:
        if WCsyst in systs:
            syst_val = WCsyst
        else:
            #syst_val = None
            if len(systs) > 0:
                syst_val = systs[0]
            else:
                syst_val = None
    #print(f'systs ind (just before return): {systs}')
    return systs, syst_val

# update quad, lin buttons
@app.callback(
    Output("collapse_quadWC_button", "children"),
    Output("collapse_linWC_button", "children"),
    Input("WC_ind", "value")
)
def update_quadWC_linWC_button(WC):
    return f'Show / Collapse Quad {WC}', f'Show / Collapse Lin {WC}'

# collapse quad
@app.callback(
    Output("collapse_quadWC", "is_open"),
    Input("collapse_quadWC_button", "n_clicks"),
    State("collapse_quadWC", "is_open"),
)
def toggle_collapse_quadWC(n, is_open):
    if n:
        return not is_open
    return is_open

# quad plot
@app.callback(
    Output(component_id='quadWC_hist_mpl_ind', component_property='src'),
    Output(component_id='quadWC_ratio_mpl_ind', component_property='src'),
    Input('WC_ind', 'value'),
    Input('WCsyst_ind', 'value'),
    Input('channel', 'value'),
    Input('dim', 'value'),
    Input('y_axis_scale_ind_WC', 'value'),
)
def plot_quadWC_hist_ind(selected_WC, selected_WCsyst, channel, dim, yscale):
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
    if not selected_WC is None:
        WC = selected_WC
        hist_lab = f'h_quad_{WC}'
        n, bins = ur[hist_lab].to_numpy()
        bins_c = bins[:-1] + np.diff(bins)/2.
        n_tot = n
        ax.hist(bins_c, bins=bins, weights=n_tot, histtype='step', color='black', label='nominal')
        ax_r.hist(bins_c, bins=bins, weights=n_tot / n_tot, histtype='step', color='black', label='nominal')
    if not selected_WCsyst is None:
        WC = selected_WC
        hist_lab_Up = f'h_quad_{WC}_{selected_WCsyst}Up'
        hist_lab_Down = f'h_quad_{WC}_{selected_WCsyst}Down'
        n_tot_Up, _ = ur[hist_lab_Up].to_numpy()
        n_tot_Down, _ = ur[hist_lab_Down].to_numpy()
        # plot
        ax.hist(bins_c, bins=bins, weights=n_tot_Up, histtype='step', color='red', label=r'Up ($+1\sigma$)')
        ax_r.hist(bins_c, bins=bins, weights=n_tot_Up/n_tot, histtype='step', color='red', label=r'Up ($+1\sigma$)')
        ax.hist(bins_c, bins=bins, weights=n_tot_Down, histtype='step', color='blue', label=r'Down ($-1\sigma$)')
        ax_r.hist(bins_c, bins=bins, weights=n_tot_Down/n_tot, histtype='step', color='blue', label=r'Down ($-1\sigma$)')
    if (not selected_WC is None):
        # set xticks only to bin edges?
        for ax_ in [ax, ax_r]:
            ax_.set_xticks(bins)
            ax_.set_xticklabels([f'{b:0.0f}' for b in bins])
            ax_.legend()
        ax.set_ylabel(f'Quad {WC}')
        ax_r.set_ylabel(f'Quad {WC} / Nominal')
    # formatting
    # labels
    for ax_ in [ax, ax_r]:
        ax_.set_yscale(yscale)
        ax_.set_xlabel(var_of_choice+bin_unit)
    title_str = f'{channel}'
    if not selected_WC is None:
        title_str += f'\nWC: {selected_WC}'
    # add systematics
    if not selected_WCsyst is None:
        title_str += f'\nSyst: {selected_WCsyst}'
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

# collapse lin
@app.callback(
    Output("collapse_linWC", "is_open"),
    Input("collapse_linWC_button", "n_clicks"),
    State("collapse_linWC", "is_open"),
)
def toggle_collapse_linWC(n, is_open):
    if n:
        return not is_open
    return is_open

# lin plot
@app.callback(
    Output(component_id='linWC_hist_mpl_ind', component_property='src'),
    Output(component_id='linWC_ratio_mpl_ind', component_property='src'),
    Output('sm_syst_name', 'children'),
    Output('sm_syst_name', 'style'),
    Input('WC_ind', 'value'),
    Input('WCsyst_ind', 'value'),
    Input('channel', 'value'),
    Input('dim', 'value'),
    Input('y_axis_scale_ind_WC', 'value'),
)
def plot_linWC_hist_ind(selected_WC, selected_WCsyst, channel, dim, yscale):
    # update the uproot file to grab from
    ur = uproot_dict[channel][dim]['ur']
    var_of_choice = uproot_dict[channel]['var_of_choice']

    sm_syst_col = ''
    style = desc_style

    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes((0.2, 0.2, 0.7, 0.65))
    ax = ticks_in(ax)
    CMSify_title(ax, lumi='138', lumi_unit='fb', energy='13 TeV', prelim=True)
    fig_r = plt.figure(figsize=(12, 10))
    ax_r = fig_r.add_axes((0.2, 0.2, 0.7, 0.65))
    ax_r = ticks_in(ax_r)
    CMSify_title(ax_r, lumi='138', lumi_unit='fb', energy='13 TeV', prelim=True)
    if not selected_WC is None:
        WC = selected_WC
        n_smlinquad, bins = ur[f'h_sm_lin_quad_{WC}'].to_numpy()
        n_sm, _ = ur['h_sm'].to_numpy()
        n_quad, _ = ur[f'h_quad_{WC}'].to_numpy()
        n_lin = n_smlinquad - n_sm - n_quad
        bins_c = bins[:-1] + np.diff(bins)/2.
        n_tot = n_lin
        ax.hist(bins_c, bins=bins, weights=n_tot, histtype='step', color='black', label='nominal')
        ax_r.hist(bins_c, bins=bins, weights=n_tot / n_tot, histtype='step', color='black', label='nominal')
    if not selected_WCsyst is None:
        WC = selected_WC
        n_tot_Syst = {'Up': None, 'Down': None}
        for d in ['Up', 'Down']:
            n_smlinquad, _ = ur[f'h_sm_lin_quad_{WC}_{selected_WCsyst}{d}'].to_numpy()
            # before fix
            #n_sm, _ = ur[f'h_sm_{WC}_{selected_WCsyst}{d}'].to_numpy()
            # after fix
            # n_sm, _ = ur[f'h_sm_{selected_WCsyst}{d}'].to_numpy()
            try:
                sm_syst_col = f'h_sm_{selected_WCsyst}{d}'
                n_sm, _ = ur[sm_syst_col].to_numpy()
                style = desc_style
            except:
                sm_syst_col = f'h_sm_{WC}_{selected_WCsyst}{d}'
                n_sm, _ = ur[sm_syst_col].to_numpy()
                style = desc_style_flag
            n_quad, _ = ur[f'h_quad_{WC}_{selected_WCsyst}{d}'].to_numpy()
            n_lin = n_smlinquad - n_sm - n_quad
            n_tot_Syst[d] = n_lin
        # plot
        ax.hist(bins_c, bins=bins, weights=n_tot_Syst['Up'], histtype='step', color='red', label=r'Up ($+1\sigma$)')
        ax_r.hist(bins_c, bins=bins, weights=n_tot_Syst['Up']/n_tot, histtype='step', color='red', label=r'Up ($+1\sigma$)')
        ax.hist(bins_c, bins=bins, weights=n_tot_Syst['Down'], histtype='step', color='blue', label=r'Down ($-1\sigma$)')
        ax_r.hist(bins_c, bins=bins, weights=n_tot_Syst['Down']/n_tot, histtype='step', color='blue', label=r'Down ($-1\sigma$)')
    if (not selected_WC is None):
        # set xticks only to bin edges?
        for ax_ in [ax, ax_r]:
            ax_.set_xticks(bins)
            ax_.set_xticklabels([f'{b:0.0f}' for b in bins])
            ax_.legend()
        ax.set_ylabel(f'Lin {WC}')
        ax_r.set_ylabel(f'Lin {WC} / Nominal')
    # formatting
    # labels
    for ax_ in [ax, ax_r]:
        ax_.set_yscale(yscale)
        ax_.set_xlabel(var_of_choice+bin_unit)
    title_str = f'{channel}'
    if not selected_WC is None:
        title_str += f'\nWC: {selected_WC}'
    # add systematics
    if not selected_WCsyst is None:
        title_str += f'\nSyst: {selected_WCsyst}'
    fig.suptitle(title_str)
    fig_r.suptitle(title_str)
    # Save it to a temporary buffer.
    returns = []
    for f in [fig, fig_r]:
        buf = BytesIO()
        f.savefig(buf, format="png")
        # Embed the result in the html output.
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_return = f'data:image/png;base64,{fig_data}'
        returns.append(fig_return)
    returns.append(sm_syst_col)
    returns.append(style)
    return returns

# collapse SM
@app.callback(
    Output("collapse_SM", "is_open"),
    Input("collapse_SM_button", "n_clicks"),
    State("collapse_SM", "is_open"),
)
def toggle_collapse_SM(n, is_open):
    if n:
        return not is_open
    return is_open

# SM plot
@app.callback(
    Output(component_id='SM_hist_mpl_ind', component_property='src'),
    Output(component_id='SM_ratio_mpl_ind', component_property='src'),
    # Output('sm_syst_name', 'children'),
    # Output('sm_syst_name', 'style'),
    Input('WC_ind', 'value'),
    Input('WCsyst_ind', 'value'),
    Input('channel', 'value'),
    Input('dim', 'value'),
    Input('y_axis_scale_ind_WC', 'value'),
)
def plot_smWC_hist_ind(selected_WC, selected_WCsyst, channel, dim, yscale):
    # update the uproot file to grab from
    ur = uproot_dict[channel][dim]['ur']
    var_of_choice = uproot_dict[channel]['var_of_choice']

    # sm_syst_col = ''
    # style = desc_style

    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes((0.2, 0.2, 0.7, 0.65))
    ax = ticks_in(ax)
    CMSify_title(ax, lumi='138', lumi_unit='fb', energy='13 TeV', prelim=True)
    fig_r = plt.figure(figsize=(12, 10))
    ax_r = fig_r.add_axes((0.2, 0.2, 0.7, 0.65))
    ax_r = ticks_in(ax_r)
    CMSify_title(ax_r, lumi='138', lumi_unit='fb', energy='13 TeV', prelim=True)
    if not selected_WC is None:
        WC = selected_WC
        #n_smlinquad, bins = ur[f'h_sm_lin_quad_{WC}'].to_numpy()
        n_sm, bins = ur['h_sm'].to_numpy()
        #n_quad, _ = ur[f'h_quad_{WC}'].to_numpy()
        #n_lin = n_smlinquad - n_sm - n_quad
        bins_c = bins[:-1] + np.diff(bins)/2.
        #n_tot = n_lin
        n_tot = n_sm
        ax.hist(bins_c, bins=bins, weights=n_tot, histtype='step', color='black', label='nominal')
        ax_r.hist(bins_c, bins=bins, weights=n_tot / n_tot, histtype='step', color='black', label='nominal')
    if not selected_WCsyst is None:
        WC = selected_WC
        n_tot_Syst = {'Up': None, 'Down': None}
        for d in ['Up', 'Down']:
            #n_smlinquad, _ = ur[f'h_sm_lin_quad_{WC}_{selected_WCsyst}{d}'].to_numpy()
            # before fix
            #n_sm, _ = ur[f'h_sm_{WC}_{selected_WCsyst}{d}'].to_numpy()
            # after fix
            # n_sm, _ = ur[f'h_sm_{selected_WCsyst}{d}'].to_numpy()
            try:
                sm_syst_col = f'h_sm_{selected_WCsyst}{d}'
                n_sm, _ = ur[sm_syst_col].to_numpy()
                #style = desc_style
            except:
                sm_syst_col = f'h_sm_{WC}_{selected_WCsyst}{d}'
                n_sm, _ = ur[sm_syst_col].to_numpy()
                #style = desc_style_flag
            #n_quad, _ = ur[f'h_quad_{WC}_{selected_WCsyst}{d}'].to_numpy()
            #n_lin = n_smlinquad - n_sm - n_quad
            #n_tot_Syst[d] = n_lin
            n_tot_Syst[d] = n_sm
        # plot
        ax.hist(bins_c, bins=bins, weights=n_tot_Syst['Up'], histtype='step', color='red', label=r'Up ($+1\sigma$)')
        ax_r.hist(bins_c, bins=bins, weights=n_tot_Syst['Up']/n_tot, histtype='step', color='red', label=r'Up ($+1\sigma$)')
        ax.hist(bins_c, bins=bins, weights=n_tot_Syst['Down'], histtype='step', color='blue', label=r'Down ($-1\sigma$)')
        ax_r.hist(bins_c, bins=bins, weights=n_tot_Syst['Down']/n_tot, histtype='step', color='blue', label=r'Down ($-1\sigma$)')
    if (not selected_WC is None):
        # set xticks only to bin edges?
        for ax_ in [ax, ax_r]:
            ax_.set_xticks(bins)
            ax_.set_xticklabels([f'{b:0.0f}' for b in bins])
            ax_.legend()
        ax.set_ylabel('SM')
        ax_r.set_ylabel('SM / Nominal')
    # formatting
    # labels
    for ax_ in [ax, ax_r]:
        ax_.set_yscale(yscale)
        ax_.set_xlabel(var_of_choice+bin_unit)
    title_str = f'{channel} SM'
    # if not selected_WC is None:
    #     title_str += f'\nWC: {selected_WC}'
    # add systematics
    if not selected_WCsyst is None:
        title_str += f'\nSyst: {selected_WCsyst}'
    fig.suptitle(title_str)
    fig_r.suptitle(title_str)
    # Save it to a temporary buffer.
    returns = []
    for f in [fig, fig_r]:
        buf = BytesIO()
        f.savefig(buf, format="png")
        # Embed the result in the html output.
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_return = f'data:image/png;base64,{fig_data}'
        returns.append(fig_return)
    #returns.append(sm_syst_col)
    #returns.append(style)
    return returns

# available WC2s dropdown
@app.callback(
    Output('WC2_ind', 'options'),
    Output('WC2_ind', 'value'),
    Input('WC_ind', 'value'),
    Input('WC_ind', 'options'),
    State('WC2_ind', 'value'),
)
def update_WC2_ind_dropdown(WC, WC_options, WC2):
    # if WC_options is None:
    #     return None
    WC2s = deepcopy(WC_options)
    WC2s.remove(WC)
    if WC2 is None:
        # WC_val = None
        WC2_val = WC2s[0]
    else:
        if WC2 in WC2s:
            WC2_val = WC2
        else:
            WC2_val = WC2s[0]
    return WC2s, WC2_val

# collapse mix
@app.callback(
    Output("collapse_mixWC", "is_open"),
    Input("collapse_mixWC_button", "n_clicks"),
    State("collapse_mixWC", "is_open"),
)
def toggle_collapse_mixWC(n, is_open):
    if n:
        return not is_open
    return is_open

# mixWC plot
@app.callback(
    Output(component_id='mixWC_hist_mpl_ind', component_property='src'),
    Output(component_id='mixWC_ratio_mpl_ind', component_property='src'),
    Output('mix_syst_name', 'children'),
    Output('mix_syst_name', 'style'),
    Input('WC_ind', 'value'),
    Input('WC2_ind', 'value'),
    Input('WCsyst_ind', 'value'),
    Input('channel', 'value'),
    Input('dim', 'value'),
    Input('y_axis_scale_ind_WC', 'value'),
)
def plot_mixWC_hist_ind(selected_WC, selected_WC2, selected_WCsyst, channel, dim, yscale):
    # update the uproot file to grab from
    ur = uproot_dict[channel][dim]['ur']
    var_of_choice = uproot_dict[channel]['var_of_choice']

    mix_syst_col = ''
    syst_exists = ''
    style = desc_style

    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes((0.2, 0.2, 0.7, 0.65))
    ax = ticks_in(ax)
    CMSify_title(ax, lumi='138', lumi_unit='fb', energy='13 TeV', prelim=True)
    fig_r = plt.figure(figsize=(12, 10))
    ax_r = fig_r.add_axes((0.2, 0.2, 0.7, 0.65))
    ax_r = ticks_in(ax_r)
    CMSify_title(ax_r, lumi='138', lumi_unit='fb', energy='13 TeV', prelim=True)
    if not selected_WC is None:
        WC = selected_WC
        WC2 = selected_WC2
        try:
            n_smlinquadmix, bins = ur[f'h_sm_lin_quad_mixed_{WC}_{WC2}'].to_numpy()
        except:
            n_smlinquadmix, bins = ur[f'h_sm_lin_quad_mixed_{WC2}_{WC}'].to_numpy()
        n_smlinquad, _ = ur[f'h_sm_lin_quad_{WC}'].to_numpy()
        n_smlinquad2, _ = ur[f'h_sm_lin_quad_{WC2}'].to_numpy()
        n_sm, bins = ur['h_sm'].to_numpy()
        n_mix = (n_smlinquadmix - (n_smlinquad + n_smlinquad2 - n_sm)) / 2.
        #n_quad, _ = ur[f'h_quad_{WC}'].to_numpy()
        #n_lin = n_smlinquad - n_sm - n_quad
        bins_c = bins[:-1] + np.diff(bins)/2.
        #n_tot = n_lin
        n_tot = n_mix
        ax.hist(bins_c, bins=bins, weights=n_tot, histtype='step', color='black', label='nominal')
        ax_r.hist(bins_c, bins=bins, weights=n_tot / n_tot, histtype='step', color='black', label='nominal')
    if not selected_WCsyst is None:
        WC = selected_WC
        WC2 = selected_WC2
        n_tot_Syst = {'Up': None, 'Down': None}
        for d in ['Up', 'Down']:
            try:
                mix_syst_col = f'h_sm_lin_quad_mixed_{WC}_{WC2}_{selected_WCsyst}{d}'
                syst_exists = ' -- Exists'
                style = desc_style
                plotSyst = True
                n_smlinquadmix, bins = ur[mix_syst_col].to_numpy()
            except:
                try:
                    mix_syst_col = f'h_sm_lin_quad_mixed_{WC2}_{WC}_{selected_WCsyst}{d}'
                    syst_exists = ' -- Exists'
                    style = desc_style
                    plotSyst = True
                    n_smlinquadmix, bins = ur[mix_syst_col].to_numpy()
                except:
                    mix_syst_col = f'h_sm_lin_quad_mixed_{WC}_{WC2}_{selected_WCsyst}{d}'
                    syst_exists = ' -- Does Not Exist'
                    style = desc_style_flag
                    plotSyst = False
                    break
            n_smlinquad, _ = ur[f'h_sm_lin_quad_{WC}_{selected_WCsyst}{d}'].to_numpy()
            n_smlinquad2, _ = ur[f'h_sm_lin_quad_{WC2}_{selected_WCsyst}{d}'].to_numpy()
            try:
                sm_syst_col = f'h_sm_{selected_WCsyst}{d}'
                n_sm, _ = ur[sm_syst_col].to_numpy()
                #style = desc_style
            except:
                sm_syst_col = f'h_sm_{WC}_{selected_WCsyst}{d}'
                n_sm, _ = ur[sm_syst_col].to_numpy()
                #style = desc_style_flag
            n_mix_Syst = (n_smlinquadmix - (n_smlinquad + n_smlinquad2 - n_sm)) / 2.
            n_tot_Syst[d] = n_mix_Syst
        # plot
        if plotSyst:
            ax.hist(bins_c, bins=bins, weights=n_tot_Syst['Up'], histtype='step', color='red', label=r'Up ($+1\sigma$)')
            ax_r.hist(bins_c, bins=bins, weights=n_tot_Syst['Up']/n_tot, histtype='step', color='red', label=r'Up ($+1\sigma$)')
            ax.hist(bins_c, bins=bins, weights=n_tot_Syst['Down'], histtype='step', color='blue', label=r'Down ($-1\sigma$)')
            ax_r.hist(bins_c, bins=bins, weights=n_tot_Syst['Down']/n_tot, histtype='step', color='blue', label=r'Down ($-1\sigma$)')
    if (not selected_WC is None):
        # set xticks only to bin edges?
        for ax_ in [ax, ax_r]:
            ax_.set_xticks(bins)
            ax_.set_xticklabels([f'{b:0.0f}' for b in bins])
            ax_.legend()
        ax.set_ylabel(f'Mixed {WC} {WC2}')
        ax_r.set_ylabel(f'Mixed {WC} {WC2} / Nominal')
    # formatting
    # labels
    for ax_ in [ax, ax_r]:
        ax_.set_yscale(yscale)
        ax_.set_xlabel(var_of_choice+bin_unit)
    title_str = f'{channel}'
    if not selected_WC is None:
        title_str += f'\nWCs: {WC}, {WC2}'
    # add systematics
    if not selected_WCsyst is None:
        title_str += f'\nSyst: {selected_WCsyst}'
    fig.suptitle(title_str)
    fig_r.suptitle(title_str)
    # Save it to a temporary buffer.
    returns = []
    for f in [fig, fig_r]:
        buf = BytesIO()
        f.savefig(buf, format="png")
        # Embed the result in the html output.
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_return = f'data:image/png;base64,{fig_data}'
        returns.append(fig_return)
    returns.append(mix_syst_col + syst_exists)
    returns.append(style)
    return returns


if __name__ == '__main__':
    #app.run_server(debug=False, port=8002)
    app.run_server(debug=True, port=8002)
