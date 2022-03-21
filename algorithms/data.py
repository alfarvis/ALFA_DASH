import os  
import time
import importlib
import base64
import argparse
import io
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State
import pandas as pd
import networkx as nx
import dash_table
from dash.exceptions import PreventUpdate
import utils.dash_reusable_components as drc
from getAlgorithms import getAlgorithms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from DataGuru import DataGuru

def init():
    global df
    df=pd.DataFrame()
    global datalst
    datalst=dict()
    global dataGuru 
    dataGuru= DataGuru()


def getfilter():
        
    df=dataGuru.getDF()
    features=list(df.columns)
    features=[{'label':i,'value':i} for i in features]
    operations=['>','<','=','<=','>=']
    operations=[{'label':i,'value':i} for i in operations]
    return html.Div([
            html.B('Select Features for Filter'),
            drc.NamedDropdown(name="Features",
                                        id="filtfeatures",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=features,
                                        value=None,
                                        multi=False
                                    ),
            drc.NamedDropdown(name="Select operation",
                            id="filtoperation",                                            
                            clearable=True,
                            searchable=True,
                            options=operations,
                            value=None,
                            multi=False
                        ),
            
                html.Div(
                [
                dcc.Input(
                id="inputvalue",
                type='number',
                placeholder="please enter value",
                )
                ]
                + [html.Div(id="out-all-types")]
                ),
        


            html.Button('Filter', id='generate-filtered-data',n_clicks=0)
            ]            
            )

def getTable():
    
    df=dataGuru.getDF()
    return dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i,"deletable": True} for i in df.columns],
        data=df.to_dict('records'),
        editable=True,
        row_deletable=True,
        
        #styling the table
        page_size=15,
        style_table={'overflowX': 'auto'},
        style_data={
        'color': 'black',
        'whiteSpace': 'normal',
        'height': 'auto',
        },
        style_data_conditional=[
        {
        'if': {'row_index': 'even'},
        'backgroundColor': 'rgb(195, 254, 173)',
        },
        {
        'if': {'row_index': 'odd'},
        'backgroundColor': 'rgb(182, 184, 221)',
        }
        ],
        style_header={
        'backgroundColor': 'rgb(155, 124, 94)',
        'color': 'black',
        'fontWeight': 'bold'
        },
        style_cell={
        'height': 'auto',
        'whiteSpace': 'normal'
        }
        )
    
def filter(filtfeatures,filtoperation,inputvalue):
    pass
