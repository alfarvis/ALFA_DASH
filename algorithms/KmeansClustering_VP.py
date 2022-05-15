import dash
import dash_table
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import networkx as nx
import utils.dash_reusable_components as drc

from DataGuru import DataGuru

class KmeansClustering_VP:

    @staticmethod

    def getAlgoProps(options,colorscales,globalData):        

        df=globalData.dataGuru.getDF()
        options=df.columns
        options=[{'label':i,'value':i} for i in options]

        x,y=df.shape

        return html.Div([
                        html.Div(id='reference'),                        
                        html.Div(id='features'),

                        dcc.Checklist(
                        id='selectall',
                        options=[{'label': 'All features', 'value':'ALL'}],
                        value=[],
                        ),
                        
                        #max for max_depth is number of rows 
                        drc.NamedDropdown(name="n_clusters",
                        id="n_clusters",                                            
                        clearable=True,
                        searchable=True,
                        options=[{'label': str(i), 'value':i} for i in range(1,x)],
                        value=None,
                        multi=False
                        ),
                        html.Br(),
                        html.Div(id='buttonbox')
            ]            
            )
        

