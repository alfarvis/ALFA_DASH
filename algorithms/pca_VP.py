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
from DataGuru import DataGuru
from algorithms.DataPreprocessAlgo import DataPreprocessAlgo


class PCA_VP:

    @staticmethod
    def getAlgoProps(options,colorscales):        

        return html.Div([
                       
                    

                        drc.NamedDropdown(name="Reference",
                        id="reference",                                            
                        clearable=True,
                        searchable=True,
                        options=options,
                        value=None,
                        multi=False
                        ),
                        
                        drc.NamedDropdown(name="Plot dimension",
                        id="plotdimensionPca",                                            
                        clearable=True,
                        searchable=True,
                        multi=False,
                        options=[{'label': '2D', 'value':2},{'label': '3D', 'value':3}],
                        value=None,
                        ),
                        html.Div(id='features'),
                        
                         dcc.Checklist(
                        id='selectall',
                        options=[{'label': 'All features', 'value':'ALL'}],
                        value=[],
                        ),
                        
                        


            html.Button('PCA Analysis', id='generate-pca-analysis',n_clicks=0)
            ]            
            )
        

