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
from algorithms.DataPreprocessAlgo import DataPreprocessAlgo
from algorithms import data

class DataVis:
    def __init__(self):
        pass

    def getFeatureVis(self):
        df=DataPreprocessAlgo()     
        features = list(df.getcolumns())
        features=[{'label': i,'value':i} for i in features ]
        #features.append({'label': 'All','value':'All'})   if want to include all   
        #print(features)
        return html.Div([
            html.B('Select Features for visualization'),
           
            drc.NamedDropdown(name="Reference",
                                        id="referencevis",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=features,
                                        value=None,
                                        multi=False
                                    ),
            html.Div(id='featuresvis'),
                        
                         dcc.Checklist(
                        id='selectallvis',
                        options=[{'label': 'All features', 'value':'ALL'}],
                        value=[],
                        ),
            html.Button('visualize', id='generate-visulization',n_clicks=0)
            ]            
            )
        # return "hello"        

