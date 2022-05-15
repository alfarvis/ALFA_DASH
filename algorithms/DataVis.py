import dash
import dash_table
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import networkx as nx
import utils.dash_reusable_components as drc

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
 
from algorithms.DataPreprocessAlgo import DataPreprocessAlgo
from algorithms import data
from getAlgorithms import getAlgorithms

class DataVis:
    def __init__(self):
        pass

    def getFeatureVis(self,globalData):
        
        df=DataPreprocessAlgo(globalData)     
        features = list(df.getcolumns())
        features=[{'label': i,'value':i} for i in features ]
        
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

            html.Div(id='visbutton'),
                        ]            
                        )
