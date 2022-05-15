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

class PCA_VP:

    @staticmethod
    def getAlgoProps(options,colorscales):        

        return html.Div([
                       
                        #for reference
                        drc.NamedDropdown(name="Reference",
                        id="reference",                                            
                        clearable=True,
                        searchable=True,
                        options=options,
                        value=None,
                        multi=False
                        ),
                        
                        #for Plot dimension
                        drc.NamedDropdown(name="Plot dimension",
                        id="plotdimensionPca",                                            
                        clearable=True,
                        searchable=True,
                        multi=False,
                        options=[{'label': '2D', 'value':2},{'label': '3D', 'value':3}],
                        value=None,
                        ),
                        
                        #for featurs to give select all functionality     
                        html.Div(id='features'),
                        
                        #select all checklist
                        dcc.Checklist(
                        id='selectall',
                        options=[{'label': 'All features', 'value':'ALL'}],
                        value=[],
                        ),
                        
                        
                        #for output button
                        html.Div(id='buttonbox'),

            ]            
            )
        

