import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd
import networkx as nx
import utils.dash_reusable_components as drc

from DataGuru import DataGuru
from algorithms import data

class Isomap_VP:
    
    @staticmethod
    def getAlgoProps(options,colorscales,globalData):   
        
        #load the data
        df=globalData.dataGuru.getDF()
        x,y=df.shape
        
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
                
                #for plot dimension
                drc.NamedDropdown(name="Plot dimension",
                        id="plotdimensionisomap",                                            
                        clearable=True,
                        searchable=True,
                        multi=False,
                        options=[{'label': '2D', 'value':2},{'label': '3D', 'value':3}],
                        value=None,
                        ),
            
                #for n_neighbors
                drc.NamedDropdown(name="Number of neighbors",
                        id="n_neighbors",                                            
                        clearable=True,
                        searchable=True,
                        multi=False,
                        options=[{'label': str(i), 'value':i} for i in range(1,x)],
                        value=5,
                        ),
                
                #for featurs to give select all functionality     
                html.Div(id='features'),
                
                #select all check box
                dcc.Checklist(
                            id='selectall',
                            options=[{'label': 'All features', 'value':'ALL'}],
                            value=[],
                            ),

                #for output button
                html.Div(id='buttonbox'),

            ]            
            )
        

