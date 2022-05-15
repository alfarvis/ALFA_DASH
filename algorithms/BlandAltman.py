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

from getAlgorithms import getAlgorithms
from DataGuru import DataGuru

class BlandAltman:
    
    @staticmethod
    def getAlgoProps(options,colorscales):        
        
        method1 = None
        method2 = None
        return html.Div([
            
            #for feature1
            drc.NamedDropdown(name="Feature-1",
                                        id="method1BlandAltman",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=method1,
                                        multi=False
                                    ),
            
            #for feature2
            drc.NamedDropdown(name="Feature-2",
                                        id="method2BlandAltman",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=method2,
                                        multi=False
                                    ),
            #for button 
            html.Button('BlandAltman Analysis', id='generate-BlandAltman-Analysis',n_clicks=0)
            ]            
            )
        

