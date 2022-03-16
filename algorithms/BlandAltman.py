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



class BlandAltman:
    @staticmethod
    def getAlgoProps(options,colorscales):        
        method1 = None
        method2 = None
        #hoverVal = None
        return html.Div([
            drc.NamedDropdown(name="Feature-1",
                                        id="method1BlandAltman",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=method1,
                                        multi=False
                                    ),
            
            drc.NamedDropdown(name="Feature-2",
                                        id="method2BlandAltman",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=method2,
                                        multi=False
                                    ),

            html.Button('BlandAltman Analysis', id='generate-BlandAltman-Analysis',n_clicks=0)
            ]            
            )
        

