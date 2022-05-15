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



class Histogram_VP:

    @staticmethod
    def getAlgoProps(options,colorscales):        
        
        return html.Div([
            drc.NamedDropdown(name="Select Feature",
                                        id="feature",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=None,
                                        multi=False
                                    ),
            html.Button('Generate a histogram plot', id='generate-histogram-plot',n_clicks=0)
            ]            
            )
