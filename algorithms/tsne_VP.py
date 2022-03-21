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
from algorithms import data

class tSNE_VP:

    @staticmethod
    def getAlgoProps(options,colorscales):        
        df=data.dataGuru.getDF()
        x,y=df.shape
        PerplexityOptions=[{'label':str(i),'value':i} for i in range(1,x)]
        PerplexityOptions.append({'label':'default','value':30})
        IterationOptions=[{'label':str(i),'value':i} for i in range(250 ,5001)]
        IterationOptions.append({'label':'default','value':1000})
        #print(type(IterationOptions))
        return html.Div([

            drc.NamedDropdown(name="Reference",
                                        id='reference',                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=None,
                                        multi=False
                                    ),
            drc.NamedDropdown(name="Perplexity",
                                        id='perplexityTsne',                                            
                                        clearable=True,
                                        searchable=True,
                                        options=PerplexityOptions,
                                        value=30,
                                        multi=False
                                    ),
            
            drc.NamedDropdown(name="Iteration",
                                        id='iterationTsne',                                            
                                        clearable=True,
                                        searchable=True,
                                        options=IterationOptions,
                                        value=1000,
                                        multi=False
                                    ),
             html.Div(id='features'),
             dcc.Checklist(
                        id='selectall',
                        options=[{'label': 'All features', 'value':'ALL'}],
                        value=[],
                        ),



            html.Button('tSNE Analysis', id='generate-tsne-analysis',n_clicks=0)
            ]            
            )
        

