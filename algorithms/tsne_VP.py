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
from algorithms import data

class tSNE_VP:

    @staticmethod
    def getAlgoProps(options,colorscales,globalData):        

        df=globalData.dataGuru.getDF()
        x,y=df.shape
        PerplexityOptions=[{'label':str(i),'value':i} for i in range(1,x)]
        IterationOptions=[{'label':str(i),'value':i} for i in range(250 ,5001)]

        return html.Div([

            #for th referecne
            drc.NamedDropdown(name="Reference",
                                        id='reference',                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=None,
                                        multi=False
                                    ),
            
            #for Perplexity
            drc.NamedDropdown(name="Perplexity",
                                        id='perplexityTsne',                                            
                                        clearable=True,
                                        searchable=True,
                                        options=PerplexityOptions,
                                        value=30,
                                        multi=False
                                    ),
            #for Iteration
            drc.NamedDropdown(name="Iteration",
                                        id='iterationTsne',                                            
                                        clearable=True,
                                        searchable=True,
                                        options=IterationOptions,
                                        value=1000,
                                        multi=False
                                    ),
             
            #for features to give select all functionality
            html.Div(id='features'),
             
            #check box for select all
            dcc.Checklist(
                        id='selectall',
                        options=[{'label': 'All features', 'value':'ALL'}],
                        value=[],
                        ),

            #button for output
            html.Div(id='buttonbox'),
            
            ]            
            )
        

