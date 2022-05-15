import dash
import dash_table
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import networkx as nx

import utils.dash_reusable_components as drc

class SVM_VP:

    @staticmethod
    def getAlgoProps(colorscales,globalData):        

        df=globalData.dataGuru.getDF()
        options=df.columns
        options=[{'label':i,'value':i} for i in options]
            
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
                        
                        #for features
                        html.Div(id='features'),
                        
                        #for selectall check box
                         dcc.Checklist(
                        id='selectall',
                        options=[{'label': 'All features', 'value':'ALL'}],
                        value=[],
                        ),
                         
                         #for kernal default is set to 'rbf'
                        drc.NamedDropdown(name="kernel",
                        id="kernel",                                            
                        clearable=False,
                        searchable=True,
                        options=[{'label': i, 'value':i} for i in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']],
                        value='rbf',
                        multi=False
                        ),
                        
                        #for degree default 3 is selected
                        drc.NamedDropdown(name="degree",
                        id="degree",                                            
                        clearable=False,
                        searchable=True,
                        options=[{'label': str(i), 'value':i} for i in range(1,10)],
                        value=3,
                        multi=False
                        ),
                        
                        html.Br(),
                        html.Br(),

                        #for creating radio button b/w cross fold and percentage split
                        html.Div(className='rowhalf2',children=[
                                
                                html.Div(className='colhalf2',children=[html.Br(),html.Br(),
                                dcc.RadioItems(id='setcreation',
                                options=[
                                {'label': 'Cross-validation     _._._._._._._._._._._ \n _._._._._._._._._._._  ', 'value':'cross'},
                                
                                
                                {'label': 'Percentage Split ', 'value': 'percentage'},
                                ],
                                value='cross'
                                ),]),
                                
                                html.Div(className='colhalf2',children=[
                                html.Div(id='selection')
                                ])


                                    ]),
                                        
                            html.Br(),
                            html.Br(),
                            #for button
                            html.Div(id='buttonspace')
                            ]            
                            )
        

