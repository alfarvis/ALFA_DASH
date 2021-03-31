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
import plotly.express as px
#import utils.figures as figs
colorscales = px.colors.named_colorscales()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def path(filename):
    parent = os.path.dirname(__file__)
    return os.path.join(parent, filename)

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets=external_stylesheets
)

app.title = 'Alfarvis'
algoClass = getAlgorithms()
algo_list = algoClass.getAlgos()
app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        
                        html.H2(
                            id="banner-title",
                            children=[
                                html.Img(
                                  src=app.get_asset_url("ALFA2.png"),
                                  style={'height':'96px', "padding-top": "6px"}
                                ),
                                html.A(                           
                                    #href="https://github.com/plotly/dash-svm",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                        "line-height": "64px",
                                    },
                                )
                            ],
                        ),
                        
                        html.Br(),
                        
                        
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            className="column",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        html.Button('Load', id='load-button-state',n_clicks=0),
                                        html.Div([
                                                dcc.Upload(
                                                    id='upload-data',
                                                    children=html.Div([
                                                        'Drag and Drop or ',
                                                        html.A('Select Files')
                                                    ]),
                                                    style={
                                                        'width': '80%',
                                                        'height': '60px',
                                                        'lineHeight': '60px',
                                                        'borderWidth': '1px',
                                                        'borderStyle': 'dashed',
                                                        'borderRadius': '5px',
                                                        'textAlign': 'center',                                                        
                                                    },
                                                    # Allow multiple files to be uploaded
                                                    multiple=False
                                                ),
                                    ]),                        
                                    html.Div(id='drgDrop'),
                                    dcc.Store(id='memory-output'),
                                    drc.NamedDropdown(
                                        name="Select Reference Feature",
                                        id="subtype-reference",                                            
                                        clearable=True,
                                        searchable=True,
                                        value=None,
                                        multi=False
                                    ),
                                    drc.NamedDropdown(
                                        name="Select Feature(s)",
                                        id="subtype-text",                                            
                                        clearable=True,
                                        searchable=True,
                                        value=None,
                                        multi=True
                                    ),
                                    drc.NamedDropdown(
                                        name="Select Algorithm",
                                        id="algorithm",
                                        options = algo_list,
                                        clearable=True,
                                        searchable=True,
                                        value=None,
                                    ),
                                    html.Hr(),    
                                    html.Div(id="functionProperties")
                                    
                                    #dash_table.DataTable(
                                    #    id='memory-table',   
                                    #    columns=[{'name': i, 'id': i} for i in range(0,100,5)]
                                    #    )
                                        
                                    ],
                                ),
                                html.Br(),
                                html.Br(),
                                html.Br(),
                                html.Br(),
                                html.Br(),
                            ],
                        ),
                        html.Div(
                          id="output_window",
                          className="two-thirds column row",
                          children=[
                            dcc.Tabs(id="my-tabs",value="tab-1", children = [
                                    dcc.Tab(label='Data',value='tab-1',
                                        children=[
                                        html.Div(id='loaded_data_table'),
                                        ]
                                        ),
                                    dcc.Tab(label='Analysis',value='tab-2',
                                        children=[
                                            html.Div(id='output_bar_graph'),
                                            html.Div(id='output_scatter_graph'),
                                        ]
                                        )
                                ]
                                )
                            
                            
                            # drc.Cards('sample-cards')
                        ]),
                    ],
                )
            ],
        ),
    ]
)


# @app.callback(
#     Output(component_id='drgDrop', component_property='children'),
#     [Input('load-button-state', 'n_clicks'),]
#     )
# def create_load_dropdown(n_clicks):
#     if n_clicks % 2 == 1:
#         return html.Div([
#             dcc.Upload(
#                 id='upload-data',
#                 children=html.Div([
#                     'Drag and Drop or ',
#                     html.A('Select Files')
#                 ]),
#                 style={
#                     'width': '100%',
#                     'height': '60px',
#                     'lineHeight': '60px',
#                     'borderWidth': '1px',
#                     'borderStyle': 'dashed',
#                     'borderRadius': '5px',
#                     'textAlign': 'center',
#                     'margin': '10px'
#                 },
#                 # Allow multiple files to be uploaded
#                 multiple=False
#             ),
# ])

@app.callback(    
    Output(component_id='memory-output', component_property='data'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
    State('upload-data', 'last_modified')])
def upload_file(contents, filename, date):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        myData  = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return myData.to_dict('records')

@app.callback(    
    [Output('subtype-text', 'options'),
    Output('subtype-reference', 'options'),
    Output('loaded_data_table', 'children')
    ],
    [Input(component_id='memory-output', component_property='data')],
    )
def update_table(data):
    
    if data is None:
        raise PreventUpdate
    print(data[0].keys())
    df = pd.DataFrame.from_records(data)
    
    return [{'label':key,'value':key} for key in data[0].keys()],[{'label':key,'value':key} for key in data[0].keys()], html.Div([dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        )])

@app.callback(    
    Output('functionProperties', 'children'),
    [Input(component_id='algorithm', component_property='value')],
    [State('subtype-text','options'),
    State('subtype-text','value'),
    State('subtype-reference','value'),    
    ]
    )
def algorithmProps(algorithm,options,value,refVal):
    
    if algorithm in 'bar plot':
        return html.Div([
            html.Button('Generate bar plot', id='generate-bar-plot',n_clicks=0)
            ]            
            )
    if algorithm in 'scatter plot':
        valueX = None
        valueY = None
        if value is not None:
            valueX = value[0]
            if len(value)>1:
                valueY = value[1]
        
        colorVal = None
        hoverVal = None
        sizeVal = None
        if refVal is not None:
            colorVal = refVal
            hoverVal = refVal
            sizeVal = refVal
        
        

        return html.Div([
            drc.NamedDropdown(name="Feature on the x axis",
                                        id="featureX",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=valueX,
                                        multi=False
                                    ),
            drc.NamedDropdown(name="Feature on the y axis",
                                        id="featureY",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=valueY,
                                        multi=False
                                    ),
            
            drc.NamedDropdown(name="Hover feature",
                                        id="featureHover",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=hoverVal,
                                        multi=True
                                    ),
            drc.NamedDropdown(name="Size feature",
                                        id="featureSize",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=sizeVal,
                                        multi=False
                                    ),

            drc.NamedDropdown(name="Reference feature for color mapping",
                                        id="featureColor",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=options,
                                        value=colorVal,
                                        multi=False
                                    ),
            drc.NamedDropdown(name="Color Scale",
                                        id="colorscale",                                            
                                        clearable=True,
                                        searchable=True,
                                        options=[{"value": x, "label": x} 
                                                 for x in colorscales],
                                        value='viridis'
                                    ),

            html.Button('Generate scatter plot', id='generate-scatter-plot',n_clicks=0)
            ]            
            )

@app.callback(
    Output('output_bar_graph','children'),
    [Input('generate-bar-plot','n_clicks'),
    ],
    [State('memory-output','data')]
    )
def createBarPlot(n_clicks,data):
    print(n_clicks)
    if n_clicks>0:        
        return html.Div([
            dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
                    }
                    }
                )
            ])
        
        
@app.callback(
    Output('output_scatter_graph','children'),
    [Input('generate-scatter-plot','n_clicks'),
    ],
    [State('memory-output','data'),
    State('featureX','value'),
    State('featureY','value'),
    State('featureColor','value'),
    State('featureHover','value'),
    State('featureSize','value'),
    State('colorscale','value'),
    ]
    )
def createScatterPlot(n_clicks,data,featureX,featureY,featureColor,featureHover,featureSize,colorscale):
    print(n_clicks)
    if n_clicks>0: 
        df = pd.DataFrame.from_records(data)                    
        if featureHover is not None:
            if type(featureHover) != list:
                featureHover = [featureHover]
        fig = px.scatter(df,x=featureX, y=featureY,color=featureColor,hover_data = featureHover, size = featureSize, color_continuous_scale=colorscale)       
        return html.Div([
            dcc.Graph(
        id='example-graph',
        figure=fig,
                )
            ])
        
    



#@app.callback(
#    Output('subtype-text', 'value'),
#    [Input('subtype-text', 'options'),],
#    )
#def setValueSubtype(options):
#    if myData is not None:
#        print(options)    
#        return options[0]['value']



if __name__ == '__main__':

    #parser = argparse.ArgumentParser(description='Run the seed server.')
    #parser.add_argument("-d", "--debug", required=False, help="running in debug mode?", default=False)
    #args = vars(parser.parse_args())
    #is_debug = args['debug']

    app.run_server(debug=False)
