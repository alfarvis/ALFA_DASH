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
from algorithms.scatter2D_VP import Scatter2D_VP
from algorithms.scatter3D_VP import Scatter3D_VP
from algorithms.bar_VP import Bar_VP
from algorithms.box_VP import Box_VP
from algorithms.violin_VP import Violin_VP
from algorithms.pr_curve_VP import PR_Curve_VP
from algorithms.roc_curve_VP import ROC_Curve_VP
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression


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
server = app.server

app.title = 'Alfarvis'
algoClass = getAlgorithms()
dataGuru = DataGuru()
algo_list = algoClass.getAlgos()
scatter2D_VP = Scatter2D_VP()
scatter3D_VP = Scatter3D_VP()
bar_VP = Bar_VP()
box_VP = Box_VP()
violin_VP = Violin_VP()
pr_curve_VP = PR_Curve_VP()
roc_curve_VP = ROC_Curve_VP()

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
                                        name="Select/Search Algorithm",
                                        id="algorithm",
                                        options = algo_list,
                                        clearable=True,
                                        searchable=True,
                                        value=None,
                                    ),
                                    html.Hr(),    
                                    html.Div(id="functionProperties")
                                                                            
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
                                    dcc.Tab(label='Data',value='tab-2',
                                        children=[
                                        html.Div(id='loaded_data_table'),
                                        ]
                                        ),
                                    dcc.Tab(label='Analysis',value='tab-1',
                                        children=[                                            
                                            html.Div(id="algoRes")                                            
                                            #html.Div(id='output_box_graph'),
                                            #html.Div(id='output_bar_graph'),
                                            #html.Div(id='output_scatter_graph'),
                                            #html.Div(id='output_scatter_graph3d'),
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
    Output('loaded_data_table', 'children'),
    [Input(component_id='memory-output', component_property='data')],
    )
def update_table(data):
    
    if data is None:
        raise PreventUpdate
    print(data[0].keys())
    df = pd.DataFrame.from_records(data)
    print('here')
    dataGuru.setDF(df)

    return html.Div([dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        )])

# Algorithm Selection
@app.callback(    
    [Output('functionProperties', 'children'),
    Output('algoRes', 'children'),
    ],
    [Input(component_id='algorithm', component_property='value')],    
    )
def algorithmProps(algorithm):
    
    df = dataGuru.getDF()
    options = [{'label':i,'value':i} for i in df.columns]
    if algorithm in 'barPlot':
        return bar_VP.getAlgoProps(options,colorscales), html.Div(id='output_bar_graph')
    if algorithm in 'boxPlot':
        return box_VP.getAlgoProps(options,colorscales), html.Div(id='output_box_graph')
    if algorithm in 'violinPlot':
        return violin_VP.getAlgoProps(options,colorscales), html.Div(id='output_violin_graph')
    if algorithm in 'scatter2D':                
        return scatter2D_VP.getAlgoProps(options,colorscales), html.Div(id='output_scatter_graph')
    if algorithm in 'scatter3D':        
        return scatter3D_VP.getAlgoProps(options,colorscales), html.Div(id='output_scatter_graph3d')
    if algorithm in 'pr_curve_VP':        
        return pr_curve_VP.getAlgoProps(options,colorscales), [html.Div(id='output_pr_curve'), html.Div(id="algoRes_Table")]
    if algorithm in 'roc_curve_VP':        
        return roc_curve_VP.getAlgoProps(options,colorscales), [html.Div(id='output_roc_curve'), html.Div(id="algoRes_Table_ROC")]

# Algorithmic implementations

@app.callback(
    Output('subClass','options'),    
    [
    Input('featureColor','value'),    
    ]
    )
def getSubClasses(featureColor):
    df = dataGuru.getDF()    
    a=np.unique(df[featureColor])
    uniqVals = [{'label':i,'value':i} for i in a]
    #uniqVals = [str(i) for i in a]
    print(uniqVals)
    return uniqVals


@app.callback(
    [Output('output_pr_curve','children'),
    Output('algoRes_Table', 'children')],
    [Input('generate-pr-curve','n_clicks'),
    ],
    [
    State('featureX','value'),    
    State('featureColor','value'),
    State('subClass','value'),
    State('colorscale','value'),
    ]
    )
def createPRPlot(n_clicks,featureX,featureColor,subClass,colorscale):
    print(n_clicks)
    if n_clicks>0:        
        df = dataGuru.getDF()
        df2 = pd.DataFrame()
        df2['Feature'] = []
        f1 = [];
        f2 = [];
        df2['AUC-PR'] = []
        # Define the inputs and outputs
       
        y = df[featureColor]

        y_onehot = pd.get_dummies(y, columns=np.unique(df[featureColor]))
        
        # Fit the model
        
        # Create an empty figure, and iteratively add new lines
        # every time we compute a new class

        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )

        uniqVals = list(np.unique(df[featureColor]))
        subClass_pos = uniqVals.index(subClass)
        for i in featureX:

            X = np.array(df[i])
            indToRemove = np.isnan(X)
            X = X[indToRemove==False]
            X = X.reshape(-1,1)
            model = LogisticRegression(max_iter=200)
            yFit = y[indToRemove==False]
            model.fit(X, yFit)
            y_scores = model.predict_proba(X)
            
            
            y_true = y_onehot.iloc[indToRemove==False, subClass_pos]
            
            y_score = y_scores[:, subClass_pos]
            #y_score = df[i]

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            auc_score = average_precision_score(y_true, y_score)

            name = f"{i} (AUC-PR={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=recall, y=precision, name=i, mode='lines'))
            f1.append(i)
            f2.append(auc_score)


        #fig = px.bar(df, x=featureX, y=featureY, color=featureColor, color_continuous_scale=colorscale) 
        df2['Feature']=f1
        df2['AUC-PR']=f2
        return html.Div([
            dcc.Graph(
        id='example-graph1',
        figure=fig,
                )
            ]), html.Div([dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df2.columns],
        data=df2.to_dict('records'),
        )])

@app.callback(
    Output('subClassROC','options'),    
    [
    Input('featureColorROC','value'),    
    ]
    )
def getSubClasses(featureColor):
    df = dataGuru.getDF()    
    a=np.unique(df[featureColor])
    uniqVals = [{'label':i,'value':i} for i in a]
    #uniqVals = [str(i) for i in a]
    print(uniqVals)
    return uniqVals

@app.callback(
    [Output('output_roc_curve','children'),
    Output('algoRes_Table_ROC', 'children')],
    [Input('generate-roc-curve','n_clicks'),
    ],
    [
    State('featureX','value'),    
    State('featureColorROC','value'),
    State('subClassROC','value'),
    State('colorscale','value'),
    ]
    )
def createROCPlot(n_clicks,featureX,featureColor,subClass,colorscale):
    print(n_clicks)
    if n_clicks>0:        
        df = dataGuru.getDF()
        df2 = pd.DataFrame()
        df2['Feature'] = []
        f1 = [];
        f2 = [];
        df2['AUC-PR'] = []
        # Define the inputs and outputs
       
        y = df[featureColor]

        y_onehot = pd.get_dummies(y, columns=np.unique(df[featureColor]))
        
        # Fit the model
        
        # Create an empty figure, and iteratively add new lines
        # every time we compute a new class

        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )

        uniqVals = list(np.unique(df[featureColor]))
        subClass_pos = uniqVals.index(subClass)
        for i in featureX:

            X = np.array(df[i])
            indToRemove = np.isnan(X)
            X = X[indToRemove==False]
            X = X.reshape(-1,1)
            model = LogisticRegression(max_iter=200)
            yFit = y[indToRemove==False]
            model.fit(X, yFit)
            y_scores = model.predict_proba(X)
            
            
            y_true = y_onehot.iloc[indToRemove==False, subClass_pos]
            
            y_score = y_scores[:, subClass_pos]
            #y_score = df[i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{i} (AUC-PR={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=i, mode='lines'))
            f1.append(i)
            f2.append(auc_score)


        #fig = px.bar(df, x=featureX, y=featureY, color=featureColor, color_continuous_scale=colorscale) 
        df2['Feature']=f1
        df2['AUC-PR']=f2
        return html.Div([
            dcc.Graph(
        id='example-graph1',
        figure=fig,
                )
            ]), html.Div([dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df2.columns],
        data=df2.to_dict('records'),
        )])


@app.callback(
    Output('output_bar_graph','children'),
    [Input('generate-bar-plot','n_clicks'),
    ],
    [
    State('featureX','value'),
    State('featureY','value'),
    State('featureColor','value'),
    State('colorscale','value'),
    ]
    )
def createBarPlot(n_clicks,featureX,featureY,featureColor,colorscale):
    print(n_clicks)
    if n_clicks>0:        
        df = dataGuru.getDF()
        fig = px.bar(df, x=featureX, y=featureY, color=featureColor, color_continuous_scale=colorscale)                
        return html.Div([
            dcc.Graph(
        id='example-graph1',
        figure=fig,
                )
            ])

@app.callback(
    Output('output_violin_graph','children'),
    [Input('generate-violin-plot','n_clicks'),
    ],
    [State('featureX','value'),
    State('featureY','value'),
    State('featureColor','value'),
    State('colorscale','value'),
    ]
    )
def createBarPlot(n_clicks,featureX,featureY,featureColor,colorscale):
    print(n_clicks)
    if n_clicks>0:        
        df = dataGuru.getDF()
        fig = px.violin(df, x=featureX, y=featureY, color=featureColor)                
        return html.Div([
            dcc.Graph(
        id='example-graph1',
        figure=fig,
                )
            ])

@app.callback(
    Output('output_box_graph','children'),
    [Input('generate-box-plot','n_clicks'),
    ],
    [
    State('featureX','value'),
    State('featureY','value'),
    State('featureColor','value'),
    State('colorscale','value'),
    ]
    )
def createBoxPlot(n_clicks,featureX,featureY,featureColor,colorscale):
    print(n_clicks)
    if n_clicks>0:        
        df = dataGuru.getDF()
        fig = px.box(df, x=featureX, y=featureY, color=featureColor)                
        return html.Div([
            dcc.Graph(
        id='example-graph2',
        figure=fig,
                )
            ])
        
        
@app.callback(
    Output('output_scatter_graph','children'),
    [Input('generate-scatter-plot','n_clicks'),
    ],
    [
    State('featureX','value'),
    State('featureY','value'),
    State('featureColor','value'),
    State('featureHover','value'),
    State('featureSize','value'),
    State('colorscale','value'),
    ]
    )
def createScatterPlot(n_clicks,featureX,featureY,featureColor,featureHover,featureSize,colorscale):
    print(n_clicks)
    if n_clicks>0: 
        #df = pd.DataFrame.from_records(data)                    
        df = dataGuru.getDF()
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
        
    
@app.callback(
    Output('output_scatter_graph3d','children'),
    [Input('generate-scatter-plot3','n_clicks'),
    ],
    [
    State('featureX','value'),
    State('featureY','value'),
    State('featureZ','value'),
    State('featureColor','value'),
    State('featureHover','value'),
    State('featureSize','value'),
    State('colorscale','value'),
    ]
    )
def createScatterPlot3d(n_clicks,featureX,featureY,featureZ,featureColor,featureHover,featureSize,colorscale):
    print(n_clicks)
    if n_clicks>0: 
        #df = pd.DataFrame.from_records(data)                    
        df = dataGuru.getDF()
        if featureHover is not None:
            if type(featureHover) != list:
                featureHover = [featureHover]
        fig = px.scatter_3d(df,x=featureX, y=featureY,z=featureZ,color=featureColor,hover_data = featureHover, size = featureSize, color_continuous_scale=colorscale)       
        return html.Div([
            dcc.Graph(
        id='example-graph3',
        figure=fig,
                )
            ],
        style = {'display': 'inline-block', 'width': '90%', 'height': '90%'}
            )


#@app.callback(
#    Output('subtype-text', 'value'),
#    [Input('subtype-text', 'options'),],
#    )
#def setValueSubtype(options):
#    if myData is not None:
#        print(options)    
#        return options[0]['value']



if __name__ == '__main__':

    app.run_server(debug=True)
