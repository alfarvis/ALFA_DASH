import os  
import time
import importlib
import base64
import argparse
import io
import numpy as np
import pandas as pd
import networkx as nx
from turtle import color
import dash
import dash_auth
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import utils.dash_reusable_components as drc
from getAlgorithms import getAlgorithms
from DataGuru import DataGuru
from algorithms.DataPreprocessAlgo import DataPreprocessAlgo
from algorithms.scatter2D_VP import Scatter2D_VP
from algorithms.scatter3D_VP import Scatter3D_VP
from algorithms.bar_VP import Bar_VP
from algorithms.box_VP import Box_VP
from algorithms.violin_VP import Violin_VP
from algorithms.pr_curve_VP import PR_Curve_VP
from algorithms.roc_curve_VP import ROC_Curve_VP
from algorithms.pca_VP import PCA_VP
from algorithms.PcaAlgo import PcaAlgo
from algorithms.tsne_VP import tSNE_VP
from algorithms.tsneAlgo import TsneAlgo
from algorithms.DataVis import DataVis
from algorithms.BlandAltman import BlandAltman
from algorithms.BlandAltmanAlgo import BlandAltmanAlgo
from algorithms.isomap_VP import Isomap_VP
from algorithms.IsomapAlgo import IsomapAlgo
from algorithms.DecisionTree_VP import DecisionTree_VP
from algorithms.DecisionTreeAlgo import DecisionTreeAlgo
from algorithms.RandomForest_VP import RandomForest_VP
from algorithms.RandomForestAlgo import RandomForestAlgo 
from algorithms.SVM_VP import SVM_VP
from algorithms.SVMAlgo import SVMAlgo
from algorithms.AutoML_VP import AutoML_VP
from algorithms.AutoMLAlgo import AutoMLAlgo
from algorithms.LogisticRegression_VP import LogisticRegression_VP
from algorithms.LogisticRegressionAlgo import LogisticRegressionAlgo
from algorithms.ArtificialNeuralNetworks_VP import ArtificialNeuralNetworks_VP
from algorithms.ArtificialNeuralNetworksAlgo import ArtificialNeuralNetworksAlgo
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from algorithms import data

# Basic dash auth
# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = {
    'hello': 'world'
}


#import utils.figures as figs
colorscales = px.colors.named_colorscales()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','./assets/grid.css']

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
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
featureselection = html.Div(id='featureselection')
featureselectionvis = html.Div(id='featureselectionvis')
referencevis=html.Div(id='referencevis')
reference=html.Div(id='reference')
app.title = 'Alfarvis'
algoClass = getAlgorithms()
algo_list = algoClass.getAlgos()
scatter2D_VP = Scatter2D_VP()
scatter3D_VP = Scatter3D_VP()
bar_VP = Bar_VP()
box_VP = Box_VP()
violin_VP = Violin_VP()
pr_curve_VP = PR_Curve_VP()
roc_curve_VP = ROC_Curve_VP()
pca_VP = PCA_VP()
blandAltman=BlandAltman()
dataVis=DataVis()
tsne_VP=tSNE_VP()
isomap_VP=Isomap_VP()
decisionTree_VP=DecisionTree_VP()
randomForest_VP=RandomForest_VP()
svm_VP=SVM_VP()
autoML_VP=AutoML_VP()
logisticRegression_VP=LogisticRegression_VP()
artificialNeuralNetworks_VP=ArtificialNeuralNetworks_VP()
data.init()
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
                                               
                                        html.Div(id='upload-data-conformation')

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
                                    featureselection,reference,    
                                    html.Div(id="functionProperties"),
                                    
                                    
                                                                            
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
                                        
                                        dcc.Tab(label='Dashboard',value='tab-1',
                                        children=[ 
                                        drc.Card(  id='viscard',children=[
                                        featureselectionvis,referencevis,
                                        html.Div(id='get_vis'),
                                          
                                        html.Div(id="vis_table") ,
                                        html.Div(id="vis_splom") ,
                                        html.B("Here we will put visulization")] 
                                        )]),
                                        
                                        dcc.Tab(label='Data',value='tab-2',
                                        children=[
                                            html.Div(id='filteration',children=[
                                        data.getfilter()]),
                                        html.Div(id='loaded_data_table')
                                        ]
                                        ),
                                        
                                        dcc.Tab(label='Analysis',value='tab-3',
                                        children=[                                            
                                        html.Div(id="algoRes")                                            
                                        #html.Div(id='output_box_graph'),
                                        #html.Div(id='output_bar_graph'),
                                        #html.Div(id='output_scatter_graph'),
                                        #html.Div(id='output_scatter_graph3d'),
                                        ]
                                        ),
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
    [Output(component_id='memory-output', component_property='data'),  
     Output('upload-data-conformation', 'children'),
     Output('get_vis', 'children'),
    ],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    State('featureselection','value'),
    ])
def upload_file(contents, filename, date,featureselection):
    if contents is not None:

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        myData  = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        myData.dropna(subset=myData.keys(),inplace=True)#changed by me
        df=pd.DataFrame.from_records(myData)
        data.dataGuru.setDF(df)
        return [myData.to_dict('records'),html.A(str(filename).upper()+' UPLOADED',style={
                                        "text-decoration": "none",
                                        "color": "Green",
                                        "line-height": "64px",
                                    },), dataVis.getFeatureVis()]
    else:
        df=pd.DataFrame()
        data.dataGuru.setDF(df)
        return ['',html.A('DATA NOT UPLOADED',style={
                                        "text-decoration": "none",
                                        "color": "Red",
                                         "line-height": "64px",
                                    },),html.A(id='generate-visulization',children=[html.A(id='visfeatures')])]

@app.callback(        
    [   Output('filteration', 'children'),
        Output('loaded_data_table', 'children'),],
    [Input(component_id='memory-output', component_property='data'),
     Input('generate-filtered-data', 'n_clicks')],
    [
        State('filtfeatures','value'),
        State('filtoperation','value'),
        State('inputvalue','value')
    ]
    )
def update_table(data1,n_clicks,filtfeatures,filtoperation,inputvalue):
    
    if data1 is None:
        raise PreventUpdate
    
    if n_clicks>0 and filtfeatures!=None and filtoperation!=None and inputvalue!=None:
        print('hello')
        df = data.dataGuru.getDF()
        if filtoperation=='=':
            df=df[df[filtfeatures]==inputvalue]
        elif filtoperation=='>':
            df=df[df[filtfeatures]>inputvalue]
        elif filtoperation=='<':
            df=df[df[filtfeatures]<inputvalue]
        elif filtoperation=='<=':
            df=df[df[filtfeatures]<=inputvalue]
        elif filtoperation=='>=':
            df=df[df[filtfeatures]>=inputvalue]
        
        #print(df)
        data.dataGuru.setDF(df)
        return data.getfilter(),data.getTable()
    else:
        # print(data1[0].keys())
        #df = pd.DataFrame.from_records(data1)
        # print('here')
        # data.dataGuru.setDF(df)
        #print(data.dataGuru.getDF())
        return data.getfilter(),data.getTable()

# Algorithm Selection
@app.callback(    
    [
    Output('functionProperties', 'children'),
    Output('algoRes', 'children'),
    ],
    [
    Input('upload-data', 'contents'),
    Input(component_id='algorithm', component_property='value')
    ],    
    )
def algorithmProps(content,algorithm):
    
    df = data.dataGuru.getDF()
    options = [{'label':i,'value':i} for i in df.columns]
    if algorithm==None:
        return ['','']
    if algorithm == 'barPlot':
        return bar_VP.getAlgoProps(options,colorscales), html.Div(id='output_bar_graph')
    if algorithm == 'boxPlot':
        return box_VP.getAlgoProps(options,colorscales), html.Div(id='output_box_graph')
    if algorithm == 'violinPlot':
        return violin_VP.getAlgoProps(options,colorscales), html.Div(id='output_violin_graph')
    if algorithm == 'scatter2D':                
        return scatter2D_VP.getAlgoProps(options,colorscales), html.Div(id='output_scatter_graph')
    if algorithm == 'scatter3D':        
        return scatter3D_VP.getAlgoProps(options,colorscales), html.Div(id='output_scatter_graph3d')
    if algorithm == 'pr_curve_VP':        
        return pr_curve_VP.getAlgoProps(options,colorscales), [html.Div(id='output_pr_curve'), html.Div(id="algoRes_Table")]
    if algorithm == 'roc_curve_VP':        
        return roc_curve_VP.getAlgoProps(options,colorscales), [html.Div(id='output_roc_curve'), html.Div(id="algoRes_Table_ROC")]
    if algorithm == 'pca_VP':        
        return pca_VP.getAlgoProps(options,colorscales), [html.Div(id='output_pca_graph'),html.Div(id="algoRes_Table_PCA"),html.Div(id='output_pca_screePlot')]
    if algorithm == 'blandAltman':        
        return blandAltman.getAlgoProps(options,colorscales), [html.Div(id='output_blandAltman_analysis')]
    if algorithm == 'tsne_VP':        
        return tsne_VP.getAlgoProps(options,colorscales), [html.Div(id='output_tsne_analysis')]
    if algorithm == 'isomap_VP':        
        return isomap_VP.getAlgoProps(options,colorscales), [html.Div(id='output_isomap_analysis')]
    if algorithm == 'dt_VP':
        return decisionTree_VP.getAlgoProps(colorscales), html.Div(id='output_decisiontree_analysis')
    if algorithm == 'rf_VP':
        return randomForest_VP.getAlgoProps(colorscales), html.Div(id='output_randomforest_analysis')
    if algorithm == 'svm_VP':
        return svm_VP.getAlgoProps(colorscales), html.Div(id='output_svm_analysis')
    if algorithm == 'automl_VP':
        return autoML_VP.getAlgoProps(colorscales), html.Div(id='output_automl_analysis')
    if algorithm == 'logreg_VP':
        return logisticRegression_VP.getAlgoProps(colorscales), html.Div(id='output_logisticregression_analysis')
    if algorithm == 'ann_VP':
        return artificialNeuralNetworks_VP.getAlgoProps(colorscales), html.Div(id='output_artificialneuralnetworks_analysis')
        

# @app.callback(   
#                   Output('aaaa', 'children'),
 
#     [
#     Input(component_id = 'table',component_property='data'),    
#     ]
#     )
# def updatedataset(data):
#     data.dataGuru.setDF(data)    
    

# @app.callback(Output(''))




# Algorithmic implementations
@app.callback(
    Output('subClass','options'),    
    [
    Input('featureColor','value'),    
    ]
    )
def getSubClasses(featureColor):
    df = data.dataGuru.getDF()    
    a=np.unique(df[featureColor])
    uniqVals = [{'label':i,'value':i} for i in a]
    #uniqVals = [str(i) for i in a]
    print(uniqVals)
    return uniqVals


         

#BLANDALTMAN
@app.callback(
    Output('output_blandAltman_analysis','children'),
    [Input('generate-BlandAltman-Analysis','n_clicks'),],
    [
    State('method1BlandAltman','value'),    
    State('method2BlandAltman','value'),
    ]
    )
def BlandAltmanAnalysis(n_clicks,method1BlandAltman,method2BlandAltman):
    print(n_clicks)
    if n_clicks>0:
        df = data.dataGuru.getDF()
        blandAltmanAlgo=BlandAltmanAlgo(df,method1BlandAltman,method2BlandAltman)
        return blandAltmanAlgo.getAnswer()
    else :
        return ''      


#PR CURVE
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
        df = data.dataGuru.getDF()
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
    df = data.dataGuru.getDF()    
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
        df = data.dataGuru.getDF()
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
        df = data.dataGuru.getDF()
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
def createViolinPlot(n_clicks,featureX,featureY,featureColor,colorscale):
    print(n_clicks)
    if n_clicks>0:        
        df = data.dataGuru.getDF()
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
        df = data.dataGuru.getDF()
        fig = px.box(df, x=featureX, y=featureY, color=featureColor)                
        return html.Div([
            dcc.Graph(
        id='example-graph2',
        figure=fig,
                )
            ])
        


#select all for algorithms
@app.callback(Output('features','children'),Input(component_id='selectall',component_property='value'),[State('reference','value'),])
def Selcectall(selectall,reference):
    df = data.dataGuru.getDF()
    options = [{'label':i,'value':i} for i in df.columns]
    value=None
    line=''
    if selectall!=[]:
        value=list(df.columns)
        if reference!=None and reference in value:
            value.remove(reference)

        line='You have selected All features'
        
    return html.Div([html.B(line),drc.NamedDropdown(name="Features", 
                    id='featureselection',                                        
                    clearable=True,
                    searchable=True,
                    options=options,
                    value=value,
                    multi=True,
                    )])

#select all for dashboard
@app.callback(
    Output('featuresvis','children'),
    Input(component_id='selectallvis',component_property='value'),
    [State('referencevis','value'),]
            )
def Selcectall(selectall,reference):
    df = data.dataGuru.getDF()
    options = [{'label':i,'value':i} for i in df.columns]
    value=None
    line=''
    if selectall!=[]:
        value=list(df.columns)
        if reference!=None and reference in value:
            value.remove(reference)

        line='You have selected All features'
        
    return html.Div([html.B(line),drc.NamedDropdown(name="Features", 
                    id='featureselectionvis',                                        
                    clearable=True,
                    searchable=True,
                    options=options,
                    value=value,
                    multi=True,
                    )])                       
#TSNE ANALYSIS
@app.callback(
    Output('output_tsne_analysis','children'),
    Input('generate-tsne-analysis','n_clicks'),
    [
    State('featureselection','value'),
    State('reference','value'),    
    State('perplexityTsne','value'),
    State('iterationTsne','value'),
    ]
    )
def TsneAnalysis(n_clicks,featureselection,reference,perplexityTsne,iterationTsne):
    print(n_clicks)
    if n_clicks>0:
        df = data.dataGuru.getDF()
        tsneAlgo=TsneAlgo(df,featureselection,reference,perplexityTsne,iterationTsne)
        return tsneAlgo.getAnswer()
    else :
        return ''      


#PCA ANALYSIS        
@app.callback(
    [Output('output_pca_graph','children'),
     Output('algoRes_Table_PCA', 'children'),
     Output('output_pca_screePlot','children'),],
    [Input('generate-pca-analysis','n_clicks'),],
    [State('featureselection','value'),
     State('reference','value'),
     State('plotdimensionPca','value'),
     ]
    )
def PcaAnalysis(n_clicks,featureselection,reference,plotdimensionPca):
    print(n_clicks) 
    if n_clicks>0 :
        df = data.dataGuru.getDF()
        pca=PcaAlgo(df,featureselection,reference,plotdimensionPca)
        return pca.getPcaAnalysis()
    else :
        return ['','','']


#ISOMAP ANALYSIS        
@app.callback(
    Output('output_isomap_analysis','children'),
    [
    Input('generate-isomap-analysis','n_clicks'),
    ],
    
    [
    State('featureselection','value'),
    State('reference','value'),
    State('n_neighbors','value'),
    State('plotdimensionisomap','value'),
    ]

    )
def IsomapAnalysis(n_clicks,featureselection,reference,n_neighbors,plotdimension):
    print(n_clicks)
    if n_clicks>0: 
        isomap=IsomapAlgo(featureselection,reference,n_neighbors,plotdimension)
        return isomap.getIsomapAnalysis()
    else :
        return ['']
        

#lock one field and unlock other one based on selected filed
@app.callback(
        [Output('selection','children'),Output('buttonspace','children'),],
        Input(component_id='setcreation',component_property='value'),
        State(component_id='algorithm', component_property='value'),
        )
def SelectOne(setcreation,algorithm):
    df=data.dataGuru.getDF()
    options=df.columns
    options=[{'label':i,'value':i} for i in options]
    
    x,y=df.shape
    fvalue=False
    svalue=True
    if setcreation!='cross':
        fvalue=True 
        svalue=False
    
    
    folds=[{'label':i,'value':i} for i in range(1,x//2)]
    percentage=[{'label':i,'value':i} for i in range(1,99)]
        
        
		# {'label': 'Machine Learning: SVM','value':'svm_VP'},
		# {'label': 'Machine Learning: Logistic Regression','value':'logreg_VP'},
		# {'label': 'Machine Learning: Decision Trees','value':'dt_VP'},
		# {'label': 'Machine Learning: Random Forest','value':'rf_VP'},
		# {'label': 'Machine Learning: AutoML','value':'automl_VP'},
		# {'label': 'Machine Learning: Artificial Neural Networks','value':'ann_VP'},

            
    if algorithm == 'dt_VP':
        idvalue='generate-decisiontree-analysis'
        buttonvalue='Decision Tree Analysis'
    if algorithm == 'rf_VP':
        idvalue='generate-randomforest-analysis'
        buttonvalue='RandomForest Analysis'
    if algorithm == 'automl_VP':
        idvalue='generate-automl-analysis'
        buttonvalue='AutoML Analysis'
    if algorithm == 'ann_VP':
        idvalue='generate-artificialneuralnetworks-analysis'
        buttonvalue='Artificial Neural Networks Analysis'
    if algorithm == 'svm_VP':
        idvalue='generate-svm-analysis'
        buttonvalue='SVM Analysis'
    if algorithm == 'logreg_VP':
        idvalue='generate-logisticregression-analysis'
        buttonvalue='Logistic Regression Analysis'
    
    
    return [html.Div(       drc.NamedDropdown(name="Folds",
                            id="numfolds",                                            
                            clearable=True,
                            searchable=True,
                            options=folds,
                            value=10,
                            multi=False,
                            disabled=fvalue
                            ),),
                            html.Div(
                            drc.NamedDropdown(name="%",
                            id="splitvalue",                                            
                            clearable=True,
                            searchable=True,
                            options=percentage,
                            value=70,
                            multi=False,
                            disabled=svalue,
                            ),),],  html.Button(buttonvalue, id=idvalue,n_clicks=0)

# We can use input type instand of dropdown
    # [html.Div(        dcc.Input(
    #             id="foldvalue", type="number", placeholder="folds",
    #             min=1, max=x//2, step=1,disabled=fvalue,
    #         ),),
    #                             html.Div(
    #                      dcc.Input(
    #             id="splitvlaue", type="number", placeholder="%",
    #             min=1, max=100, step=1,disabled=svalue,
    #         ),),]



#Decision Tree ANALYSIS        
@app.callback(
    Output('output_decisiontree_analysis','children'),
    [
    Input('generate-decisiontree-analysis','n_clicks'),
    ],
    
    [
    State('featureselection','value'),
    State('reference','value'),
    State('numfolds','value'),
    State('splitvalue','value'),
    State('setcreation','value'),
    State('max_depth','value')
    ]

    )
def DecisionTreeAnalysis(n_clicks,featureselection,reference,numfolds,splitvalue,setcreation,max_depth):
    
    
    print(n_clicks)
    if n_clicks>0:
        decisiontree=DecisionTreeAlgo(featureselection,reference,numfolds,splitvalue,setcreation,max_depth)
        return decisiontree.getDecisionTreeAnalysis()
    else :
        return ['']
        
#RandomForest ANALYSIS        
@app.callback(
    Output('output_randomforest_analysis','children'),
    [
    Input('generate-randomforest-analysis','n_clicks'),
    ],
    
    [
    State('featureselection','value'),
    State('reference','value'),
    State('numfolds','value'),
    State('splitvalue','value'),
    State('setcreation','value'),
    State('n_estimators','value'),
    State('max_depth','value'),
    
    ]

    )
def RandomForestAnalysis(n_clicks,featureselection,reference,numfolds,splitvalue,setcreation,n_estimators,max_depth):
    
    
    print(n_clicks)
    if n_clicks>0:
        randomforest=RandomForestAlgo(featureselection,reference,numfolds,splitvalue,setcreation,n_estimators,max_depth)
        return randomforest.getRandomForestAnalysis()
    else :
        return ['']
    
    
#SVM ANALYSIS        
@app.callback(
    Output('output_svm_analysis','children'),
    [
    Input('generate-svm-analysis','n_clicks'),
    ],
    
    [
    State('featureselection','value'),
    State('reference','value'),
    State('numfolds','value'),
    State('splitvalue','value'),
    State('setcreation','value'),
    State('kernel','value'),
    State('degree','value')
    
    ]

    )
def SVMAnalysis(n_clicks,featureselection,reference,numfolds,splitvalue,setcreation,kernel,degree):
    
    
    print(n_clicks)
    if n_clicks>0:
        svmAlgo=SVMAlgo(featureselection,reference,numfolds,splitvalue,setcreation,kernel,degree)
        return svmAlgo.getSVMAnalysis()
    else :
        return ['']



#AutoML ANALYSIS        
@app.callback(
    Output('output_automl_analysis','children'),
    [
    Input('generate-automl-analysis','n_clicks'),
    ],
    
    [
    State('featureselection','value'),
    State('reference','value'),
    State('numfolds','value'),
    State('splitvalue','value'),
    State('setcreation','value')
    ]

    )
def DecisionTreeAnalysis(n_clicks,featureselection,reference,numfolds,splitvalue,setcreation):
    
    
    print(n_clicks)
    if n_clicks>0:
        automl=AutoMLAlgo(featureselection,reference,numfolds,splitvalue,setcreation)
        return automl.getAutoMLAnalysis()
    else :
        return ['']


#Artificial Neural Networks ANALYSIS        
@app.callback(
    Output('output_artificialneuralnetworks_analysis','children'),
    [
    Input('generate-artificialneuralnetworks-analysis','n_clicks'),
    ],
    
    [
    State('featureselection','value'),
    State('reference','value'),
    State('numfolds','value'),
    State('splitvalue','value'),
    State('setcreation','value')
    ]

    )
def ArtificialNeuralNetworksAnalysis(n_clicks,featureselection,reference,numfolds,splitvalue,setcreation):
    
    print(n_clicks)
    if n_clicks>0:
        artificialneuralnetworks=ArtificialNeuralNetworksAlgo(featureselection,reference,numfolds,splitvalue,setcreation)
        return artificialneuralnetworks.getArtificialNeuralNetworksAnalysis()
    else :
        return ['']


#Logistic Regression ANALYSIS        
@app.callback(
    Output('output_logisticregression_analysis','children'),
    [
    Input('generate-logisticregression-analysis','n_clicks'),
    ],
    
    [
    State('featureselection','value'),
    State('reference','value'),
    State('numfolds','value'),
    State('splitvalue','value'),
    State('setcreation','value')
    ]

    )
def LogisticRegressionAnalysis(n_clicks,featureselection,reference,numfolds,splitvalue,setcreation):
    
    
    print(n_clicks)
    if n_clicks>0:
        logisticregression=LogisticRegressionAlgo(featureselection,reference,numfolds,splitvalue,setcreation)
        return logisticregression.getLogisticRegressionAnalysis()
    else :
        return ['']


#scatterplot    
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
        df = data.dataGuru.getDF()
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
        df = data.dataGuru.getDF()
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

# Visualization Implementation
@app.callback(
    [Output('vis_table','children'),Output('vis_splom','children')   
   ], [ 
    Input('generate-visulization','n_clicks') ,
    ],
    [State('featureselectionvis','value'),State('referencevis','value'),State('upload-data', 'filename'),],
    )
def getvisulization(n_clicks,featureselection,referencevis,filename):
    print(n_clicks)
    if n_clicks!= None and n_clicks>0 and featureselection!=None:
        df = data.dataGuru.getDF()
        DataPre=DataPreprocessAlgo()
        vistable,fig=DataPre.getdescri(featureselection,referencevis,filename)
        #print(vistable)
        #fig = px.scatter_matrix(df)
        #fig.show()
        return [vistable ,fig]
    else :
        return ['','']


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
