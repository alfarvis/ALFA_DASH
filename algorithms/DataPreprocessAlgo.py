import imp
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
#from algorithms import data
#from algorithms import data
from algorithms import data
import plotly.graph_objs as go
import plotly.express as px


class DataPreprocessAlgo :
    
    def __init__(self):
        self.df =data.dataGuru.getDF()

    def getdescri(self,option,reference,datasetname):

        if option==None or option==[]:
            return html.Div('please select features'),''
        df=self.df
        df2=self.df.describe()
        print(option)
        #print([str(i) for i in df2.index])
        df3=pd.DataFrame()

        
        for i in list(option):
            df3[i]=df2[i]

        df3.insert(0, "",[str(i) for i in df2   .index], True)
        df3=df3.round(decimals=2)
        vistable=df3
        color=None
        index_vals=None
        if reference!=None:
            color=df[reference]
            index_vals = df[reference].astype('category').cat.codes

        dim=[dict(label=str(i),values=df[str(i)]) for i in option]
        print(index_vals)
        fig = go.Figure(data=go.Splom(
                dimensions=[dict(label=str(i),values=df[str(i)]) for i in option],
                text=index_vals,
                marker=dict(color=color,
                            showscale=False, # colors encode categorical variables
                            line_color='white', line_width=0.5)
                ))


        fig.update_layout(
            title=datasetname,
            dragmode='select',
            width=150*(max(len(option),1)),
            height=150*(max(len(option),1)),
            hovermode='closest',
        )
        #fig.show()
        
        return html.Div([dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in vistable.columns],
        data=vistable.to_dict('records'),      
        fixed_rows={'headers': True},
        fixed_columns={'headers': True, 'data': 1},

        style_table={'height': '400px', 'overflowY': 'auto','minWidth': '100%'},
        style_data={
        'color': 'black',
        'whiteSpace': 'normal',
        'height': 'auto',
        },
        style_data_conditional=[
        {
        'if': {'row_index': 'even'},
        'backgroundColor': 'rgb(195, 254, 173)',
        },
        {
        'if': {'row_index': 'odd'},
        'backgroundColor': 'rgb(182, 184, 221)',
        }
        ],
        style_header={
        'backgroundColor': 'rgb(155, 124, 94)',
        'color': 'black',
        'fontWeight': 'bold'
        },
        )],),html.Div(dcc.Graph(
                id='example-graph',
                figure=fig,))

    def getcolumns(self):
        return list(self.df.columns)
    
    