import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table


class BlandAltmanAlgo:
    
    def __init__(self,data,method1,method2):
        self.df=data
        self.method1=method1
        self.method2=method2
        
        
    def getAnswer(self):    
        
        #check methods are set or not
        df2 = pd.DataFrame()
        
        if self.method1==None:
            return html.B('please select feature1')
        
        if self.method2==None:
            return html.B('please select feature2')
        
        flag=0
        
        for j in self.df[self.method1]:
            if isinstance(j, str):
                stringcontaingfeature=self.method1
                flag=1
                break 
            if flag==1:
                break

        if flag==1:
            return html.B('string value is not allowed in feature :'+str(stringcontaingfeature))

        for j in self.df[self.method2]:
            if isinstance(j, str):
                stringcontaingfeature=self.method2
                flag=1
                break 
            if flag==1:
                break

        if flag==1:
            return html.B('string value is not allowed in feature :'+str(stringcontaingfeature))
        
        if self.method1!=None and self.method2!=None:
            
            #conver to numpy array
            m1= np.asarray(self.df[self.method1])
            m2= np.asarray(self.df[self.method2])
            
            df2['m1'] = self.df[self.method1]
            df2['m2'] = self.df[self.method2]
            
            mean = np.mean([m1, m2], axis=0)
            diff = m1 - m2                         # Difference between m1 and m2
            md =np.single(np.mean(diff))           # Mean of the difference
            sd = np.std(diff, axis=0)              # Standard deviation of the difference
            
            df2['mean']=mean
            df2['diff']=diff
            
            md1=np.single(md + 1.96*sd)
            md2=np.single(md - 1.96*sd)
            
            p1=min(mean)-1
            p2=max(mean)+1
            
            fig = go.Figure()
            fig=px.scatter( df2,x="mean", y="diff",labels={'mean':'Mean of ' + str(self.method1)+' and '+str(self.method2),'diff':'('+str(self.method1)+' - '+str(self.method2)+')',},title="Bland Altman Plot")

            #add line to figure
            fig.add_shape(type="line",
                y0=md,y1=md,x0=p1,x1=p2,
                line=dict(color="LightSeaGreen",dash="dashdot",))
            fig.add_shape(type="line",
                y0=md1,y1=md1,x0=p1,x1=p2,
                line=dict(color="LightSeaGreen",dash="dashdot",))          
            fig.add_shape(type="line",
                y0=md2,y1=md2,x0=p1,x1=p2,
                line=dict(color="LightSeaGreen",dash="dashdot",))
            
            
            # Create scatter trace of text labels
            fig.add_trace(go.Scatter(
                x=[p2,p2,p2],
                y=[md+0.5,md1+0.5,md2+0.5   ],
                text=["mean diff","+SD 1.96 : "+str(md1),"-SD 1.96 : "+str(md2)],
                mode="text",
            ))
            
            fig.update_shapes()
            
            return html.Div([
                dcc.Graph(
                id='BlandAltmanPlot',
                figure=fig,
                )
                ])


        else:
            return html.B('please try again') 
        
