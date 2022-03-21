from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from sklearn import manifold
from sklearn.preprocessing import StandardScaler



class IsomapAlgo:
    
    def __init__(self,data,features,reference,n_neighbors):
        self.reference=reference
        self.features=features  
        self.df3=data
        self.df=pd.DataFrame()
        self.n_neighbors=n_neighbors
        for i in features:
            self.df[i]=self.df3[i]
        if reference!=None:
            self.df[reference]=self.df3[reference]
        
        
        
        
    def getIsomapAnalysis(self):    
        
        #check reference is set or not
        df2 = pd.DataFrame()
        if self.reference!=None and self.reference in self.df.keys():
            df2['reference'] = self.df[self.reference]
            self.df.drop(self.reference,  axis='columns', inplace=True)
        else:
            df2['reference']=[] 
        
        #scalling the data
        scalar = StandardScaler()
        scalar.fit(self.df)
        self.df = scalar.transform(self.df)

        
        #find the number of components
        n,m=self.df.shape
        n_component=min(n,m)
        possible_pc='Possible Number of Components is : '+str(n_component)                
        
        
        if n_component<2:
            return [html.B(possible_pc),html.B(),html.B()]        
            
        else:
            
            #fitting
            x_pca = manifold.Isomap(n_components=2,n_neighbors=self.n_neighbors).fit_transform(self.df)
            pc=['Component'+str(i) for i in range(n_component)]
            
            # scatterplot be    tween pc1 and pc2 
            pca_df=pd.DataFrame()
            pca_df['Component1']=x_pca[:,0]
            pca_df['Component2']=x_pca[:,1]
            pca_df['reference']=df2['reference']
            fig1 = px.scatter(pca_df,x=pca_df['Component1'], y=pca_df['Component2'],color=pca_df['reference'], title="ISOMAP Graph")    

            
            return [
                #return possible number of pc
                html.Div((html.Br(),html.Div("Here 2D ISOMAP between component1 and component2 is"),
                #return 2D scatterplot between pc1 and pc2          
                dcc.Graph(
                id='example-graph',
                figure=fig1,
                )))
                ]