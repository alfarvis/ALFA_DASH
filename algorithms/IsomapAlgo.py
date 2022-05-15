import pandas as pd
import numpy as np

import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from sklearn import utils
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from algorithms import data

class IsomapAlgo:
    
    def __init__(self,features,reference,n_neighbors,plotdimension,globalData):
        self.reference=reference
        self.features=features  
        self.df3=globalData.dataGuru.getDF()
        self.df=pd.DataFrame()
        self.n_neighbors=n_neighbors
        self.plotdimension=plotdimension
        
        if features!=None:
            for i in features:
                self.df[i]=self.df3[i]
        if reference!=None:
            self.df[reference]=self.df3[reference]

                
    def getIsomapAnalysis(self):            


        #check reference is set or not
        df2 = pd.DataFrame()
        if self.reference!=None and self.reference in self.df.keys():
            df2['reference000'] = self.df[self.reference]
            self.df.drop(self.reference,  axis='columns', inplace=True)
        else:
            df2['reference000']=[] 
        
        #check reference is selected or not
        if self.reference==None:
            return [html.B('Please select reference')]
         
        #check features are selected or not
        if self.features==[] or self.features==None:
            return [html.B('Please select features')]
                
        flag=0
        
        for i in self.features:
            if i==self.reference:
                flag=1
                break 
        if flag==1:
            return [html.B('please do not select reference in features')]

        
        # #check features is continious or not
        # for i in self.features:
        #     if str(utils.multiclass.type_of_target(self.df[i]))!='continuous':
        #         flag=1
        #         break
        # if flag==1:
        #     return [html.B(str(i)+' is not Continuos variable')]           
                    
        
        #check reference is not selected in features     
        
        #check features contain string value or not            
        for i in self.features:
            for j in self.df[str(i)]:
                if isinstance(j, str):
                    stringcontaingfeature=i
                    flag=1
                    break 
            if flag==1:
                break

        if flag==1:
            return [html.B('string value is not allowed in feature :'+str(stringcontaingfeature))]
        
        #check n_neighbors is selected or not
        if self.n_neighbors==None:
            self.n_neighbors=5
            
        #check plotdimesion is selected or not
        if self.plotdimension==None:
            self.plotdimension=2
        
        #scalling the data
        scalar = StandardScaler()
        scalar.fit(self.df)
        self.df = scalar.transform(self.df)

        
        #find the number of components
        n,m=self.df.shape
        n_component=min(n,m)
        possible_pc='Possible Number of Components is : '+str(n_component)                
        
        #number of components is less than 2 then
        if n_component<2:
            return [html.B(possible_pc),html.B(),html.B()]        
        
        else:
        
            #fitting
            x_pca = manifold.Isomap(n_components=self.plotdimension,n_neighbors=self.n_neighbors).fit_transform(self.df)
            pc=['Component'+str(i) for i in range(n_component)]
            
            # scatterplot be    tween pc1 and pc2 
            pca_df=pd.DataFrame()
            pca_df['Component1']=x_pca[:,0]
            pca_df['Component2']=x_pca[:,1]
            pca_df['reference']=list(df2['reference000'])
            
            
            if n_component>2 and self.plotdimension==3 :
                    pca_df['Component3']=x_pca[:,2]
                    fig1 = px.scatter_3d(pca_df,x=pca_df['Component1'], y=pca_df['Component2'],z=pca_df['Component3'],color=pca_df['reference'], title="ISOMAP Graph")    
                    plotdis="Here 3D ISOMAP between component1, component2 and component3 is"
            else:
                fig1 = px.scatter(pca_df,x=pca_df['Component1'], y=pca_df['Component2'],color=pca_df['reference'], title="ISOMAP Graph")    
                plotdis="Here 2D ISOMAP between component1 and component2 is"
                            
            return [
                
                #return possible number of pc
                html.Div((html.Br(),html.Div(plotdis),
                
                #return 2D scatterplot between pc1 and pc2          
                dcc.Graph(
                id='example-graph',
                figure=fig1,
                )))
                ]