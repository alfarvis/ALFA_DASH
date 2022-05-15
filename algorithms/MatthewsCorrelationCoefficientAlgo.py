from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from algorithms import data


class MatthewsCorrelationCoefficientAlgo:
    
    def __init__(self,data1,features,reference,plotdimension):
        self.reference=reference
        self.df3=data1
        self.df=pd.DataFrame()
        self.plotdimension=plotdimension
        self.features=features

        # self.features=features
        #print(data.dataGuru.getDF())
        #print(features)
        #print(reference)
        if features!=None:
            for i in features:
                self.df[i]=self.df3[i]
        if reference!=None:
            self.df[reference]=self.df3[reference]
                    
        
        
    def getMatthewsCorrelationCoefficientAnalysis(self):    
        
        #check features are selected or not
        if self.features==[]:
            return [html.B('Please select features'),html.B(),html.B()]                
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
        scaled_data = scalar.transform(self.df)
        
        #find the number of pca
        n,m=scaled_data.shape
        n_component=min(n,m)
        possible_pc='Possible Number of PC is : '+str(n_component)                
        
        
        if n_component<2:
            return [html.B(possible_pc),html.B(),html.B()]        
        else:
            
            #fitting
            pca = PCA()
            pca.fit(scaled_data)
            x_pca = pca.transform(scaled_data)
            pc=['PC'+str(i) for i in range(n_component)]
            
            # scatterplot between pc1 and pc2 
            pca_df=pd.DataFrame()
            pca_df['reference']=df2['reference']
            pca_df['PC1']=x_pca[:,0]
            pca_df['PC2']=x_pca[:,1]
            if n_component>2 and self.plotdimension==3 :
                pca_df['PC3']=x_pca[:,2]
                fig1 = px.scatter_3d(pca_df,x=pca_df['PC1'], y=pca_df['PC2'],z=pca_df['PC3'],color=pca_df['reference'], title="PCA Graph")    
                plotdis="Here 3D scatter plot between pc1, pc2 and pc3 is"
            else:
                fig1 = px.scatter(pca_df,x=pca_df['PC1'], y=pca_df['PC2'],color=pca_df['reference'], title="PCA Graph")    
                plotdis="Here 2D scatter plot between pc1 and pc2 is"
            #For Percentage of Explained Variance Table 
            per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
            labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
            s_df=pd.DataFrame()
            s_df['Principal Component'] = labels
            s_df['Percentage of Explained Variance'] = per_var
            columns = [{"name": i, "id": i} for i in s_df.columns]
            data = s_df.to_dict('records')
            t = [columns,data]
            
            #PCA Scree Plot
            fig2 = px.bar(x=labels,y=per_var, labels={
                        'x':'Principal Component',
                        'y':'Percentage of Explained Variance',
                    },title="PCA Scree Plot")
            ans=[possible_pc,fig1,columns,data,fig2]
        
            return [
                #return possible number of pc
                html.Div([ html.H1(ans[0]),html.Br(),html.Div(plotdis),
                   
                #return 2D scatterplot between pc1 and pc2          
                dcc.Graph(
                id='example-graph',
                figure=ans[1],
                )
                ]),
                
                #return table Percentage of Explained Variance
                html.Div([dash_table.DataTable(
                id='table',
                columns=ans[2],
                data=ans[3],

                )]),
                
                #return the scree plot
                html.Div([
                dcc.Graph(
                id='example-graph_2',
                figure=ans[4],
                )
                ]),
                ]