from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

import plotly.express as px
import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import time


class TsneAlgo:
    
    def __init__(self,dataset,features,reference,perplexity,iteration):
        self.features=features
        self.reference=reference
        self.df3=dataset
        self.perplexity=perplexity
        self.iteration=iteration
        self.df=pd.DataFrame()
        
        if features!=None:
            for i in features:
                self.df[i]=self.df3[i]
        if reference!=None:
            self.df[reference]=self.df3[reference]    
    
    def getAnswer(self):    

        #check values are set or not
        if self.features==[] or self.features==None:
            return html.Div('Please select at least one feature')
        if self.reference==None:
            return html.Div('Please select reference')
        if self.perplexity==None:
            return html.Div('Please select value of perplexity')
        if self.iteration==None:
            return html.Div('Please select value of iteration')

        flag =0

        #check reference is not selected in features 
        for i in self.features:
            if i==self.reference:
                flag=1
                break 
        if flag==1:
            return [html.B('please do not select reference in features')]

        
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
            return html.B('string value is not allowed in feature :'+str(stringcontaingfeature))


        
        
        df2 = pd.DataFrame()
        if self.reference!=None and self.perplexity!=None and self.iteration!=None  :
            
            reference=self.reference
            df=self.df
            df3=df.copy()
            p=self.perplexity
            iteration=self.iteration
            df2['reference'] = self.df[reference]

            df3.drop(self.reference,  axis='columns', inplace=True)
                
            time_start = time.time()
            tsne_data = TSNE(n_components=2,random_state=1,perplexity=p,n_iter=iteration).fit_transform(df3)
            timetaken='t-SNE done! Time elapsed: '+ str(format(time.time()-time_start))+' seconds'
            tsne_data=np.vstack((tsne_data.T,df2['reference'])).T
            tsne_df=pd.DataFrame(data=tsne_data,columns=('Dim_1','Dim_2','label'))
            fig=px.scatter( tsne_df,x="Dim_1", y="Dim_2",color='label',title="Tsne 2-D Plot")
            
            return   [
                
                #time taken
                html.B(timetaken),
                html.Br(),
                
                #return possible number of pc
                html.Div([html.Div("Here 2D scatter plot between dim1 and dim2 is"),
                   
                #return 2D scatterplot between pc1 and pc2          
                dcc.Graph(
                id='tsne-plot',
                figure=fig,
                )
                ])]
                
        else:
            return html.B('please try again')
        
        