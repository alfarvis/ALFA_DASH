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
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
            

class RandomForestAlgo:
        
    def __init__(self,featureselection,reference,numfolds,splitvalue,setcreation,n_estimators,max_depth):
        self.reference=reference
        self.features=featureselection
        self.numfolds=numfolds
        self.splitvalue=splitvalue
        self.setcreation=setcreation
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.df3=data.dataGuru.getDF()
        features=self.features
        self.df=pd.DataFrame()
        if features!=None:
            for i in features:
                self.df[i]=self.df3[i]
        if reference!=None:
            self.df[reference]=self.df3[reference]
                    
                
    def getRandomForestAnalysis(self):    
        
        #check reference is set or not
        df2 = pd.DataFrame()
        if self.reference!=None and self.reference in self.df.keys():
            df2['reference'] = self.df[self.reference]
            self.df.drop(self.reference,  axis='columns', inplace=True)
        
        #check reference is selected or not
        if self.reference==None:
            return [html.B('Please select reference')]
        
        #check reference is continious or not
        if str(utils.multiclass.type_of_target(df2.reference))=='continuous':
            return [html.B('Continuos reference variable is not acceptable')]           
        
        #check features are selected or not
        if self.features==[] or self.features==None:
            return [html.B('Please select features')]
        
        #check fold value is selected or not
        if self.setcreation=='cross' and self.numfolds==None:
            return [html.B('Please select fold value')]
        
        if self.n_estimators==None:
            self.n_estimators=100
            
        
        #check split value is selected or not
        if self.setcreation=='percentage' and self.splitvalue==None:
            return [html.B('Please select split percentage value')]    
        
 

        
        labels=df2.reference.unique()
        scaler = preprocessing.MinMaxScaler()

        if self.setcreation=='percentage':
            X, y = self.df,df2['reference']
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.splitvalue/100)
            sc1 = scaler.fit(X_train)
            X_train = sc1.transform(X_train)
            X_test = sc1.transform(X_test)
            
            
            clf_gini = RandomForestClassifier(criterion = "gini",n_estimators=self.n_estimators,max_depth=self.max_depth)          
            # Performing training
            clf_gini.fit(X_train, y_train)
            y_pred = clf_gini.predict(X_test)
            
            mat=confusion_matrix(y_test, y_pred)
            #mat=mat.tolist()
            con=pd.DataFrame(mat,columns=labels)
            con.insert(0, column='*', value=labels)
            cm1=dash_table.DataTable(
                id='table',
                columns=[{"name": str(i), "id": str(i)} for i in con.columns],
                data=con.to_dict('records'),
                )
            
            ac1=html.Div('Accuracy : '+str(accuracy_score(y_test,y_pred)*100))
            report1=classification_report(y_test, y_pred,output_dict=True,digits=2)
            report1=pd.DataFrame.from_dict(report1)
            report1=report1.round(decimals=2)
            report1.insert(0,"*",[str(i) for i in report1.index], True)
            re1=dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in report1.columns],
                data=report1.to_dict('records'),
                )
            
            
            
            clf_entropy = RandomForestClassifier(criterion = "entropy",n_estimators=self.n_estimators,max_depth=self.max_depth)
            # Performing training
            clf_entropy.fit(X_train, y_train)
            y1_pred = clf_entropy.predict(X_test)
                   
            mat=confusion_matrix(y_test, y1_pred)
            con=pd.DataFrame(mat,columns=labels)
            con.insert(0, column='*', value=labels)
            cm2=dash_table.DataTable(
                id='table',
                columns=[{"name": str(i), "id": str(i)} for i in con.columns],
                data=con.to_dict('records'),
                )
            ac2=html.Div("Accuracy : "+str(accuracy_score(y_test,y1_pred)*100))
            
            report2=classification_report(y_test, y1_pred,output_dict=True,digits=2)
            report2=pd.DataFrame.from_dict(report2)
            report2.insert(0,"*",[str(i) for i in report2.index], True)
            report2=report2.round(decimals=2)

            re2=dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in report2.columns],
                data=report2.to_dict('records'),
                )
            return html.Div(children=[html.Br(),html.B('Results Using Gini Index:'),html.Br(),html.Br(),html.B('Confusion Matrix'),cm1,html.Br(),ac1,html.Br(),html.B('Report : '),re1,html.Br(),html.Br(),html.B('Results Using Entropy:'),html.Br(),html.Br(),html.B('Confusion Matrix'),html.Br(),cm2,html.Br(),ac2,html.Br(),html.B('Report :'),re2])
    
        if self.setcreation=='cross':

            X, y = self.df,df2['reference']
            X=X.to_numpy()
            clf_gini = RandomForestClassifier(criterion = "gini",n_estimators=self.n_estimators,max_depth=self.max_depth)          
            clf_entropy = RandomForestClassifier(criterion = "entropy",n_estimators=self.n_estimators,max_depth=self.max_depth)
            skf = StratifiedKFold(n_splits=self.numfolds, shuffle=True, random_state=1)
            lst_accu_stratified1 = []
            lst_accu_stratified2 = []
            yy1_pred=[1]
            yy1_test=[1]
            yy2_pred=[1]
            yy2_test=[1]
            for train_index, test_index in skf.split(X,y):
                x_train_fold, x_test_fold = X[train_index], X[test_index]
                sc1 = scaler.fit(x_train_fold)
                x_train_fold = sc1.transform(x_train_fold)
                x_test_fold = sc1.transform(x_test_fold)
                y_train_fold, y_test_fold = y[train_index], y[test_index]
                clf_gini.fit(x_train_fold, y_train_fold)
                clf_entropy.fit(x_train_fold,y_train_fold)
                # a=clf_gini.predict(x_test_fold)
                # print(len(list(a)))
                yy1_pred=yy1_pred+list(clf_gini.predict(x_test_fold))
                yy1_test=yy1_test+list(y_test_fold)
                yy2_pred=yy2_pred+list(clf_entropy.predict(x_test_fold))
                yy2_test=yy2_test+list(y_test_fold)
                lst_accu_stratified1.append(clf_gini.score(x_test_fold, y_test_fold))
                lst_accu_stratified2.append(clf_entropy.score(x_test_fold, y_test_fold))
            
            
            lst_accu_stratified1=[ round(elem,4) for elem in lst_accu_stratified1 ]
            lst_accu_stratified2=[ round(elem,4) for elem in lst_accu_stratified2 ]
            # print(len(yy_pred),1212)
            # print(len(yy_test),1212)
            # #print(yy_pred)
            # Print the output.
            ans=[]
            # scores = cross_val_score(clf_gini, X, y, cv=10)
            # y_pred = cross_val_predict(clf_gini, X, y, cv=10)
            # print(len(y_pred))
            # print(accuracy_score(y,y_pred)*100)
            # print(scores.mean())
            #print(abcd)
            ans.append(html.Br())
            ans.append(html.H1(children=[html.B('Results Using Gini Index:')]))
            ans.append(html.Br())
            ans.append(html.Br())
            ans.append(html.B('List of possible accuracy:'))
            folds=dict()
            for i in range(self.numfolds):
                fkey='fold'+str(i+1)
                folds[fkey]=[]
                folds[fkey].append(str(lst_accu_stratified1[i]*100)+'%')
            fcolumns=[{'name':'fold'+str(i+1),'id':'fold'+str(i+1)} for i in range(self.numfolds)]
            folds=pd.DataFrame.from_dict(folds)
            ftable=dash_table.DataTable(
                id='foldtable',
                columns=fcolumns,
                data=folds.to_dict('records'),
                style_table={'height': 'auto', 'overflowY': 'auto','minWidth': '100%'},

                )
            #print(folds)
            ans.append(html.Div(children=[ftable]))
            #ans.append(html.P(' '.join(str(i) for i in lst_accu_stratified)))
            ans.append(html.Br())
            ans.append(html.B('\nMaximum Accuracy That can be obtained from this model is: '+str(max(lst_accu_stratified1)*100)+'%'))
            ans.append(html.Br())
            ans.append(html.B('\nMinimum Accuracy: '+str(min(lst_accu_stratified1)*100)+'%'))
            ans.append(html.Br())
            ans.append(html.B('\nOverall Accuracy: '+str(round(mean(lst_accu_stratified1)*100,2))+'%'))
            ans.append(html.Br())
            ans.append(html.B('\nStandard Deviation is: '+str(round(stdev(lst_accu_stratified1),4))))
            
            yy1_test.pop(0)
            yy1_pred.pop(0)      
            mat=confusion_matrix(yy1_test, yy1_pred)
            con=pd.DataFrame(mat,columns=labels)
            con.insert(0, column='*', value=labels)
            cm1=dash_table.DataTable(
                id='table',
                columns=[{"name": str(i), "id": str(i)} for i in con.columns],
                data=con.to_dict('records'),
                )
            report1=classification_report(yy1_test, yy1_pred,output_dict=True,digits=2)
            report1=pd.DataFrame.from_dict(report1)
            report1.insert(0,"*",[str(i) for i in report1.index], True)
            report1=report1.round(decimals=2)

            re1=dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in report1.columns],
                data=report1.to_dict('records'),
                )
            ans.append(html.Div(children=[html.Br(),html.Br(),html.B('Confusion Matrix'),cm1,html.Br(),html.Br(),html.B('Report : '),re1,html.Br(),html.Br()]))
        
            ans.append(html.Br())
            ans.append(html.H1(children=[html.B('Results Using Entropy:')]))
            ans.append(html.Br())
            ans.append(html.Br())
            ans.append(html.B('List of possible accuracy:'))
            folds=dict()
            for i in range(self.numfolds):
                fkey='fold'+str(i+1)
                folds[fkey]=[]
                folds[fkey].append(str(lst_accu_stratified2[i]*100)+'%')
            fcolumns=[{'name':'fold'+str(i+1),'id':'fold'+str(i+1)} for i in range(self.numfolds)]
            folds=pd.DataFrame.from_dict(folds)
            ftable=dash_table.DataTable(
                id='foldtable2',
                columns=fcolumns,
                data=folds.to_dict('records'),
                style_table={'height': 'auto', 'overflowY': 'auto','minWidth': '100%'},

                )
            #print(folds)
            ans.append(html.Div(children=[ftable]))
            #ans.append(html.P(' '.join(str(i) for i in lst_accu_stratified)))
            ans.append(html.Br())
            ans.append(html.B('\nMaximum Accuracy That can be obtained from this model is: '+str(max(lst_accu_stratified2)*100)+'%'))
            ans.append(html.Br())
            ans.append(html.B('\nMinimum Accuracy: '+str(min(lst_accu_stratified2)*100)+'%'))
            ans.append(html.Br())
            ans.append(html.B('\nOverall Accuracy: '+str(round(mean(lst_accu_stratified2)*100,2))+'%'))
            ans.append(html.Br())
            ans.append(html.B('\nStandard Deviation is: '+str(round(stdev(lst_accu_stratified2),4))))
            
            yy2_test.pop(0)
            yy2_pred.pop(0)      
            mat=confusion_matrix(yy2_test, yy2_pred)
            con=pd.DataFrame(mat,columns=labels)
            con.insert(0, column='*', value=labels)
            cm1=dash_table.DataTable(
                id='table',
                columns=[{"name": str(i), "id": str(i)} for i in con.columns],
                data=con.to_dict('records'),
                )
            report1=classification_report(yy2_test, yy2_pred,output_dict=True,digits=2)
            report1=pd.DataFrame.from_dict(report1)
            report1.insert(0,"*",[str(i) for i in report1.index], True)
            report1=report1.round(decimals=2)

            re1=dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in report1.columns],
                data=report1.to_dict('records'),
                )
            ans.append(html.Div(children=[html.Br(),html.Br(),html.B('Confusion Matrix'),cm1,html.Br(),html.Br(),html.B('Report : '),re1,html.Br(),html.Br()]))
                
            return ans
        
        