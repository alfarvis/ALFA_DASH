import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import losses 
from keras import optimizers 
from keras import metrics 
from keras.callbacks import TensorBoard

import plotly.express as px
import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from statistics import mean, stdev
            
class ArtificialNeuralNetworksAlgo:
        
    def __init__(self,featureselection,reference,numfolds,splitvalue,setcreation,globalData):
        #set selected reference
        self.reference=reference
        #set selected list of features
        self.features=featureselection
        #set the value of folds
        self.numfolds=numfolds
        #set the value of percentage split for traing 
        self.splitvalue=splitvalue
        #set the choice b/w cross validation and percentage split
        self.setcreation=setcreation
        
        #set the selected dataset
        self.df3=globalData.dataGuru.getDF()
        features=self.features
        self.df=pd.DataFrame()
        if features!=None:
            for i in features:
                if i!=reference:                
                    self.df[i]=self.df3[i]
        if reference!=None:
            self.df[reference]=self.df3[reference]
            
    
    #function to add layer
    def addlayer(self,globalData,model,layer,activationfunction,layercnt):
        
        #set the layer name with it layer number
        layname="Layer"+str(layercnt)
        
        #add layer to the model
        model.add(layers.Dense(layer, activation=activationfunction,name=layname))
        
        #set model to make accessable from any in platform
        globalData.dataGuru.setMODEL(model) 
        print(model.summary())

        #process to return model summary in html component
        stringlist = []
        stringlist.append('->Layer '+str(layercnt)+' is added in MODEL')
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = [html.Div(i) for i in stringlist]
        return (short_model_summary )     
    
    def deletelayer(self,globalData,model,layercnt):
        
        #delete last layer from the model
        model.pop()
        
        #if only one layer in the model then
        if layercnt==0:
            return '->Model Layer '+str(layercnt)+' is deleted' 
            #set model to none
            globalData.dataGuru.setMODEL(None)
        
        else:
            #set model
            globalData.dataGuru.setMODEL(model)
            print(model.summary())

            #process to return model summary in html component
            stringlist = []
            stringlist.append('->Layer '+str(layercnt)+' is deleted in MODEL')
            model.summary(print_fn=lambda x: stringlist.append(x))
            short_model_summary = [html.Div(i) for i in stringlist]
            return (short_model_summary )
    
    
    def getArtificialNeuralNetworksAnalysis(self,globalData,model):    
                                
        #check reference is set or not
        df2 = pd.DataFrame()
        df4 = pd.DataFrame()
        df4=self.df.copy()
        if self.reference!=None and self.reference in self.df.keys():
            df2['reference000'] = self.df[self.reference]
            df4.drop(self.reference,  axis='columns', inplace=True)
        
        #check reference is selected or not        
        if self.reference==None:
            return [html.B('Please select reference')]
        
        #check model have layer or not
        if model==None:
            return [html.B('Please add layer to model')]            
        
        #check features are selected or not
        if self.features==[] or self.features==None:
            return [html.B('Please select features')]
        
        #check fold value is selected or not
        if self.setcreation=='cross' and self.numfolds==None:
            return [html.B('Please select fold value')]    
        
        #check split value is selected or not
        if self.setcreation=='percentage' and self.splitvalue==None:
            return [html.B('Please select split percentage value')]    
 
        flag=0
        
        #check reference contain string value or not
        for i in  df2['reference000']:
            if isinstance(i, str):
                flag=1
                break 
            
        if flag==1:
            return [html.B('string value is not allowed in reference')]
        
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
            return [html.B('string value is not allowed in feature :'+str(stringcontaingfeature))]
                    
        #get class or labels from the referece    
        labels=df2.reference000.unique()
        labels.sort()
        
        #check fold value is less than min
        referencelst=list(df2.reference000)
        for i in labels:
            classcnt=referencelst.count(i)
            if classcnt<self.numfolds:
                flag=1
                break 
        if flag==1:
            return [html.B('ValueError: Value of folds = '+str(self.numfolds)+' cannot be greater than the number of members in each class.')]

        if model!=None and 'Output_Layer':
            try:
                model.add(layers.Dense(len(labels), activation='softmax',name='Output_Layer'))
            except:
                print('here')
        if self.setcreation=='percentage':
            
            X, y = df4,df2['reference000']

            # One hot encoding
            enc = OneHotEncoder()
            y = enc.fit_transform(y[:, np.newaxis]).toarray()
            
            # Scale data to have mean 0 and variance 1 
            # which is importance for convergence of the neural network
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.splitvalue/100)
            
            #if only one label got for training classifiation
            yy_train=labels[[np.where(r==1)[0][0] for r in y_train]]
            
            yy_trian=set(yy_train)
            if len(yy_trian)<=1:
                return html.B('Please select higher value of percentage split')
                        
            #compile model using loss function as mean_squared_error and optimizer sgd
            model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = [metrics.categorical_accuracy])
            print('Model name:', model.name)
            
            #fitting model
            model.fit(X_train, y_train,
                                        batch_size=5,
                                        epochs=50,                                        verbose=0,
                                        validation_data=(X_test, y_test),
                                        )
            #to convert y_pred in 1d 
            y_pred=labels[model.predict(X_test).argmax(axis=1)]
            score = model.evaluate(X_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            model.pop()
            
            #to conver y_test in 1d
            y_test=labels[ [np.where(r==1)[0][0] for r in y_test]]
            
            #to create confusion matrix
            mat=confusion_matrix(y_test, y_pred,labels)
            con=pd.DataFrame(mat,columns=labels)
            con.insert(0, column='*', value=labels)
            cm1=dash_table.DataTable(
                id='table',
                columns=[{"name": str(i), "id": str(i)} for i in con.columns],
                data=con.to_dict('records'),
                )
            
            ac1=html.Div('Accuracy : '+str(accuracy_score(y_test,y_pred)*100))
            
            #to create report
            report1=classification_report(y_test, y_pred,output_dict=True,digits=2)
            report1=pd.DataFrame.from_dict(report1)
            report1=report1.round(decimals=2)
            report1.insert(0,"*",[str(i) for i in report1.index], True)
            re1=dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in report1.columns],
                data=report1.to_dict('records'),
                )
                        
            return html.Div(children=[html.Br(),html.B('Results Using ANN:'),html.Br(),html.Br(),html.B('Confusion Matrix'),cm1,html.Br(),ac1,html.Br(),html.B('Report : '),re1,html.Br(),html.Br(),html.Br(),])
    
        if self.setcreation=='cross':

            X, y = df4,df2['reference000']
            
            # One hot encoding
            enc = OneHotEncoder(categories=[labels])
            
            scaler = preprocessing.MinMaxScaler()
            x_scaled = scaler.fit_transform(X)
            
            model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = [metrics.categorical_accuracy])
            print('Model name:', model.name)
            
            skf = StratifiedKFold(n_splits=self.numfolds, shuffle=True, random_state=1)
            lst_accu_stratified1 = []
            lst_accu_stratified2 = []
            yy1_pred=[1]
            yy1_test=[1]
            yy2_pred=[1]
            yy2_test=[1]
            for train_index, test_index in skf.split(X,y):
                
                x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
                y_train_fold, y_test_fold = y[train_index], y[test_index]
                y_train_fold = enc.fit_transform(y_train_fold[:, np.newaxis]).toarray()
                yy_test_fold = enc.fit_transform(y_test_fold[:, np.newaxis]).toarray()

                model.fit(x_train_fold, y_train_fold,
                                        batch_size=5,
                                        epochs=50,                                        verbose=0,
                                        validation_data=(x_test_fold, yy_test_fold),
                                        )
            
                yy1_pred=yy1_pred+list(labels[model.predict(x_test_fold).argmax(axis=1)])
                yy1_test=yy1_test+list(y_test_fold)
                score = model.evaluate(x_test_fold, yy_test_fold, verbose=0)
                lst_accu_stratified1.append(score[1])
                lst_accu_stratified2.append(score[0])
            
            lst_accu_stratified1=[ round(elem,4) for elem in lst_accu_stratified1 ]
            lst_accu_stratified2=[ round(elem,4) for elem in lst_accu_stratified2 ]
            
            ans=[]
            ans.append(html.Br())
            ans.append(html.H1(children=[html.B('Results using Ann with cross validation:')]))
            ans.append(html.Br())
            ans.append(html.Br())
            ans.append(html.B('List of possible test accuracy:'))
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
            ans.append(html.Div(children=[ftable]))
            ans.append(html.Br())
            ans.append(html.B('\nMaximum Accuracy That can be obtained from this model is: '+str(max(lst_accu_stratified1)*100)+'%'))
            ans.append(html.Br())
            ans.append(html.B('\nMinimum Accuracy: '+str(min(lst_accu_stratified1)*100)+'%'))
            ans.append(html.Br())
            ans.append(html.B('\nOverall Accuracy: '+str(round(mean(lst_accu_stratified1)*100,2))+'%'))
            ans.append(html.Br())
            stdlst=[ elem*100 for elem in lst_accu_stratified1 ]
            ans.append(html.B('\nStandard Deviation is: '+str(round(stdev(stdlst)/100,4))))
            
            yy1_test.pop(0)
            yy1_pred.pop(0)      
            mat=confusion_matrix(yy1_test, yy1_pred,labels)
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
            ans.append(html.B('List of possible test loss:'))
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
            ans.append(html.B('\nMaximum loss That can be obtained from this model is: '+str(max(lst_accu_stratified2)*100)+'%'))
            ans.append(html.Br())
            ans.append(html.B('\nMinimum loss: '+str(min(lst_accu_stratified2)*100)+'%'))
            ans.append(html.Br())
            ans.append(html.B('\nOverall loss: '+str(round(mean(lst_accu_stratified2)*100,2))+'%'))
            ans.append(html.Br())
            stdlst=[ elem*100 for elem in lst_accu_stratified2 ]
            ans.append(html.B('\nStandard Deviation is: '+str(round(stdev(stdlst)/100,4))))
            yy2_test.pop(0)
            yy2_pred.pop(0)                      
            model.pop()
            return ans
        
        