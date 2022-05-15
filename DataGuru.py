import pandas as pd


class DataGuru:
    def __init__(self):
        self.df=pd.DataFrame()
        self.model=None
        self.layercnt=-1
        self.algo=None
    def setDF(self,df):
        self.df = df
    def getDF(self):
        return self.df
    def setMODEL(self,model):
        self.model=model
    def getMODEL(self):
        return self.model
    def setLAYERCNT(self,layercnt):
        self.layercnt=layercnt
    def getLAYERCNT(self):
        return self.layercnt
    def setALGO(self,algo):
        self.algo=algo
    def getALGO(self):
        return self.algo
    

