import pandas as pd


class DataGuru:
    def __init__(self):
        self.df=pd.DataFrame()
    def setDF(self,df):
        self.df = df
    def getDF(self):
        return self.df

