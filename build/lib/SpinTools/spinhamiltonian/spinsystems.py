import pandas as pd
class SpinSystems:
    def __init__(self):
        return 
    
    def systems(self,sysindex):        
        dic = {'Bi':{'I': 9/2,
               'J': 1/2,
               'gS': -2.00032,
               'gI':0.9135,
               'A':1475.171e06}}
        
        df = pd.DataFrame(data=dic)
        return df[sysindex].astype(object)