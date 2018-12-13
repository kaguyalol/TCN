import pandas as pd

data=pd.read_csv('Sentiment_cleaned.csv')
#dct={'Positive':1,'Negative':0}
#data['sentiment']=data['sentiment'].apply(lambda x:dct[x])
data=data[data.columns[::-1]]
data.to_csv('Sentiment_cleaned.csv',index=False)

pass