import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('summaries_AUC.csv')
plt.plot(data.Step, data.Value, 'b')
#plt.plot(data.Step, data.Value.rolling(window=3).mean().fillna(method='ffill'), 'pink')
plt.xlabel('Steps')
plt.ylabel('AUC')
plt.grid(True)
plt.savefig('Plot.png', dpi=300)
pass