import numpy as np
import pandas as pd

import matplotlib.pylab as plt

# Data loading
df = pd.read_csv('./data/count.csv',delimiter=',')
#df = pd.read_csv('./data/amp.csv',delimiter=',')
#df = pd.read_csv('./data/std.csv',delimiter=',')
#df = pd.read_csv('./data/resistivity.csv',delimiter=',')

print(df)


# Calculate error
erdf = np.sqrt(df)
erdf = erdf.drop(['HV'], axis=1)
erdf.columns = ['Ch1err','Ch2err','Ch3err','Ch4err','Ch5err']
# Concat
df = pd.concat([df, erdf], axis=1)
print(df)

#df = df[ df['HV']==630 ] 

ax = df.plot(x='HV', y=['Ch1','Ch2','Ch3','Ch4','Ch5'], yerr=df[['Ch1err','Ch2err','Ch3err','Ch4err','Ch5err']].values.T, marker='o',grid=True, legend=True, alpha=0.96, capsize=4)
plt.style.use('ggplot')
plt.xlabel('HV [V]')
plt.ylabel('#spark')
plt.show()


