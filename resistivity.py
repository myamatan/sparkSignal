import numpy as np
import pandas as pd

import matplotlib.pylab as plt

# Data loading
df = pd.read_csv('./data/resistivity.csv',delimiter=',',header=None).T

df.columns = df.iloc[0]
df = df.drop(df.index[0])
df = df.apply(pd.to_numeric, errors='ignore')
df['countErr'] = np.sqrt(df['count630'])

print(df)

#df.plot.scatter(x='Surface',y='HVline',xerr=df[['SurfaceStd']].values.T, yerr=df[['HVlineStd']].values.T, grid=True)
df.plot.scatter(x='Surface',y='count630',xerr=df[['SurfaceStd']].values.T, yerr=df[['countErr']].values.T, grid=True,color='r')
plt.style.use('ggplot')
plt.xlabel('Surface resistivity [MOhm/sq]')
#plt.ylabel('HV line [MOhm]')
plt.ylabel('#spark')


plt.show()


