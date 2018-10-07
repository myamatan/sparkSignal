import numpy as np
import pandas as pd

import matplotlib.pylab as plt

CLF = 1.96
xlabel='Surface'
#ylabel='HVline'
#ylabel='count630'
ylabel='amp630'

yerrlabel='HVlineStd'
if ylabel=='count630':
    yerrlabel='countErr'
elif ylabel=='amp630':
    yerrlabel='ampErr630'

# Data loading and print
df = pd.read_csv('./data/resistivity.csv',delimiter=',',header=None).T
df.columns = df.iloc[0]
df = df.drop(df.index[0])
df = df.apply(pd.to_numeric, errors='ignore')
df['countErr'] = np.sqrt(df['count630']) * CLF
df['ampErr630'] = df['ampStd630']/np.sqrt(df['count630']) * CLF
print(df)

# Plot
df.plot.scatter(x=xlabel,y=ylabel,xerr=df[['SurfaceStd']].values.T, yerr=df[[yerrlabel]].values.T, grid=True,color='red')

# plot style
# +++++++++++++++++++++++++++++++++++++++
xRange=max(df[xlabel])-min(df[xlabel])
xMin=min(df[xlabel])-xRange*0.15
xMax=max(df[xlabel])+xRange*0.15
xRange=xMax-xMin

yRange=max(df[ylabel])-min(df[ylabel])
yMin=min(df[ylabel])-yRange*0.7
yMax=max(df[ylabel])+yRange*1.2
yRange=yMax-yMin

plt.text(xMin+xRange*0.05,
         yMax-yRange*0.165,
         #'Sensitive area resistivity [MOhm/sq]\nv.s.\nSensitive area to HV line resistivity [MOhm]',
         #'Sensitive area resistivity [MOhm/sq]\nv.s.\nSpark count',
         'Sensitive area resistivity [MOhm/sq]\nv.s.\nMean shape amplitude [V]',
         size=11,
         bbox=dict(facecolor='white',alpha=0.8))

for i in range(len(df)):
    plt.text(df.iloc[i][xlabel]+xRange*0.01, df.iloc[i][ylabel]+yRange*0.01, 'Ch'+str(i+1))

plt.style.use('ggplot')

plt.xlabel('Sensitive area resistivity [MOhm/sq]')
plt.ylabel('Sensitive are to HV line resistivity [MOhm]')
if ylabel=='count630':
    plt.ylabel('Spark count')
elif ylabel=='amp360':
    plt.ylabel('Mean Shape Amplitude [V]')

plt.xlim(xMin, xMax)
plt.ylim(yMin, yMax)


plt.show()


