import numpy as np
import pandas as pd
import scipy as sp

from scipy.stats import pearsonr
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table



import matplotlib.pylab as plt

CLF = 1.96
xlabel='Surface'

#ylabel='HVline'
ylabel='count630'
#ylabel='amp630'

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

# Correlation
r, p = pearsonr(df[xlabel],df[ylabel])

# LeaseSquare fit
n = len(df[xlabel])
xsum = sum(df[xlabel])
xsum2 = sum(df[xlabel]**2)
ysum = sum(df[ylabel])
ysum2 = sum(df[ylabel]**2)
xysum = sum(df[xlabel]*df[ylabel])

sxx = xsum2 - xsum*xsum/n
sxy = xysum - xsum*ysum/n
xmean = xsum/n
ymean = ysum/n
beta0a = ymean
beta1 = sxy/sxx
ubvar = (ysum2-ysum*ysum/n-sxy*sxy/sxx)/(n-2)

# Fit
#X = sm.add_constant(df[xlabel])
#res = sm.OLS(df[ylabel], X).fit()
#st, data, ss2 = summary_table(res, alpha=0.05)
#fittedvalues = data[:,2]
#predict_mean_se  = data[:,3]
#predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
#predict_ci_low, predict_ci_upp = data[:,6:8].T

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

X = np.linspace(xMin, xMax, 100)
mu = beta0a + beta1 * (X-xmean)
muu = beta0a + beta1 * (X-xmean) + abs(t.ppf(0.025,n-2)) * np.sqrt( (1./n + (X-xmean)**2/sxx)*ubvar ) 
mul = beta0a + beta1 * (X-xmean) - abs(t.ppf(0.025,n-2)) * np.sqrt( (1./n + (X-xmean)**2/sxx)*ubvar ) 
plt.plot(X, mu, 'b--', linewidth=1)
plt.plot(X, muu, 'g--', linewidth=1)
plt.plot(X, mul, 'g--', linewidth=1)

textStr='Sensitive area resistivity [MOhm/sq]\nv.s.\nSensitive area to HV line resistivity [MOhm]'
if ylabel=='count630':
    textStr='Sensitive area resistivity [MOhm/sq]\nv.s.\nSpark count'
elif ylabel=='amp630':
    textStr='Sensitive area resistivity [MOhm/sq]\nv.s.\nMean shape amplitude [V]'

plt.text(xMin+xRange*0.05,
         yMax-yRange*0.165,
         textStr,
         size=11,
         bbox=dict(facecolor='white',alpha=0.8))

plt.text(xMin+xRange*0.6,
         yMax-yRange*0.925,
         'Correlation r: {r}'.format(r=round(r,3))+'\nSignificance p: {p}'.format(p=round(p,3)),
         size=11,
         bbox=dict(facecolor='white',alpha=0.8))

for i in range(len(df)):
    plt.text(df.iloc[i][xlabel]+xRange*0.01, df.iloc[i][ylabel]+yRange*0.01, 'Ch'+str(i+1))

plt.style.use('ggplot')

plt.xlabel('Sensitive area resistivity [MOhm/sq]')
plt.ylabel('Sensitive are to HV line resistivity [MOhm]')
if ylabel=='count630':
    plt.ylabel('Spark count')
elif ylabel=='amp630':
    plt.ylabel('Mean Shape Amplitude [V]')

plt.xlim(xMin, xMax)
plt.ylim(yMin, yMax)


plt.show()


