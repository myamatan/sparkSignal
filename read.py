import sys, os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

outpath='/Users/masahiroyamatani/Desktop/spark_sig/plots'
    
# File reading 
directory = './data/20180709/all.c300a300.2h'
fileID = '11422.CSV' 
dt = 1e-10

def FFT_AMP(data):
    data=np.hamming(len(data))*data
    data=np.fft.fft(data)
    data=np.abs(data)
    return data

if __name__ == "__main__":

    args = sys.argv
    if len(args)==3:
        directory = args[1]
        fileID = args[2]
    
    df = pd.read_csv(directory+'/'+fileID, header=None)
    df.columns = ['Param','nan','nan1','time','vol','nan3']
    
    # Get first index in which voltage is lower than 0.2
    trig_index = df[ df.vol <= -0.2 ].index[0]
    trig_index = df[ (df.index < trig_index) & (df['vol']==0) ]['vol'].index[-1]
    inteVol = df.iloc[ df.index > trig_index ]['vol'].sum(axis=0) * dt
    print('Integral :', '{:.3E}'.format(inteVol), '[V*s]')
    
    line = pd.DataFrame({ 'zeroVol' : np.array([0]*len(df)) } )
    df = pd.concat([df, line], axis=1)

    # FFT 
    vol = np.array(df['vol'])
    volfft = np.fft.fft(vol)
    t = np.array(df['time'])
    freq = np.fft.fftfreq(len(vol), t[1] - t[0])
    
    # Plotting 
    ax = df.plot(x='time', y=['vol'], color=['green'], grid=True, legend=False, alpha=0.96)
    ax.text(0.4e-7,0.5,'Integral :'+str('{:.3E}'.format(inteVol))+' [V*s]', size=14)
    
    plt.fill_between(df['time'], df['vol'], df['zeroVol'],
                    where=np.abs(df['vol']) >= 0,
                    facecolor='green', alpha=0.3, interpolate=True)
    plt.vlines(df.iloc[trig_index]['time'], -1., 1., 'black', linestyles='dashed', linewidth=1)
   
    plt.title(directory+'/'+fileID)
    plt.style.use('ggplot')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    #plt.show()

    if directory[0]=='.':
        if not os.path.exists('./plots/'+directory.split('/')[2]):
            os.mkdir('./plots/'+directory.split('/')[2])
        directory = directory.split('/')[2] + '/' + directory.split('/')[3]
    else:
        if not os.path.exists('./plots/'+directory.split('/')[1]):
            os.mkdir('./plots/'+directory.split('/')[1])
        directory = directory.split('/')[1] + '/' + directory.split('/')[2]

    if not os.path.exists('./plots/'+directory):
        os.mkdir('./plots/'+directory)
    plt.savefig('./plots/'+directory+'/'+fileID[:-4]+'.png')
    plt.close('all')

    plt.figure()
    plt.subplot(2,1,1)
    plt.style.use('ggplot')
    plt.plot(t,vol, color='blue')
    plt.xlabel('Time [s]', fontsize=7)
    plt.ylabel('Voltage [V]', fontsize=7)
    plt.tick_params(labelsize=7)

    plt.subplot(2,1,2)
    plt.plot(freq,abs(volfft), color='blue')
    plt.axis([0,1./(t[1] - t[0])/2,0,max(abs(volfft))])
    plt.xlabel('Frequency [Hz]', fontsize=7)
    plt.ylabel('Amp', fontsize=7)
    plt.tick_params(labelsize=7)
    plt.savefig('./plots/'+directory+'/'+fileID[:-4]+'.fft.png')
    #plt.show()
