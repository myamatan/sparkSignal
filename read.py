import sys, os

import pandas as pd
import numpy as np

import scipy.signal

import matplotlib as mpl
import matplotlib.pyplot as plt

outpath='/Users/masahiroyamatani/Desktop/spark_sig/plots'
    
# File reading 
directory = './data/20180709/all.c300a300.2h'
fileID = '11422.CSV' 
dt = 1e-10
trigger = -0.01 #-0.2

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

    # Data for raw plotting
    vol = np.array(df['vol'])
    t = np.array(df['time'])

    # Bandpass (stop) Filter 
    nyq = ( 1.0/(t[1] - t[0]) ) / 2.0
    fe1 = 1e+3 / nyq
    fe2 = 0.027e+9 / nyq #0.05e+9
    numtaps = 255
    b = scipy.signal.firwin(numtaps, [fe1, fe2], pass_zero=False) #band pass
    #b = scipy.signal.firwin(numtaps, [fe1, fe2]) #band stop

    # FFT 
    fft = np.fft.fft(vol)
    spectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in fft]
    freq = np.fft.fftfreq(len(vol), t[1] - t[0])

    # Filtered voltage and spectrum
    bvol = scipy.signal.lfilter(b, 1, vol)
    adf = pd.DataFrame(bvol, columns=['bvol'])
    df = pd.concat([df, adf], axis=1)
    bfft = np.fft.fft(bvol)
    bspectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in bfft]
    
    # Dataframe arrange
    line = pd.DataFrame({ 'zeroVol' : np.array([0]*len(df)) } )
    df = pd.concat([df, line], axis=1)

    # Get first index in which voltage is lower than 0.2
    trig_index = df[ df.bvol <= trigger ].index[0]
    trig_index = df[ (df.index < trig_index) & (abs(df['bvol'])<=0.001) ]['bvol'].index[-1]
    inteVol = df.iloc[ df.index > trig_index ]['bvol'].sum(axis=0) * dt
    print('Integral :', '{:.3E}'.format(inteVol), '[V*s]')
    
    # Plotting 
    ax = df.plot(x='time', y=['bvol'], color=['green'], grid=True, legend=False, alpha=0.96)
    ax.text(0.1e-7,max(bvol)*1.25,'Integral :'+str('{:.3E}'.format(inteVol))+' [V*s]', size=14)
    
    plt.fill_between(df['time'], df['bvol'], df['zeroVol'],
                    where=np.abs(df['bvol']) >= 0,
                    facecolor='green', alpha=0.3, interpolate=True)
    plt.vlines(df.iloc[trig_index]['time'], -1., 1., 'black', linestyles='dashed', linewidth=1)
   
    plt.title(directory+'/'+fileID)
    plt.style.use('ggplot')
    plt.xlim(t[trig_index-100], t[-1])
    plt.ylim(-1.2*max(abs(bvol)), 1.2*max(abs(bvol)))
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
    plt.xlim(t[trig_index-300], t[-1])
    plt.ylim(-1.2*max(abs(vol)), 1.2*max(abs(vol)))
    plt.xlabel('Time [s]', fontsize=7)
    plt.ylabel('voltage [V]', fontsize=7)
    plt.tick_params(labelsize=7)

    plt.subplot(2,1,2)
    plt.plot(freq, spectrum, color='blue')
    plt.axis([0, nyq, 0, max(spectrum)])
    plt.xlabel('frequency [Hz]', fontsize=7)
    plt.ylabel('spectrum', fontsize=7)
    plt.xlim(0, 1.2e+9)
    plt.tick_params(labelsize=7)
    plt.savefig('./plots/'+directory+'/'+fileID[:-4]+'.raw.png')
    #plt.show()
    plt.close('all')

    plt.figure()
    plt.subplot(2,1,1)
    plt.style.use('ggplot')
    plt.plot(t,bvol, color='red')
    plt.xlim(t[trig_index-300], t[-1])
    plt.ylim(-1.2*max(abs(bvol)), 1.2*max(abs(bvol)))
    plt.xlabel('Time [s]', fontsize=7)
    plt.ylabel('voltage [V]', fontsize=7)
    plt.tick_params(labelsize=7)

    plt.subplot(2,1,2)
    plt.plot(freq, bspectrum, color='red')
    plt.axis([0, nyq, 0, max(spectrum)])
    plt.xlabel('frequency [Hz]', fontsize=7)
    plt.ylabel('spectrum', fontsize=7)
    plt.xlim(0, 1.2e+9)
    plt.tick_params(labelsize=7)
    plt.savefig('./plots/'+directory+'/'+fileID[:-4]+'.bpass.png')
    #plt.show()
    plt.close('all')

