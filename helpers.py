import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import mne                               #package to handle EEG data files
import os, seaborn, re
from scipy import io                     #for loading matlab file
from scipy import fftpack                #for Fourier Transform Analysis
from sklearn import datasets, linear_model, model_selection


plt.style.use("seaborn")
plt.rcParams["figure.dpi"] = 300

fs = 256                                 #sample size


def read_data(filename):
    return pd.read_pickle(filename)
    
def plot_signal(data, channels=None, slices=None):
    #read in data if it's a string
    if isinstance(data, str):
        data = read_data(data)
        
    #get all channels needed
    if channels == None:
        channels = sorted(list(data.columns))
        remove = ["ECG EKG", "expertA", "expertB", "expertC"]
        for i in remove:
            if i in channels:
                channels.remove(i)
        
    #get indices of data
    if slices == None:
        low = 0
        up = len(data.index)
    elif slices is tuple or slices is list:
        low = slices[0]*fs
        up = slices[1]*fs
    else:
        length = fs*slices
        low = np.random.randint(slices, len(data.index)-slices)
        up = low + length
        
    #plot them
    n = int(np.ceil( len(channels) / 2))
    fig, ax = plt.subplots(n, 2, sharex=True, sharey=True, gridspec_kw={'hspace': 0.2, 'wspace': 0.01})
    ax = ax.reshape(-1)
    fig.set_size_inches(20, 2*n, forward=True)
    for i, chan in enumerate(channels):
        ax[i].plot(data.index[low:up], data[chan].iloc[low:up])  
        ax[i].set_title(chan)
    # Set common labels
    fig.text(0.5, 0.01*n, 'Time (sec)', ha='center', va='center')
    fig.text(0.08, 0.5, r'$\mu$V', ha='center', va='center', rotation='vertical')