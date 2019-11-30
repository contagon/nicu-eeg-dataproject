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
    fig.text(0.08, 0.5, 'mV', ha='center', va='center', rotation='vertical')
    
    
#     name any feature you'd like to add here
#             dic['Expert'] = data.expert
#             #max value found in each channel
#             max_val = data.add_prefix('Max_Value_')
#             #max frequency found in each channel
#             max_freq = data_fft.add_prefix('Max_Freq_')
#             #max frequency value as found in each channel
#             max_freq_val = data_fft.add_prefix('Max_Freq_Value_')
#             dic = {**dic, **max_freq.idxmax().to_dict(), **max_freq_val.max().to_dict(), **max_val.max().to_dict()}
#             #whether it was a seizure or not
#             dic['Seizure'] = data.seizure
#             #same as before maxes, but across all the channels
#             dic['Max_Freq'] = data_fft.drop("ECG EKG", axis=1).stack().idxmax()[0]
#             dic['Max_Freq_Value'] = data_fft.drop("ECG EKG", axis=1).max().max()
#             dic['Max_Value'] = data.drop("ECG EKG", axis=1).max().max()

# import statsmodels.api as sm
# data = read_data('data-final/eeg.pkl')
# Y = data['Seizure']
# X = data.drop('Seizure', axis=1)

# filter_col = [col for col in df if (col.startswith('Freq') and not col.endswith("ECG EKG"))]
# results = sm.Logit(Y,X[filter_col]).fit_regularized(alpha=1, L1_wt=0)
# print(results.params)
# print(X[filter_col].shape)
# np.linalg.matrix_rank(X[filter_col].values.T@X[filter_col].values, tol=1e-8)