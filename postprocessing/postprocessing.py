
import os
import soundfile as sf
import scipy.signal as ss
import torch
import numpy as np

s_r = 8000


def masks_to_signals(amps, phases, output_dir, name):
    """ Correct the noisy amplitude with the mask from the model, 
    convert the spectograms to wav.
    Input : 2 arrays
            amps = amplitudes of the noisy spectograms corrected
            phases = angles of the noisy spectogram
            1 path 
            output_dir = where to save all the denoised signals as a nump  
    Create the new audio files in the directory of path output_dir."""
    signals = []
    for i in range(len(amps)):
      new_specto = amps[i]*np.exp(1j*phases[i])
      signal = ss.istft(new_specto)
      signals.append(signal)
    output_path = os.path.join(output_dir,name+'_denoised.npy')
    np.save(output_path, np.array(signals))
