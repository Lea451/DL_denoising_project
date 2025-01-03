
import os
import soundfile as sf
import scipy.signal as ss
import torch
import numpy as np

s_r = 8000
modele = torch.load(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\checkpoints\resunet_model.pt")


def masks_to_signals(masks, noisy_spectos, output_dir, name):
    """ Correct the noisy amplitude with the mask from the model, 
    convert the spectograms to wav.
    Input : 2 arrays
            masks = corrections of the amplitudes of the noisy spectograms
            noisy_spectos =  noisy spectograms
            1 path 
            output_dir = where to save all the denoised signals as a nump  
    Create the new audio files in the directory of path output_dir."""
    signals = []
    for i in range(len(masks)):
      new_specto = masks[i]*noisy_spectos[i]
      signal = ss.istft(new_specto)
      signals.append(signal)
    output_path = os.path.join(output_dir,name+'_denoised.npy')
    np.save(output_path, np.array(signals))
