import os
import soundfile as sf
import scipy.signal as ss
import torch
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def model_evaluate(model, test_dataloader, output_path):
    masks = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
    with torch.no_grad():
        for noisy, clean in tqdm(test_dataloader, desc="Evaluating"):
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy).cpu()
            mask = np.array((output.squeeze(0)).squeeze(0))
            masks.append(mask)
    np.save(output_path, np.array(masks, dtype=object))  # Save with dtype=object  # Save masks to file
    print(f"Masks saved at {output_path}")


def masks_to_signals(masks, noisy_spectos, noisy_phases, output_dir, name):
    """ Correct the noisy amplitude with the mask from the model, 
    convert the spectograms to wav.
    Input : 3 arrays
            masks = corrections of the amplitudes of the noisy spectograms
            noisy_spectos =  noisy spectograms amplitudes
            noisy_phases = noisy spectograms angles
            1 path 
            output_dir = where to save all the denoised signals as a nump 
            name = 'train' or 'test'  
    Create the new audio files in the directory of path output_dir."""
    signals = []
    for i in range(len(masks)):
      new_specto = masks[i]*noisy_spectos[i][0]*np.exp(1j*noisy_phases[i])
      t, signal = ss.istft(new_specto)
      signals.append(signal)
    output_path = os.path.join(output_dir,name+'_denoised.npy')
    print("Saving denoised signals...")
    np.save(output_path, np.array(signals))


def denoised_signal_to_wav(denoised, names, number=10, s_r = 8000) :
    """ Input : 
    denoised = array of denoised signals
    names = array of signals names
    number = number of signals to convert (if =0, all the signals)
    s_r = sampling rate

    Save (number) of the first denoised signals as .wav in the current directory
    """
    if number==0 : number = len(names)
    for i in range(number):
        file_name = names[i]+'_denoised.wav'
        signal = (denoised[i])
        signal = np.float32(signal/np.max(np.abs(signal)))
        scipy.io.wavfile.write(filename=file_name,rate=s_r,data=signal)