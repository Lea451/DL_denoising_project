
import os
import soundfile as sf
import scipy.signal as ss
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from models.ResUnet import ResUnet
from scripts import train
import scipy.io.wavfile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s_r = 8000
#modele = ResUnet(1,1).to(device)
#modele.load_state_dict(torch.load(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\checkpoints\resunet_model.pt"))
#modele.eval()
#path_test = r"./data/global/test_spectos.npy"
#path_test_names = r"./data/global/test_names.npy"
#test_dataset = train.SpectrogramDataset(path_to_signals=path_test, path_to_names=path_test_names)
#test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def model_evaluate(model, test_dataloader, output_path):
    masks = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
    with torch.no_grad():
        for noisy, clean, norm_param in tqdm(test_dataloader, desc="Evaluating"):
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy).cpu()
            mask = np.array(output.squeeze(0))
            masks.append(mask)
            #print(output.shape, mask.shape)
    np.save(output_path, np.array(masks, dtype=object))  # Save with dtype=object  # Save masks to file
    print(f"Masks saved at {output_path}")


#output_path = r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\postprocessing\results\test_masks.npy"
#model_evaluate(modele,output_path)


def masks_to_signals(masks, noisy_spectos, noisy_phases, output_dir, name):
    """ Correct the noisy amplitude with the mask from the model, 
    convert the spectograms to wav.
    Input : 3 arrays
            masks = corrections of the amplitudes of the noisy spectograms
            noisy_spectos =  noisy spectograms amplitudes
            noisy_phases = noisy spectograms angles
            1 path 
            output_dir = where to save all the denoised signals as a nump  
    Create the new audio files in the directory of path output_dir."""
    signals = []
    for i in range(len(masks)):
      new_specto = masks[i]*noisy_spectos[i][0]*np.exp(1j*noisy_phases[i])
      
      #denormalize the spectrogram here
      #new_specto_denorm = train.denormalize_spectrogram(new_specto, noisy_spectos[i][1][0], noisy_spectos[i][1][1])
      
      
      t, signal = ss.istft(new_specto)
      signals.append(signal[0])
    output_path = os.path.join(output_dir,name+'_denoised.npy')
    print("Saving denoised signals...")
    np.save(output_path, np.array(signals))
    print(np.array(signal).shape)

#masks = np.load(output_path)
#noisy_spectos = np.load(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\data\global\test_spectos.npy")
#noisy_phases = np.load(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\data\global\test_phases.npy")
#masks_to_signals(masks, noisy_spectos, noisy_phases, r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\postprocessing\results","test")
#denoised = np.load(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\postprocessing\results\test_denoised.npy")

#names = np.load(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\data\global\test_names.npy")
#for i in range(10):
#   file_name =  names[i] +'_denoised.wav'
#   signal = denoised[i]/np.max(np.abs(denoised[i]))
#   scaled = np.int16(signal* 32767)
#   scipy.io.wavfile.write(filename=file_name,rate=s_r,data=scaled)
   