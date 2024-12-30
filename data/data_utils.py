import os
import soundfile as sf
import scipy.signal as ss
import numpy as np 
import torch
from torch.utils.data import DataLoader

import torch.share


def wav_to_npy(audio_dir, output_dir):
    """
    Simple function extracting .wav in audio_dir to .npy in output_dir
    """
    for filename in os.listdir(audio_dir):
        filepath = os.path.join(audio_dir, filename)
        data, _ = sf.read(filepath)
        npy_filename = os.path.splitext(filename)[0] + '.npy'
        npy_filepath = os.path.join(output_dir, npy_filename)
        np.save(npy_filepath, data)
        print(f"OK: {npy_filepath}")
    
def extract_wav(project_path) :
    """
    Input : path of the project folder 
    Output : nothing

    You need to rename voice_origin as "clean_signals" and to put 
    the noisy signals in a folder "noisy_signals"
    These two folders are just in the folder /audio/
    
    """
    folder = [r'train', r'test']
    subfolder = [r'noisy_signals', r'clean_signals']
    for word_1 in folder :
        for word_2 in subfolder :
            audio_dir = os.path.join(project_path, r'audio_files', word_2, word_1)
            output_dir = os.path.join(project_path, r'data', word_1, word_2)
            wav_to_npy(audio_dir,output_dir)

# TO RUN ONCE :
# extract_wav(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project")


def generate_global_signals(dir_path, name, output_folder):
    """
    Input : directory path (in which two subdirectories 'train' and 'test'), 
            name of the output file
    Output : one torch tensors, one array
        signals = array with all [noisy signal, clean signal]
        names = array with the names
    """
    signals = []
    names = []
    noisy_dir = os.path.join(dir_path, r"noisy_signals")
    clean_dir = os.path.join(dir_path, r"clean_signals")
    output_path_signals = os.path.join(output_folder, name +'_signals.npy')
    output_path_names = os.path.join(output_folder, name +'_names.npy')
    for filename in os.listdir(noisy_dir):
        if filename[-4:]=='.npy':
            signal_name = filename[:-4]  
            noisy_filepath = os.path.join(noisy_dir, filename)
            clean_filepath = os.path.join(clean_dir, filename)
            noisy_signal = np.load(noisy_filepath)
            clean_signal = np.load(clean_filepath)
            names.append(signal_name)
            signals.append([noisy_signal, clean_signal])
    np.save(output_path_signals, np.array(signals))
    np.save(output_path_names,np.array(names))

# TO DO ONCE
#generate_global_signals(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\data\train",'train',r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\data\global")
#generate_global_signals(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\data\test",'test',r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\data\global")



def generate_global_specto(dir_path, name, output_folder):
    """
    Input : directory path (in which two subdirectories 'train' and 'test'), 
            name of the output file
    Output : one torch tensors, one array
        spectos = array with all [noisy spectogram, clean spectogram]
    """
    spectos = []
    noisy_dir = os.path.join(dir_path, r"noisy_signals")
    clean_dir = os.path.join(dir_path, r"clean_signals")
    output_path_spectos = os.path.join(output_folder, name +'_spectos.npy')
    for filename in os.listdir(noisy_dir):
        if filename[-4:]=='.npy':
            noisy_filepath = os.path.join(noisy_dir, filename)
            clean_filepath = os.path.join(clean_dir, filename)
            noisy_signal = np.load(noisy_filepath)
            clean_signal = np.load(clean_filepath)
            longueur = len(noisy_signal)
            _, _, specto_clean = ss.stft(clean_signal, 1, nperseg=longueur//20, noverlap=longueur//40)
            _, _, specto_noisy = ss.stft(noisy_signal, 1, nperseg= longueur//20, noverlap=longueur//40)
            spectos.append([specto_noisy, specto_clean])
    np.save(output_path_spectos, np.array(spectos))
#TO DO ONCE
#generate_global_specto(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\data\train",'train',r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\data\global")
#generate_global_specto(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\data\test",'test',r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project\data\global")



def load_data(data_path): 
    pass



def preprocess_data(args):
    pass