import os
import soundfile as sf
import scipy.signal as ss
import scipy.io.wavfile
import numpy as np 



def wav_to_npy(audio_dir, output_dir):
    """
    Simple function extracting all the .wav files
    in audio_dir to the same number of .npy in output_dir
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
    
    You need to rename voice_origin as "clean_signals" and to put 
    the noisy signals in a folder "noisy_signals"
    These two folders are just in the folder /audio_files/
    
    """
    folder = [r'train', r'test']
    subfolder = [r'noisy_signals', r'clean_signals']
    print("Extracting wav files...")
    for word_1 in folder :
        for word_2 in subfolder :
            audio_dir = os.path.join(project_path, r'audio_files', word_2, word_1)
            output_dir = os.path.join(project_path, r'data', word_1, word_2)
            wav_to_npy(audio_dir,output_dir)

    
# TO RUN ONCE :
# extract_wav(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\DL_denoising_project")
#extract_wav(r".")
#/Users/leabohbot/Desktop/MVA/DL_Signal/Projet_DL_signal/Project_TDS/DL_denoising_project

def generate_global_signals(dir_path, name, output_folder):
    """
    Input : data directory path (in which two subdirectories 'train' and 'test'), 
            name of the output file
     2 arrays are saved in output_folder
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
            # Skip if clean file is missing
            if not os.path.exists(clean_filepath):
                print(f"Skipping {filename}: missing in clean_signals")
                continue

            noisy_signal = np.load(noisy_filepath)
            clean_signal = np.load(clean_filepath)
            names.append(signal_name)
            signals.append([noisy_signal, clean_signal])
    np.save(output_path_signals, np.array(signals))
    np.save(output_path_names,np.array(names))



# TO DO ONCE

#generate_global_signals(r"..\data",'train',r"./global")
#generate_global_signals("./test",'test',"./global")



def generate_global_specto(dir_path, name, output_folder):
    """
    Input : directory path (in which two subdirectories 'train' and 'test'), 
            name of the output file
    Save one array :
        spectos = array with all the [noisy spectogram amplitude, clean spectogram amplitude]
        phases = array with the angles of the noisy spectograms
    """
    spectos = []
    spectos_angles = []
    noisy_dir = os.path.join(dir_path, r"noisy_signals")
    clean_dir = os.path.join(dir_path, r"clean_signals")
    output_path_spectos = os.path.join(output_folder, name +'_spectos.npy')
    output_path_phases = os.path.join(output_folder, name +'_phases.npy')
    for filename in os.listdir(noisy_dir):
        if filename[-4:]=='.npy':
            noisy_filepath = os.path.join(noisy_dir, filename)
            clean_filepath = os.path.join(clean_dir, filename)
            noisy_signal = np.load(noisy_filepath)
            clean_signal = np.load(clean_filepath)
            longueur = len(noisy_signal)
            _, _, specto_clean = ss.stft(clean_signal, 1, nperseg=longueur//20, noverlap=longueur//40)
            _, _, specto_noisy = ss.stft(noisy_signal, 1, nperseg= longueur//20, noverlap=longueur//40)
            specto_noisy_phase = np.angle(specto_noisy)
            specto_clean = np.abs(specto_clean)
            specto_noisy = np.abs(specto_noisy)  #comme dans l'article Unet, on utilise uniquement les magnitudes du spectogramme A MODIFIER SI BESOIN
            spectos.append([specto_noisy, specto_clean])
            spectos_angles.append(specto_noisy_phase) 
    np.save(output_path_spectos, np.array(spectos))
    np.save(output_path_phases, np.array(spectos_angles))
    print(f"Spectrograms saved at: {output_path_spectos}")
    print(f"Phases saved at: {output_path_phases}")

#TO DO ONCE
#generate_global_specto(r"../data/",'train', r"./global")
#generate_global_specto(".\test",'test',"./global")

def compute_snr(denoised, signals, output_dir, name, boo=True ):
    """
    Input : 2 arrays
        denoised = denoised signals
        signals = array of [noisy_signal, clean_signal]
        output_dir = path of the output dir
        name = name for the file
        boo : boolean, if False compute the snr of noisy_signals (and the array denoised does not matter)
    Save an .npy array with snr (10log(|signal|**2/|noise|^2))
    """
    snr = []
    for i in range(len(signals)):
        signal = signals[i][0]
        clean = signals[i][1]
        if boo :
            signal = denoised[i]
        n1 = np.sum(np.abs(clean*clean))
        n2 = np.sum(np.abs((clean-signal)*(clean-signal)))
        if n2 != 0 : snr.append(20*np.log10(n1/n2))
        else : snr.append(np.infty)
    snr = np.array(snr)
    output_path = os.path.join(output_dir, name+'_snr.npy')
    np.save(output_path,snr)
    print('SNRs saved')

output_dir = r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\dl\DL_denoising_project\data\global"
signals = np.load(r"C:\Users\valen\Documents\Travail\X\MVA\S1\ProjetTDS\dl\DL_denoising_project\data\global\test_signals.npy")

compute_snr([None], signals,output_dir,'test', boo=False)

