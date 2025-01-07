# DL_denoising_project

**Denoising Audio Deep Learning ans Signal processing course.**

In this notebook we implemented a ResUnet architecture to denoise audio signals.

### Preprocessing

The audio files must be placed in audio_files/clean_signals audio_files/noisy_signals.
Then, you need to run data_utils to generate all useful .npy from the audio files.

### Training

*scripts/train.py* contains the dataset class for the spectrogram, the function s_loss_1 (a loss that can be used for the training), the loss MSLE, and the training pipeline.
*postprocessing.py* contains the functions used to compute the denoised signals using a model. These signals can be saved as .wav but are not automatically.
*config/config_file.py* is a file containing information about what will be used for the training and where to save the model. You can create other similar files to use.

**To launch the training with the parameters of 'config_file', you should type in the terminal : python main.py --exp='config_file'.**

### Test

Similarly, you can test a model with **python main.py --exp='config_file' --evaluate=True --directory='(path of the model)'**.
It calls the functions of *postprocessing.py*

### Notebook

One notebook (Example) presenting some results with two different models : how look denoised signals and spectrogam, the stoi associated... There are also two examples of denoising (.wav) in the folder.


