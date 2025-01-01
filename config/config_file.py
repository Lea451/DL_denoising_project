# to put in a config file 
config = {}

opt = {
    'batch_size': 32,
    'epochs': 1, #100 , #1 for testing
    'learning_rate': 1e-3,
    'data_dir': './data',
    'model_save_path': './checkpoints/unet_model.pth',
    'model_type': 'resunet',  # Options: 'unet', 'resunet'
    'input_shape': (512, 128)  # Shape of the spectrogram input (height, width)
}

config['opt'] = opt
