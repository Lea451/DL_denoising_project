# to put in a config file 
config = {}

test = {
    'batch_size': 5,
    'epochs': 1, #100 , #1 for testing
    'learning_rate': 1e-3,
    'data_dir': './data',
    'model_save_path': './checkpoints/test.pt',
    'model_type': 'resunet',  # Options: 'unet', 'resunet'
}

opt = {
    'batch_size': 6,
    'epochs': 5, #100 , #1 for testing
    'learning_rate': 1e-3,
    'data_dir': './data',
    'model_save_path': './checkpoints/resunet_model1.pt',
    'model_type': 'resunet',  
}

config['opt'] = opt
config['test'] = test
