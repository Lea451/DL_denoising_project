# to put in a config file 
import torch
config = {}

test = {
    'batch_size': 6,
    'epochs': 1, #100 , #1 for testing
    'learning_rate': 1e-3,
    'data_dir': './data',
    'model_save_path': './checkpoints/test.pt',
    'model_type': 'resunet',  # Options: 'unet', 'resunet'
}

opt = {
    'batch_size': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epochs': 100, #100 , #1 for testing
    'learning_rate': 1e-3,
    'data_dir': './data',
    'model_save_path': './checkpoints/resunet_model.pt',
    'model_type': 'resunet', 
    'losses_path': './losses.txt' 
}

config['opt'] = opt
config['test'] = test
