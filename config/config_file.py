# to put in a config file 
config = {}
opt = {
    'batch_size': 6,
    'epochs': 15, #100 , #1 for testing
    'learning_rate': 1e-3,
    'data_dir': './data',
    'model_save_path': './checkpoints/resunet_model1.pt',
    'model_type': 'resunet',  
}

config['opt'] = opt