import argparse
import os
import importlib
import importlib.util
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from models.ResUnet import ResUnet
from scripts import train
from sklearn.metrics import mean_squared_log_error

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, help='Config file name in the config folder (without .py extension)')
parser.add_argument('--evaluate', default=False, action='store_true', help='Evaluate the model instead of training')
parser.add_argument('--checkpoint', type=str, default='', help='Path to the checkpoint to load')
args = parser.parse_args()

# Construct the path to the configuration file
exp_config_file = os.path.join('.', 'config', args.exp + '.py')

# Use importlib to load the module dynamically
spec = importlib.util.spec_from_file_location("config", exp_config_file)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Access the 'opt' variable from the config file
opt = config.opt
print(opt)


# Load data
path_train = r"./data/global/train_spectos.npy"
path_train_names = r"./data/global/train_names.npy"
path_test = r"./data/global/test_spectos.npy"
path_test_names = r"./data/global/test_names.npy"
#path_train = r"\data\global\train_spectos.npy"
#path_train_names = r"\data\global\train_names.npy"
#path_test = r"\data\global\test_spectos.npy"
#path_test_names = r"\data\global\test_names.npy"

torch.manual_seed(172)


# Main logic
def main():
    # Load data + create datasets + create dataloaders
    train_full = train.SpectrogramDataset(path_to_signals=path_train, path_to_names=path_train_names)
    test_dataset = train.SpectrogramDataset(path_to_signals=path_test, path_to_names=path_test_names)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_full, [0.85, 0.15])
    
    #print("len(train_dataset)", len(train_dataset))
    #print("len(valid_dataset)", len(valid_dataset))
    #print(train_dataset[0][0])

    train_dataloader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=20, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt['model_type'] == 'unet':
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif opt['model_type'] == 'resunet':
        model = ResUnet(in_channels=1, out_channels=1).to(device) #si on prend les amplitudes en entrée on devrait bien mettre in_channels=1 et out_channels=1
    else:
        raise ValueError("Invalid model type specified in config.")

    #criterion = torch.nn.MSELoss()
    criterion = train.RMSLELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['learning_rate'])

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Train or evaluate
    if not args.evaluate:
        print("Starting training...")
        train.train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, device, opt)
    else:
        print("Evaluation logic to be added...")

if __name__ == "__main__":
    main()
