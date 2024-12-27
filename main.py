import argparse
import os
import importlib
import torch
from torch.utils.data import DataLoader
from model import UNet, ResUNet
from train import train_model, SpectrogramDataset
from data import load_data, preprocess_data #TODO : coder les fonctions load_data et preprocess_data

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, help='Config file name in the config folder (without .py extension)')
parser.add_argument('--evaluate', default=False, action='store_true', help='Evaluate the model instead of training')
parser.add_argument('--checkpoint', type=str, default='', help='Path to the checkpoint to load')
args = parser.parse_args()

# Load configuration : les paramètres sont dans les folder py de config
config_module = importlib.import_module(f'config.{args.exp}')
opt = config_module.opt

# Main logic
def main():
    # Load data
    noisy_train, clean_train = load_data(os.path.join(opt['data_dir'], 'train')) #TODO : coder les fonctions load_data et preprocess_data
    noisy_val, clean_val = load_data(os.path.join(opt['data_dir'], 'val'))

    # Preprocess data
    train_data = preprocess_data(noisy_train, clean_train) #TODO : coder les fonctions load_data et preprocess_data
    val_data = preprocess_data(noisy_val, clean_val)

    train_dataset = SpectrogramDataset(*train_data)
    val_dataset = SpectrogramDataset(*val_data)

    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'], shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt['model_type'] == 'unet':
        model = UNet(input_channels=1, output_channels=1).to(device)
    elif opt['model_type'] == 'resunet':
        model = ResUNet(input_channels=1, output_channels=1).to(device)
    else:
        raise ValueError("Invalid model type specified in config.")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['learning_rate'])

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Train or evaluate
    if not args.evaluate:
        print("Starting training...")
        train_model(model, train_loader, val_loader, criterion, optimizer, device, opt)
    else:
        print("Evaluation logic to be added...")

if __name__ == "__main__":
    main()
