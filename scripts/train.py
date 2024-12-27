import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SpectrogramDataset(Dataset): #suppose qu'on met les spectrogrammes en entrée
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        return torch.tensor(noisy, dtype=torch.float32), torch.tensor(clean, dtype=torch.float32)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, opt):
    best_loss = float('inf')

    for epoch in range(opt['epochs']):
        model.train()
        train_loss = 0.0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            output = model(noisy.unsqueeze(1))
            loss = criterion(output, clean.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy.unsqueeze(1))
                loss = criterion(output, clean.unsqueeze(1))
                val_loss += loss.item() * noisy.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{opt['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), opt['model_save_path'])
            print(f"Model saved with Val Loss: {best_loss:.4f}")
