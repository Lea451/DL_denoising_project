import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


### Dataset pour méthode à spectogrammes
class SpectrogramDataset(Dataset): #suppose qu'on met les spectrogrammes en entrée
    def __init__(self, path_to_signals, path_to_names):
        self.spectos = np.load(path_to_signals)
        self.names = np.load(path_to_names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        clean_noisy = (self.spectos[i])
        s_clean = torch.unsqueeze(torch.tensor((clean_noisy[1])),0) #Tenseur 2D : temps x frequence sur le specto [T,C] = [2001, 41] normalement 
        s_noisy = torch.unsqueeze(torch.tensor((clean_noisy[0])),0)  #Tenseur 2D : temps x frequence sur le specto [T,C]
        #print(s_clean.shape)
        return s_noisy.type(torch.LongTensor), s_clean.type(torch.LongTensor)    

# Dataset pour méthode à signaux
class SignalsDataset(Dataset):
    def __init__(self, path_to_signals, path_to_names):  #charge les signaux (par paires bruité / non bruités)
        self.signals = np.load(path_to_signals)
        self.names = np.load(path_to_names)
        
    def __len__(self): #retourne le nombre de signaux dans le dataset
        return self.signals.shape[0]
    
        
    def __getitem__(self,i): #retourne pour chaque indice i un couple (data_i, label_i), data_i étant un signal et label_i le label associé au signal
        clean_noisy = (self.signals[i])
        clean = (clean_noisy[1])
        noisy = (clean_noisy[0])
        return torch.tensor(noisy).type(float), torch.tensor(clean).type(float)    


### Création des datasets, et validation, Dataloaders

#train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])



### Entraînement du modèle
def train_model(model, train_loader, val_loader, criterion, optimizer, device, opt):
    best_loss = float('inf')
    for epoch in range(opt['epochs']):
        print(f"Epoch {epoch+1}/{opt['epochs']}")
        model.train()
        train_loss = 0.0
        for b_noisy, b_clean in train_loader:
            cond = np.random.choice([0,1], p=[0.7,0.3]) # to do mini-epochs without all the data
            if cond :
                b_noisy, b_clean = b_noisy.to(device).float(), b_clean.to(device).float() #convert both to float
                output = model(b_noisy) #normalement, output= predicted spectro mask donc on doit le multiplier par l'input avant de calculer la loss
                print("ok")
                loss = criterion(output*b_noisy, b_clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_noisy.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device).float(), clean.to(device).float()
                output = model(noisy)
                loss = criterion(output*noisy, clean)
                val_loss += loss.item() * noisy.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{opt['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), opt['model_save_path'])
            print(f"Model saved with Val Loss: {best_loss:.4f}")
