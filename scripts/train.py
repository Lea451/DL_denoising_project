import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tqdm


### Dataset pour méthode à spectogrammes
#class SpectrogramDataset(Dataset): #suppose qu'on met les spectrogrammes en entrée
#    def __init__(self, path_to_signals, path_to_names):
#        self.spectos = np.load(path_to_signals)
#        self.names = np.load(path_to_names)

#    def __len__(self):
#        return len(self.names)

#    def __getitem__(self, i):
#        clean_noisy = (self.spectos[i])
 #       s_clean = torch.unsqueeze(torch.tensor((clean_noisy[1])),0) #Tenseur 2D : temps x frequence sur le specto [T,C] = [2001, 41] normalement 
#       s_noisy = torch.unsqueeze(torch.tensor((clean_noisy[0])),0)  #Tenseur 2D : temps x frequence sur le specto [T,C]
 #       #print(s_clean.shape)
#        return s_noisy.type(torch.float), s_clean.type(torch.float) 


# Normalization and denormalization functions
def normalize_spectrogram(spectrogram):
    """
    Normalize a spectrogram by applying log transformation followed by min-max normalization.
    """
    spectrogram_log = np.log(np.abs(spectrogram) + 1e-10)  # Avoid log(0) by adding a small epsilon
    spectrogram_min = spectrogram_log.min()
    spectrogram_max = spectrogram_log.max()
    normalized_spectrogram = (spectrogram_log - spectrogram_min) / (spectrogram_max - spectrogram_min)
    return normalized_spectrogram, spectrogram_min, spectrogram_max

def denormalize_spectrogram(normalized_spectrogram, spectrogram_min, spectrogram_max):
    """
    Denormalize a spectrogram back to its original scale.
    """
    spectrogram_log = (normalized_spectrogram * (spectrogram_max - spectrogram_min)) + spectrogram_min
    spectrogram = np.exp(spectrogram_log) - 1e-10 # Reverse the log transformation
    return spectrogram

# Dataset class with normalization integrated
class SpectrogramDataset(Dataset):  # suppose qu'on met les spectrogrammes en entrée
    def __init__(self, path_to_signals, path_to_names):
        self.spectos = np.load(path_to_signals)
        self.names = np.load(path_to_names)
        self.normalization_params = []  # Store min and max for each spectrogram

        # Normalize spectrograms
        normalized_spectos = []
        for noisy, clean in self.spectos:
            noisy_norm, noisy_min, noisy_max = normalize_spectrogram(noisy)
            clean_norm, clean_min, clean_max = normalize_spectrogram(clean)
            normalized_spectos.append([noisy_norm, clean_norm])
            self.normalization_params.append([(noisy_min, noisy_max), (clean_min, clean_max)])
        self.spectos_norm = np.array(normalized_spectos)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        clean_noisy = self.spectos_norm[i]
        s_clean = torch.unsqueeze(torch.tensor((clean_noisy[1])), 0)  # Tenseur 2D : temps x frequence sur le specto [T,C] = [2001, 41] normalement
        s_noisy = torch.unsqueeze(torch.tensor((clean_noisy[0])), 0)  # Tenseur 2D : temps x frequence sur le specto [T,C]
        return s_noisy.type(torch.float), s_clean.type(torch.float), self.normalization_params[i]  # Include normalization params

# Example usage
# dataset = SpectrogramDataset('path_to_original_data.npy', 'path_to_names.npy')

   

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

class SLoss_1(nn.Module):
    def __init__(self, weight):
        super(SLoss_1, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        # Compute the loss
        loss = torch.mean(self.weight * (input - target) ** 2)
        return loss


def s_loss_1(specto_1, specto_2):
    somme =  torch.sum(10*torch.abs(torch.log(specto_1)-torch.log(specto_2)))
    somme /= torch.numel(specto_1)
    return somme


### Création des datasets, et validation, Dataloaders

#train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])



### Entraînement du modèle
def train_model(model, train_loader, val_loader, criterion, optimizer, device, opt):
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(opt['epochs']):
        print(f"Epoch {epoch+1}/{opt['epochs']}")
        model.train()
        train_loss = 0.0
        for b_noisy, b_clean, _ in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{opt['epochs']} - Training"):
            #cond = np.random.choice([0,1], p=[0.6,0.4]) # to do mini-epochs without all the data
            #if cond :
            b_noisy, b_clean = b_noisy.to(device), b_clean.to(device) #convert both to float
            output = model(b_noisy) #normalement, output= predicted spectro mask donc on doit le multiplier par l'input avant de calculer la loss
            loss = s_loss_1(output*b_noisy, b_clean)
            print("loss=",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_noisy.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean, _ in tqdm.tqdm(val_loader, desc=f"Epoch {epoch + 1}/{opt['epochs']} - Training"):
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                loss = criterion(output*noisy, clean)
                val_loss += loss.item() * noisy.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{opt['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        #make sure directory to text file exists
        os.makedirs(os.path.dirname(opt['losses_path']), exist_ok=True)
        with open(opt['losses_path'], 'w') as f: #add train and val loss to a txt file
            f.write(f"Epoch {epoch+1}/{opt['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
        

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), opt['model_save_path'])
            print(f"Model saved with Val Loss: {best_loss:.4f}")

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()