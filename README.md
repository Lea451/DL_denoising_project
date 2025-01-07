# DL Denoising Project

**Denoising Audio Deep Learning and Signal Processing Course**

This project implements a **ResUnet architecture** for denoising audio signals, combining deep learning techniques with signal processing methods.

---

## Project Structure

| **Folder/File**            | **Description**                                                                                                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `audio_files/`             | Contains the audio signals used for the project: **clean_signals/** and **noisy_signals/** subfolders.                                                                            |
| `checkpoints/`             | Stores trained model checkpoints.                                                                                                                                               |
| `config/`                  | Includes configuration files (e.g., `config_file.py`) that define training parameters and paths for saving outputs.                                                              |
| `data/`                    | Contains processed `.npy` files generated from the audio signals.                                                                                                                |
| `models/`                  | Implements the ResUnet architecture and related models.                                                                                                                          |
| `notebooks/`               | Jupyter notebooks showcasing results, visualizations, and evaluations of the denoised signals.                                                                                   |
| `scripts/`                 | Contains Python scripts for data preprocessing, training, and testing.                                                                                                           |
| `tests/`                   | Scripts for unit testing different parts of the codebase.                                                                                                                        |
| `compare_signals.py`       | A script to compare original and denoised signals and compute evaluation metrics (e.g., STOI).                                                                                    |
| `main.py`                  | The main entry point for training and evaluation.                                                                                                                                |
| `postprocessing.py`        | Functions for postprocessing the model outputs, including saving denoised signals as `.wav` files.                                                                               |

---

## Setup and Usage

### Preprocessing
1. Place your audio files in the following directories:
   - `audio_files/clean_signals/`
   - `audio_files/noisy_signals/`
2. Run the `scripts/data_utils.py` script to generate `.npy` files from the audio files.

### Training
1. Configure the training parameters in `config/config_file.py`.
2. Launch training with:
   ```bash
   python main.py --exp='config_file'
   ```

### Testing
1. Test a trained model with:
   ```bash
   python main.py --exp='config_file' --evaluate=True --directory='(path to model)'
   ```

---

## Results

### Spectrogram Visualization

| **Clean Signal**          | **Noisy Signal**          | **Denoised Signal**          |
|----------------------------|---------------------------|-------------------------------|
| ![Clean Signal Plot](path_to_clean_signal_plot.png) | ![Noisy Signal Plot](path_to_noisy_signal_plot.png) | ![Denoised Signal Plot](path_to_denoised_signal_plot.png) |

### Audio Comparisons
- [Play Clean Signal](path_to_clean_signal.wav)
- [Play Noisy Signal](path_to_noisy_signal.wav)
- [Play Denoised Signal](path_to_denoised_signal.wav)

### Metrics
- **STOI:** 0.92
- **PESQ:** 3.1

---

## Notebooks
Explore example notebooks in `notebooks/` for visualizing results, spectrograms, and evaluation metrics.

---

### How to Add Plots and Audio
1. **Add Plots**: Save the generated plots as `.png` files in a folder (e.g., `notebooks/plots/`). Reference them in the README using:
   ```markdown
   ![Plot Description](path_to_plot.png)
   ```
2. **Embed Audio**: Save audio as `.wav` files in a folder (e.g., `audio_files/results/`) and link them using:
   ```markdown
   [Play Audio Description](path_to_audio.wav)
   ```

---
