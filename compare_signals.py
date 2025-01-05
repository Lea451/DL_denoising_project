import numpy as np
import os
import random
import scipy.io.wavfile
import argparse

def save_wav(signal, sample_rate, file_name):
    """
    Normalize and save a signal as a WAV file in the range [0, 1].
    
    Args:
        signal (np.ndarray): The signal to save.
        sample_rate (int): The sample rate of the signal.
        file_name (str): The path to the output WAV file.
    """
    # Handle invalid signal values (e.g., NaNs, Infs)
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        print(f"Warning: Signal contains NaNs or Infs. File: {file_name}")
        signal = np.nan_to_num(signal)  # Replace NaNs/Infs with zero

    # Normalize signal to [0, 1]
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if signal_max > signal_min:  # Avoid division by zero
        signal = (signal - signal_min) / (signal_max - signal_min)
    else:
        print(f"Warning: Signal has constant values. File: {file_name}")
        signal = np.zeros_like(signal)

    # Scale to int16 range [0, 32767] for saving as WAV
    scaled = np.int16(signal * 32767)

    print(f"Saving WAV file: {file_name}")
    print(f"Signal max after scaling: {np.max(scaled)}, min: {np.min(scaled)}")
    scipy.io.wavfile.write(file_name, rate=sample_rate, data=scaled)


def compare_signals(test_signals_path, test_denoised_path, names_path, output_dir, sample_rate=8000):
    """
    Compare test signals to denoised signals by saving 5 random pairs as WAV files.

    Args:
        test_signals_path (str): Path to the test signals file (npy).
        test_denoised_path (str): Path to the test denoised signals file (npy).
        names_path (str): Path to the file containing the names of the signals (npy).
        output_dir (str): Directory to save the output WAV files.
        sample_rate (int): Sample rate for saving the WAV files.
    """
    # Load signals and names
    test_signals = np.load("./data/global/test_signals.npy", allow_pickle=True)
    test_denoised = np.load(test_denoised_path, allow_pickle=True)
    names = np.load(names_path, allow_pickle=True)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Randomly select 5 indices
    data_size = min(len(test_signals), len(test_denoised), len(names))
    indices = random.sample(range(data_size), 5)
    print(f"Randomly selected indices: {indices}")

    for i in indices:
        # Original noisy and clean signals
        noisy_signal, clean_signal = test_signals[i]

        # Corresponding denoised signal
        denoised_signal = test_denoised[i]

        # Names for files
        base_name = names[i]

        # Save as WAV files
        noisy_file = os.path.join(output_dir, f"{base_name}_noisy.wav")
        clean_file = os.path.join(output_dir, f"{base_name}_clean.wav")
        denoised_file = os.path.join(output_dir, f"{base_name}_denoised.wav")

        save_wav(noisy_signal, sample_rate, noisy_file)
        save_wav(clean_signal, sample_rate, clean_file)
        save_wav(denoised_signal, sample_rate, denoised_file)

        print(f"Saved:\n  {noisy_file}\n  {clean_file}\n  {denoised_file}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Compare test signals to denoised signals.")
    parser.add_argument("--test_signals", type=str, required=True, help="Path to test signals (npy).")
    parser.add_argument("--test_denoised", type=str, required=True, help="Path to test denoised signals (npy).")
    parser.add_argument("--names", type=str, required=True, help="Path to signal names (npy).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for WAV files.")
    parser.add_argument("--sample_rate", type=int, default=8000, help="Sample rate for WAV files.")
    args = parser.parse_args()

    compare_signals(
        test_signals_path=args.test_signals,
        test_denoised_path=args.test_denoised,
        names_path=args.names,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate
    )
