import torch
import torchaudio
import numpy as np
import os
import glob
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ------------Note------------
#    You should have this package installed "soundfile", if you don't have it, run "pip install soundfile" in your terminal.
#    and then restart your vscode terminal. 


# Dynamically find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ESC50_PATH = os.path.join(DATA_DIR, "ESC-50-master", "audio")

class SpeechNoiseDataset(Dataset):
    def __init__(self, speech_ds, noise_dir, noise_level=0.0, sample_rate=16000, transform=None):
        self.speech_ds = speech_ds
        self.noise_level = noise_level
        self.sample_rate = sample_rate
        self.transform = transform
        
        # Load noise files
        self.noise_files = glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
        
        # Create a mapping from word strings to integers
        # This is essential for the CNN and RNN classifiers
        self.labels = sorted(list(set(datapoint[2] for datapoint in speech_ds.dataset)))
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

    def _mix_noise(self, speech_wave, noise_level):
        if noise_level == 0 or not self.noise_files:
            return speech_wave
        
        # Load random noise and resample to 16kHz
        noise_idx = np.random.randint(0, len(self.noise_files))
        noise_wave, sr_n = torchaudio.load(self.noise_files[noise_idx])
        
        if sr_n != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr_n, self.sample_rate)
            noise_wave = resampler(noise_wave)
            
        # Match lengths (crop or pad)
        if noise_wave.shape[1] > speech_wave.shape[1]:
            start = np.random.randint(0, noise_wave.shape[1] - speech_wave.shape[1])
            noise_wave = noise_wave[:, start:start + speech_wave.shape[1]]
        else:
            pad = speech_wave.shape[1] - noise_wave.shape[1]
            noise_wave = torch.nn.functional.pad(noise_wave, (0, pad))

        # RMS Mixing to achieve the desired noise level
        speech_rms = torch.sqrt(torch.mean(speech_wave**2))
        noise_rms = torch.sqrt(torch.mean(noise_wave**2))
        
        if noise_rms > 0:
            factor = (speech_rms / noise_rms) * noise_level
            mixed_wave = speech_wave + (noise_wave * factor)
            # Normalize to avoid clipping
            mixed_wave = mixed_wave / (mixed_wave.abs().max() + 1e-7)
            return mixed_wave
        return speech_wave

    def __len__(self):
        return len(self.speech_ds)

    def __getitem__(self, idx):
        waveform, sr, label_str, _, _ = self.speech_ds[idx]
        
        # Standardize speech length to exactly 1 second (16000 samples)
        if waveform.shape[1] < self.sample_rate:
            waveform = torch.nn.functional.pad(waveform, (0, self.sample_rate - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.sample_rate]

        # Apply noise and convert label to integer
        mixed_waveform = self._mix_noise(waveform, self.noise_level)
        label_idx = self.label_to_idx[label_str]
        
        if self.transform:
            mixed_waveform = self.transform(mixed_waveform)
            
        return mixed_waveform, torch.tensor(label_idx)

def get_dataloaders(noise_dir=ESC50_PATH, batch_size=32, noise_level=0.0, transform=None, train_split=0.8, val_split=0.1):
    if train_split + val_split >= 1.0:
        raise ValueError("train_split and val_split must sum to less than 1.0")
    
    # Load Google Speech Commands
    base_ds = torchaudio.datasets.SPEECHCOMMANDS(root=DATA_DIR, download=True)
    
    # Default 80/10/10 Split (Train/Val/Test)
    total = len(base_ds)
    train_len = int(train_split * total)
    val_len = int(val_split * total)
    test_len = total - train_len - val_len
    
    train_b, val_b, test_b = torch.utils.data.random_split(base_ds, [train_len, val_len, test_len])
    
    # Return loaders for the specific noise levels required: 0%, 10%, or 50%
    train_loader = DataLoader(SpeechNoiseDataset(train_b, noise_dir, noise_level, transform=transform), 
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SpeechNoiseDataset(val_b, noise_dir, noise_level, transform=transform), 
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SpeechNoiseDataset(test_b, noise_dir, noise_level, transform=transform), 
                            batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


import matplotlib.pyplot as plt

def test_dataset_logic():
    print("--- Starting Detailed Logic Test ---")
    
    # 0.1 noise level for the first check
    try:
        train_loader, _, _ = get_dataloaders(noise_level=0.1, batch_size=1)
        dataset = train_loader.dataset
    except Exception as e:
        print(f"Failed to initialize loaders: {e}")
        return

    # 2. Test Label Mapping
    # Verify that we have the expected number of classes (usually 30 or 35)
    num_classes = len(dataset.label_to_idx)
    print(f"Detected {num_classes} unique speech command classes.")
    
    # 3. Test Output Shapes and Types
    waveform, label = dataset[0]
    print(f"Waveform shape: {waveform.shape}") # Should be [1, 16000]
    print(f"Label type: {type(label)}, Value: {label}") # Should be torch.Tensor (int)

    # 4. Verification of Noise Levels (0%, 10%, 50%)
    levels = [0.0, 0.1, 0.5]
    results = {}

    for level in levels:
        # Update dataset noise level temporarily for testing
        dataset.noise_level = level
        wave, _ = dataset[0]
        
        rms = torch.sqrt(torch.mean(wave**2))
        max_val = wave.abs().max()
        results[level] = (rms.item(), max_val.item())
        
        print(f"Level {int(level*100)}% -> RMS: {rms:.4f}, Max: {max_val:.4f}")

    # 5. Normalization Check
    # Ensure no audio is clipping (max should be <= 1.0 due to our normalization)
    for level, (rms, max_v) in results.items():
        if max_v > 1.01: # Small epsilon for float math
            print(f"Warning: Clipping detected at {level*100}% noise!")
        else:
            print(f"Normalization OK at {level*100}% noise.")

    # 6. Visual Verification (Optional)
    # Plotting the three levels to see the noise additive effect
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    for i, level in enumerate(levels):
        dataset.noise_level = level
        wave, _ = dataset[0]
        axs[i].plot(wave[0].numpy())
        axs[i].set_title(f"Waveform at {int(level*100)}% Noise")
        axs[i].set_ylim([-1, 1])
    
    plt.tight_layout()
    plt.show()
    print("--- Test Complete ---")

# -----------Test code to verify dataset logic and noise mixing-----------

if __name__ == "__main__":
    test_dataset_logic()