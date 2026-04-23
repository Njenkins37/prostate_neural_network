import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset


class PICAI25DDataset(Dataset):
    """
    Dynamically loads 3D .npz files and extracts 2.5D slices (z-1, z, z+1) 
    to feed into the Dual-Encoder U-Net.
    """
    def __init__(self, manifest_path, data_dir, split="train", pad_edges=True):
        self.data_dir = Path(data_dir)
        self.pad_edges = pad_edges
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        self.patient_ids = manifest["splits"][split]
        self.samples = []
        
        # Build the Index.
        # We iterate through patients and log how many Z-slices each has.
        # This prevents us from loading 100GB of arrays into RAM at once.
        for pid in self.patient_ids:
            npz_path = self.data_dir / f"{pid}_clean.npz"
            if not npz_path.exists():
                continue
            
            # Fast-load just the metadata to get the Z-axis depth.
            with np.load(npz_path) as data:
                num_slices = data['t2'].shape[-1]

            # Parameterized Edge Handling.
            start_z = 0 if self.pad_edges else 1
            end_z = num_slices if self.pad_edges else num_slices - 1

            # Create an index entry for every valid slice.
            for z in range(start_z, end_z):
                self.samples.append((pid, z, num_slices))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # Look up which patient and which slice we are loading.
        pid, z, num_slices = self.samples[idx]
        npz_path = self.data_dir / f"{pid}_clean.npz"
        
        # Load the actual 3D arrays.
        with np.load(npz_path) as data:
            t2_vol     = data['t2'].astype(np.float32)
            adc_vol    = data['adc'].astype(np.float32)
            lesion_vol = data['lesion']

        # The 2.5D Slicing Logic (z-1, z, z+1).
        z_min = max(0, z - 1)
        z_max = min(num_slices - 1, z + 1)
        
        t2_slices = t2_vol[..., z_min:z_max+1]
        adc_slices = adc_vol[..., z_min:z_max+1]
        
        # Edge-Case Padding: Only triggers if pad_edges=True and z is at boundaries.
        # If we are at the very bottom (z=0) or top (z=Max) of the prostate,
        # we only have 2 slices. We duplicate the edge slice to force exactly 3 channels.
        if self.pad_edges:
            if z == 0:
                t2_slices  = np.concatenate([t2_slices[..., 0:1],  t2_slices], axis=-1)
                adc_slices = np.concatenate([adc_slices[..., 0:1], adc_slices], axis=-1)
            elif z == num_slices - 1:
                t2_slices  = np.concatenate([t2_slices,  t2_slices[..., -1:]], axis=-1)
                adc_slices = np.concatenate([adc_slices, adc_slices[..., -1:]], axis=-1)
            
        # The Ground Truth Mask.
        # The model is predicting the tumor for the MIDDLE slice only.
        mask = lesion_vol[..., z]
        
        # Convert to PyTorch Tensors.
        # Numpy arrays are (H, W, Channels). PyTorch requires (Channels, H, W).
        t2_tensor = torch.from_numpy(np.transpose(t2_slices, (2, 0, 1))).float()
        adc_tensor = torch.from_numpy(np.transpose(adc_slices, (2, 0, 1))).float()
        
        # Add a channel dimension to the mask -> (1, H, W).
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Enforce strict binary masking in case resamplers introduced decimals.
        # When we save binary NumPy arrays into compressed .npz archives 
        # and then load them back into PyTorch's 32-bit floating-point tensors, 
        # PyTorch can sometimes interpret a 0 as 0.0000001 or a 1 as 0.9999999.
        mask_tensor = (mask_tensor > 0.5).float()
        
        return t2_tensor, adc_tensor, mask_tensor