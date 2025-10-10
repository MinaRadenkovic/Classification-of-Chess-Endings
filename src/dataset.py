"""
PyTorch Dataset class for chess endgames.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import pickle
import os

from fen_to_tensor import fen_to_tensor, add_turn_channel


class ChessEndgameDataset(Dataset):
    """
    PyTorch Dataset for chess endgames.
    
    Attributes:
        - fen: FEN string position
        - type: Endgame type (string)
        - wdl: Outcome (-2, 0, 2)
        - turn: Who is to move (0=black, 1=white)
    """
    
    def __init__(
        self, 
        csv_path: str,
        max_samples: Optional[int] = None,
        normalize: bool = True,
        add_turn_channel: bool = True,
        cache_tensors: bool = True
    ):
        """
        Args:
            csv_path (str): Path to CSV file
            max_samples (int, optional): Maximum number of samples
            normalize (bool): Whether to normalize positions
            add_turn_channel (bool): Whether to add turn channel
            cache_tensors (bool): Whether to cache tensors
        """
        self.csv_path = csv_path
        self.normalize = normalize
        self.add_turn_channel = add_turn_channel
        self.cache_tensors = cache_tensors
        
        # Load data
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        if max_samples:
            self.df = self.df.head(max_samples)
            print(f"Limiting to {max_samples} samples")
        
        print(f"Dataset size: {len(self.df)}")
        
        # Prepare labels
        self._prepare_labels()
        
        # Cache for tensors
        self.tensor_cache = {} if cache_tensors else None
        
        print("Dataset ready!")
    
    def _prepare_labels(self):
        """Prepares labels for training."""
        # Encode endgame type
        self.type_encoder = LabelEncoder()
        self.type_labels = self.type_encoder.fit_transform(self.df['type'])
        
        # Encode WDL into 3 classes (0=loss, 1=draw, 2=win)
        self.wdl_labels = (self.df['wdl'] + 2) // 2  # -2,0,2 -> 0,1,2
        
        # Turn information
        self.turn_labels = self.df['turn'].astype(int)
        
        print(f"Number of endgame types: {len(self.type_encoder.classes_)}")
        print(f"Types: {self.type_encoder.classes_[:10]}...")
        print(f"WDL distribution: {np.bincount(self.wdl_labels)}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a sample from the dataset.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - tensor: Position (8, 8, 12) or (8, 8, 13)
                - type_label: Endgame type (int)
                - wdl_label: Outcome (int)
        """
        if self.tensor_cache is not None and idx in self.tensor_cache:
            tensor = self.tensor_cache[idx]
        else:
            # Convert FEN to tensor
            fen = self.df.iloc[idx]['fen']
            tensor = fen_to_tensor(fen, self.normalize)
            
            # Add turn channel if needed
            if self.add_turn_channel:
                turn = self.turn_labels[idx]
                tensor = add_turn_channel(tensor, turn)
            
            # Cache tensor
            if self.tensor_cache is not None:
                self.tensor_cache[idx] = tensor
        
        # Convert to PyTorch tensor and transpose to (channels, height, width)
        tensor = torch.tensor(tensor, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1)  # (height, width, channels) -> (channels, height, width)
        type_label = torch.tensor(self.type_labels[idx], dtype=torch.long)
        wdl_label = torch.tensor(self.wdl_labels[idx], dtype=torch.long)
        
        return tensor, type_label, wdl_label
    
    def get_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns class weights for balanced loss.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Weights for type and wdl classes
        """
        # Weights for endgame type
        type_counts = np.bincount(self.type_labels)
        type_weights = 1.0 / type_counts
        type_weights = type_weights / type_weights.sum() * len(type_weights)
        
        # Weights for WDL
        wdl_counts = np.bincount(self.wdl_labels)
        wdl_weights = 1.0 / wdl_counts
        wdl_weights = wdl_weights / wdl_weights.sum() * len(wdl_weights)
        
        return torch.tensor(type_weights, dtype=torch.float32), torch.tensor(wdl_weights, dtype=torch.float32)
    
    def save_encoders(self, save_path: str):
        """Saves encoders for later use."""
        encoders = {
            'type_encoder': self.type_encoder,
            'type_classes': self.type_encoder.classes_,
            'num_type_classes': len(self.type_encoder.classes_),
            'num_wdl_classes': 3
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(encoders, f)
        
        print(f"Encoders saved to {save_path}")
    
    @classmethod
    def load_encoders(cls, load_path: str) -> Dict[str, Any]:
        """Loads encoders."""
        with open(load_path, 'rb') as f:
            encoders = pickle.load(f)
        
        print(f"Encoders loaded from {load_path}")
        return encoders


def create_data_loaders(
    csv_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_samples: Optional[int] = None,
    normalize: bool = True,
    add_turn_channel: bool = True,
    num_workers: int = 4,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, ChessEndgameDataset]:
    """
    Creates train/val/test DataLoaders.
    
    Args:
        csv_path (str): Path to CSV file
        batch_size (int): Batch size
        train_ratio (float): Percentage for training
        val_ratio (float): Percentage for validation
        test_ratio (float): Percentage for testing
        max_samples (int, optional): Maximum number of samples
        normalize (bool): Whether to normalize positions
        add_turn_channel (bool): Whether to add turn channel
        num_workers (int): Number of worker processes
        random_state (int): Random seed
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, ChessEndgameDataset]:
            train_loader, val_loader, test_loader, full_dataset
    """
    # Load full dataset
    full_dataset = ChessEndgameDataset(
        csv_path=csv_path,
        max_samples=max_samples,
        normalize=normalize,
        add_turn_channel=add_turn_channel
    )
    
    # Split into train/val/test
    train_size = int(len(full_dataset) * train_ratio)
    val_size = int(len(full_dataset) * val_ratio)
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, temp_dataset = train_test_split(
        full_dataset, 
        train_size=train_size, 
        random_state=random_state
    )
    
    val_dataset, test_dataset = train_test_split(
        temp_dataset,
        train_size=val_size,
        random_state=random_state
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory for CPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory for CPU
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory for CPU
    )
    
    return train_loader, val_loader, test_loader, full_dataset


# Test function
if __name__ == "__main__":
    # Test with small dataset
    csv_path = "data/generated_data.csv"
    
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader, dataset = create_data_loaders(
        csv_path=csv_path,
        batch_size=16,
        max_samples=1000,  # Test with small number
        num_workers=0  # For Windows
    )
    
    # Test batch
    print("\nTest batch:")
    for batch_idx, (tensors, type_labels, wdl_labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Tensor shape: {tensors.shape}")
        print(f"  Type labels shape: {type_labels.shape}")
        print(f"  WDL labels shape: {wdl_labels.shape}")
        print(f"  Type labels: {type_labels[:5]}")
        print(f"  WDL labels: {wdl_labels[:5]}")
        
        if batch_idx >= 2:  # Only first 3 batches
            break
    
    # Test class weights
    type_weights, wdl_weights = dataset.get_class_weights()
    print(f"\nType weights shape: {type_weights.shape}")
    print(f"WDL weights shape: {wdl_weights.shape}")
    
    print("✅ Dataset class works correctly!")