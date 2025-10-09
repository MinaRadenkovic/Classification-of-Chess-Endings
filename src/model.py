"""
CNN+RNN model for chess endgame classification.
Multi-task learning: endgame type + outcome (WDL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ChessEndgameCNN(nn.Module):
    """
    CNN part for extracting features from chess board.
    """
    
    def __init__(self, input_channels: int = 12):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout2d(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, channels, 8, 8)
        
        Returns:
            torch.Tensor: Features (batch_size, 256, 2, 2)
        """
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 4x4 -> 2x2
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        return x


class ChessEndgameRNN(nn.Module):
    """
    RNN part for processing sequential features.
    """
    
    def __init__(self, input_size: int = 256, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Output size is 2 * hidden_size due to bidirectional
        self.output_size = 2 * hidden_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            torch.Tensor: Output (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take last output
        output = lstm_out[:, -1, :]  # (batch_size, output_size)
        
        return output


class ChessEndgameModel(nn.Module):
    """
    Main model for chess endgame classification.
    Multi-task: endgame type + outcome (WDL).
    """
    
    def __init__(
        self,
        input_channels: int = 12,
        num_type_classes: int = 100,
        num_wdl_classes: int = 3,
        cnn_hidden: int = 256,
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_type_classes = num_type_classes
        self.num_wdl_classes = num_wdl_classes
        
        # CNN part
        self.cnn = ChessEndgameCNN(input_channels)
        
        # RNN part
        self.rnn = ChessEndgameRNN(
            input_size=cnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers
        )
        
        # Fully connected slojevi
        self.fc = nn.Sequential(
            nn.Linear(self.rnn.output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads
        self.type_head = nn.Linear(256, num_type_classes)
        self.wdl_head = nn.Linear(256, num_wdl_classes)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, channels, 8, 8)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - type_logits: Logits for endgame type
                - wdl_logits: Logits for outcome (WDL)
        """
        batch_size = x.size(0)
        
        # CNN forward pass
        cnn_out = self.cnn(x)  # (batch_size, 256, 2, 2)
        
        # Reshape for RNN
        # Treat each 2x2 region as a sequence
        cnn_out = cnn_out.view(batch_size, 256, 4)  # 4 = 2*2
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, 4, 256)
        
        # RNN forward pass
        rnn_out = self.rnn(cnn_out)  # (batch_size, 256)
        
        # Fully connected
        fc_out = self.fc(rnn_out)  # (batch_size, 256)
        
        # Output heads
        type_logits = self.type_head(fc_out)
        wdl_logits = self.wdl_head(fc_out)
        
        return type_logits, wdl_logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction with softmax.
        
        Args:
            x: Input tensor (batch_size, channels, 8, 8)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - type_probs: Probabilities for endgame type
                - wdl_probs: Probabilities for outcome (WDL)
        """
        type_logits, wdl_logits = self.forward(x)
        
        type_probs = F.softmax(type_logits, dim=1)
        wdl_probs = F.softmax(wdl_logits, dim=1)
        
        return type_probs, wdl_probs


class ChessEndgameLoss(nn.Module):
    """
    Loss function for multi-task learning.
    """
    
    def __init__(
        self,
        type_weight: float = 1.0,
        wdl_weight: float = 1.0,
        type_class_weights: Optional[torch.Tensor] = None,
        wdl_class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.type_weight = type_weight
        self.wdl_weight = wdl_weight
        
        # Cross-entropy loss with class weights
        self.type_loss = nn.CrossEntropyLoss(weight=type_class_weights)
        self.wdl_loss = nn.CrossEntropyLoss(weight=wdl_class_weights)
    
    def forward(
        self,
        type_logits: torch.Tensor,
        wdl_logits: torch.Tensor,
        type_labels: torch.Tensor,
        wdl_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            type_logits: Logits za tip završnice
            wdl_logits: Logits za ishod (WDL)
            type_labels: True labele za tip
            wdl_labels: True labele za WDL
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - total_loss: Total loss
                - type_loss: Loss for type
                - wdl_loss: Loss for WDL
        """
        type_loss = self.type_loss(type_logits, type_labels)
        wdl_loss = self.wdl_loss(wdl_logits, wdl_labels)
        
        total_loss = self.type_weight * type_loss + self.wdl_weight * wdl_loss
        
        return total_loss, type_loss, wdl_loss


def create_model(
    num_type_classes: int,
    num_wdl_classes: int = 3,
    input_channels: int = 12,
    device: str = "cpu"
) -> Tuple[ChessEndgameModel, ChessEndgameLoss]:
    """
    Creates model and loss function.
    
    Args:
        num_type_classes (int): Number of endgame types
        num_wdl_classes (int): Number of WDL classes (3)
        input_channels (int): Number of input channels
        device (str): Device for model
    
    Returns:
        Tuple[ChessEndgameModel, ChessEndgameLoss]: Model and loss function
    """
    model = ChessEndgameModel(
        input_channels=input_channels,
        num_type_classes=num_type_classes,
        num_wdl_classes=num_wdl_classes
    )
    
    loss_fn = ChessEndgameLoss()
    
    # Move to device
    model = model.to(device)
    
    return model, loss_fn


# Test function
if __name__ == "__main__":
    print("Testing model...")
    
    # Test with small batch
    batch_size = 4
    input_channels = 12
    
    # Create model
    model, loss_fn = create_model(
        num_type_classes=50,
        num_wdl_classes=3,
        input_channels=input_channels
    )
    
    print(f"Model created:")
    print(f"  Input channels: {input_channels}")
    print(f"  Type classes: 50")
    print(f"  WDL classes: 3")
    
    # Test forward pass
    x = torch.randn(batch_size, input_channels, 8, 8)
    type_logits, wdl_logits = model(x)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Type logits shape: {type_logits.shape}")
    print(f"  WDL logits shape: {wdl_logits.shape}")
    
    # Test loss
    type_labels = torch.randint(0, 50, (batch_size,))
    wdl_labels = torch.randint(0, 3, (batch_size,))
    
    total_loss, type_loss, wdl_loss = loss_fn(type_logits, wdl_logits, type_labels, wdl_labels)
    
    print(f"\nLoss:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Type loss: {type_loss.item():.4f}")
    print(f"  WDL loss: {wdl_loss.item():.4f}")
    
    # Test predictions
    type_probs, wdl_probs = model.predict(x)
    
    print(f"\nPredictions:")
    print(f"  Type probs shape: {type_probs.shape}")
    print(f"  WDL probs shape: {wdl_probs.shape}")
    print(f"  Type probs sum: {type_probs.sum(dim=1)}")
    print(f"  WDL probs sum: {wdl_probs.sum(dim=1)}")
    
    # Number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("✅ Model works correctly!")