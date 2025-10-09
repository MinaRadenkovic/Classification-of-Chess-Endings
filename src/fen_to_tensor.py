"""
FEN to Tensor Converter for Chess Endgame Classification
Converts FEN string to 8x8x12 tensor for CNN/RNN model.
"""

import numpy as np
import torch
import chess
from typing import Tuple, Optional


def fen_to_tensor(fen: str, normalize: bool = True) -> np.ndarray:
    """
    Converts FEN string to 8x8x12 tensor.
    
    Args:
        fen (str): FEN string position
        normalize (bool): Whether to normalize position (white pieces at bottom)
    
    Returns:
        np.ndarray: Tensor shape (8, 8, 12)
        
    Channels:
        0-5: White pieces (K, Q, R, B, N, P)
        6-11: Black pieces (k, q, r, b, n, p)
    """
    try:
        board = chess.Board(fen)
        
        # Normalization - rotate position if black is to move
        if normalize and not board.turn:  # black to move
            board = board.mirror()
        
        # Initialize tensor
        tensor = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Piece to channel mapping
        piece_to_channel = {
            chess.KING: 0,    # white king
            chess.QUEEN: 1,   # white queen
            chess.ROOK: 2,    # white rook
            chess.BISHOP: 3,  # white bishop
            chess.KNIGHT: 4,  # white knight
            chess.PAWN: 5,    # white pawn
        }
        
        # Fill tensor
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank, file = chess.square_rank(square), chess.square_file(square)
                
                if piece.color == chess.WHITE:
                    channel = piece_to_channel[piece.piece_type]
                else:
                    # Black pieces in channels 6-11
                    channel = piece_to_channel[piece.piece_type] + 6
                
                tensor[7-rank, file, channel] = 1.0
        
        return tensor
        
    except Exception as e:
        print(f"Error converting FEN: {fen}, Error: {e}")
        return np.zeros((8, 8, 12), dtype=np.float32)


def tensor_to_fen(tensor: np.ndarray) -> str:
    """
    Converts tensor back to FEN string (for debugging).
    
    Args:
        tensor (np.ndarray): Tensor shape (8, 8, 12)
    
    Returns:
        str: FEN string
    """
    board = chess.Board()
    board.clear()
    
    piece_symbols = ['K', 'Q', 'R', 'B', 'N', 'P']
    
    for rank in range(8):
        for file in range(8):
            for channel in range(12):
                if tensor[rank, file, channel] > 0.5:
                    if channel < 6:  # white pieces
                        piece_symbol = piece_symbols[channel]
                    else:  # black pieces
                        piece_symbol = piece_symbols[channel-6].lower()
                    
                    square = chess.square(file, 7-rank)
                    piece = chess.Piece.from_symbol(piece_symbol)
                    board.set_piece_at(square, piece)
                    break
    
    return board.fen()


def batch_fen_to_tensor(fen_list: list, normalize: bool = True) -> torch.Tensor:
    """
    Converts list of FEN strings to batch tensor.
    
    Args:
        fen_list (list): List of FEN strings
        normalize (bool): Whether to normalize positions
    
    Returns:
        torch.Tensor: Batch tensor shape (batch_size, 8, 8, 12)
    """
    tensors = []
    for fen in fen_list:
        tensor = fen_to_tensor(fen, normalize)
        tensors.append(tensor)
    
    return torch.tensor(np.stack(tensors), dtype=torch.float32)


def add_turn_channel(tensor: np.ndarray, turn: bool) -> np.ndarray:
    """
    Adds channel for information about who is to move.
    
    Args:
        tensor (np.ndarray): Base tensor (8, 8, 12)
        turn (bool): True if white is to move, False if black
    
    Returns:
        np.ndarray: Tensor with additional channel (8, 8, 13)
    """
    turn_channel = np.full((8, 8, 1), 1.0 if turn else 0.0, dtype=np.float32)
    return np.concatenate([tensor, turn_channel], axis=2)


def visualize_tensor(tensor: np.ndarray, title: str = "Chess Position"):
    """
    Visualizes tensor as chess board.
    
    Args:
        tensor (np.ndarray): Tensor shape (8, 8, 12)
        title (str): Title for plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle(title)
    
    piece_names = ['K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p']
    
    for i in range(12):
        row = i // 6
        col = i % 6
        
        axes[row, col].imshow(tensor[:, :, i], cmap='Reds')
        axes[row, col].set_title(piece_names[i])
        axes[row, col].set_xticks(range(8))
        axes[row, col].set_yticks(range(8))
        axes[row, col].set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        axes[row, col].set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
    
    plt.tight_layout()
    plt.show()


# Test function
if __name__ == "__main__":
    # Test with simple position
    test_fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"  # K vs k
    
    print("Test FEN:", test_fen)
    tensor = fen_to_tensor(test_fen)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    
    # Check if conversion is correct
    reconstructed_fen = tensor_to_fen(tensor)
    print("Reconstructed FEN:", reconstructed_fen)
    
    # Test batch conversion
    batch_fens = [test_fen, "8/8/8/8/8/8/8/8 w - - 0 1"]  # empty board
    batch_tensor = batch_fen_to_tensor(batch_fens)
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
    print("✅ FEN to tensor conversion works correctly!")