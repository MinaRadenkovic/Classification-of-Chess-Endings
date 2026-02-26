"""
FEN to Tensor Converter for Chess Endgame Classification
Converts FEN string to 8x8x12 tensor for CNN/RNN model.
"""

import numpy as np
import chess


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
