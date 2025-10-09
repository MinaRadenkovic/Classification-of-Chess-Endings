import chess

def classify_endgame(fen: str) -> str:
    """
    Classify chess endgame type based on FEN string.
    Returns endgame type description (e.g., K+P vs K, R vs P, etc.)
    Args:
        fen (str): FEN string representing the chess position  
    Returns:
        str: Endgame type classification
    """
    board = chess.Board(fen)
    pieces = board.piece_map().values()

    # Count pieces by color
    white_counts = {"K": 0, "Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}
    black_counts = {"k": 0, "q": 0, "r": 0, "b": 0, "n": 0, "p": 0}

    for piece in pieces:
        symbol = piece.symbol()
        if symbol.isupper():
            white_counts[symbol] += 1
        else:
            black_counts[symbol] += 1

    def material_string(counts: dict) -> str:
        """
        Create material description string from piece counts.
        Args:
            counts (dict): Dictionary of piece counts  
        Returns:
            str: Material description (e.g., "Q", "2R", "P")
        """
        material = []
        for piece_type, count in counts.items():
            if piece_type != "K" and count > 0:
                material.append(f"{count if count > 1 else ''}{piece_type}")
        return "+".join(material) if material else "K"

    white_material = material_string(white_counts)
    black_material = material_string({k.lower(): v for k, v in black_counts.items()})
    return f"{white_material} vs {black_material}"
