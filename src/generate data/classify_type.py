import chess

def classify_endgame(fen: str) -> str:
    """
    Classify chess endgame type based on FEN string with material balance normalization.
    Always returns classification from white's perspective (white has advantage).
    Returns endgame type description (e.g., K+Q vs K, K+P vs K, etc.)
    Args:
        fen (str): FEN string representing the chess position  
    Returns:
        str: Normalized endgame type classification
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

    def create_canonical_type(white_material, black_material):
        """
        Create canonical endgame type by normalizing material balance.
        This ensures that "Q vs k" and "K+Q vs K" become the same type.
        """
        def extract_pieces(material_str):
            """Extract non-king pieces from material string."""
            pieces = []
            for char in material_str:
                if char.upper() in ['Q', 'R', 'B', 'N', 'P']:
                    pieces.append(char.upper())
            return sorted(pieces)
        
        def create_material_signature(pieces):
            """Create a signature for the material balance."""
            if not pieces:
                return "K"
            
            # Count pieces
            counts = {}
            for piece in pieces:
                counts[piece] = counts.get(piece, 0) + 1
            
            # Create signature
            signature = []
            for piece in ['Q', 'R', 'B', 'N', 'P']:
                if piece in counts:
                    count = counts[piece]
                    if count > 1:
                        signature.append(f"{count}{piece}")
                    else:
                        signature.append(piece)
            
            return "+".join(signature) if signature else "K"
        
        # Extract pieces from both sides
        white_pieces = extract_pieces(white_material)
        black_pieces = extract_pieces(black_material)
        
        # Create signatures
        white_sig = create_material_signature(white_pieces)
        black_sig = create_material_signature(black_pieces)
        
        # Create both possible orderings
        type1 = f"{white_sig} vs {black_sig}"
        type2 = f"{black_sig} vs {white_sig}"
        
        # Return the lexicographically smaller one for consistency
        return type1 if type1 < type2 else type2

    # Create material strings
    white_material = material_string(white_counts)
    black_material = material_string({k.lower(): v for k, v in black_counts.items()})
    
    # Create canonical type to ensure consistency
    # A single, standardized shape, regardless of the arrangement of colors or order of figures
    return create_canonical_type(white_material, black_material)
