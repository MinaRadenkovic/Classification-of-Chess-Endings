import chess
import chess.syzygy
import random
import csv
import os
import glob
import logging
from tqdm import tqdm
from classify_type import classify_endgame

# Configuration constants
# Use os.path to create correct path relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TABLEBASE_DIR = os.path.join(PROJECT_ROOT, "data", "syzygy", "3-4-5")
BOARD_SIZE = 64  # Number of squares on chess board (0-63)
MAX_PIECES_DEFAULT = 6  # Maximum pieces in endgame position (2 kings and 4 other pieces)
DEFAULT_SAMPLES = 250000  # Default number of samples to generate
REQUIRED_KINGS = 2  # Both kings are mandatory in any position

# All possible chess pieces (limited to <6 total pieces)
PIECES = ['K', 'Q', 'R', 'B', 'N', 'P']

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def random_piece():
    """Return a random chess piece symbol."""
    return random.choice(PIECES)

def random_square(used):
    """
    Generate a random square number (0-63) that hasn't been used yet.
    Args:
        used (set): Set of already used square numbers
    Returns:
        int: Random square number that's not in the used set
    """
    sq = random.randint(0, BOARD_SIZE - 1)
    while sq in used:
        sq = random.randint(0, BOARD_SIZE - 1)
    used.add(sq)
    return sq

def random_endgame_position(max_pieces=MAX_PIECES_DEFAULT):
    """
    Generate a random endgame position with specified maximum number of pieces.
    Args:
        max_pieces (int): Maximum number of pieces on the board (default: 5)
    Returns:
        chess.Board: Randomly generated chess board position
    """
    board = chess.Board(None)  # Empty board
    used = set()

    # Kings are mandatory in any position
    board.set_piece_at(random_square(used), chess.Piece.from_symbol('K'))
    board.set_piece_at(random_square(used), chess.Piece.from_symbol('k'))

    # Add random additional pieces
    num_extra = random.randint(0, max_pieces - REQUIRED_KINGS)
    for _ in range(num_extra):
        symbol = random.choice([p for p in PIECES if p != 'K'])
        color = random.choice([True, False])  # True = white, False = black
        sym = symbol if color else symbol.lower()
        board.set_piece_at(random_square(used), chess.Piece.from_symbol(sym))

    # Randomly choose which player's turn it is
    board.turn = random.choice([chess.WHITE, chess.BLACK])
    
    return board

def validate_tablebase_directory():
    """
    Validate that the tablebase directory exists and contains required files.
    Returns:
        bool: True if directory is valid, False otherwise
    """
    if not os.path.exists(TABLEBASE_DIR):
        logger.error(f"Tablebase directory not found: {TABLEBASE_DIR}")
        return False
    
    # Check for at least one .rtbw or .rtbz file
    tablebase_files = []
    for ext in ['*.rtbw', '*.rtbz']:
        tablebase_files.extend(glob.glob(os.path.join(TABLEBASE_DIR, ext)))
    
    if not tablebase_files:
        logger.error(f"No tablebase files found in {TABLEBASE_DIR}")
        return False
    
    logger.info(f"Found {len(tablebase_files)} tablebase files")
    return True

def generate_dataset(out_csv="data/generated_data.csv", n_samples=DEFAULT_SAMPLES, max_pieces=MAX_PIECES_DEFAULT):
    """
    Generate a dataset of random endgame positions with their tablebase evaluations.
    Ensures all positions are unique by checking FEN strings.
    
    Args:
        out_csv (str): Output CSV file path
        n_samples (int): Number of unique samples to generate
        max_pieces (int): Maximum number of pieces per position
    """
    # Validate tablebase directory before starting
    if not validate_tablebase_directory():
        logger.error("Cannot proceed without valid tablebase directory")
        return
    
    logger.info(f"Starting dataset generation: {n_samples} unique samples, max {max_pieces} pieces")
    
    # Set to store unique FEN strings for duplicate checking
    seen_positions = set()
    
    try:
        with chess.syzygy.open_tablebase(TABLEBASE_DIR) as tb, open(out_csv, "w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["fen", "type", "wdl", "dtz", "num_pieces", "turn"])
            
            pbar = tqdm(range(n_samples), desc="Generating unique positions")
            count = 0
            skipped_count = 0
            duplicate_count = 0
            total_attempts = 0
            
            while count < n_samples:
                total_attempts += 1
                board = random_endgame_position(max_pieces=max_pieces)
                fen = board.fen()
                
                # Check for duplicates
                if fen in seen_positions:
                    duplicate_count += 1
                    continue
                
                # Validate the generated position
                try:
                    if not board.is_valid():
                        skipped_count += 1
                        continue
                except Exception:
                    skipped_count += 1
                    continue
                
                num_pieces = len(board.piece_map())
                
                try:
                    # Query tablebase for position evaluation
                    wdl = tb.probe_wdl(board)
                    dtz = tb.probe_dtz(board)
                except chess.syzygy.MissingTableError:
                    # Position not covered by tablebase
                    skipped_count += 1
                    continue
                except Exception as e:
                    # Handle all other errors (invalid positions, corrupted tables, etc.)
                    logger.debug(f"Skipping position due to error: {e}")
                    skipped_count += 1
                    continue
                
                # Add to seen positions and write to CSV
                seen_positions.add(fen)
                end_type = classify_endgame(fen)
                writer.writerow([fen, end_type, wdl, dtz, num_pieces, int(board.turn)])
                count += 1
                
                # Update progress bar with current stats
                pbar.set_postfix({
                    'Unique': count,
                    'Skipped': skipped_count,
                    'Duplicates': duplicate_count,
                    'Total Attempts': total_attempts,
                    'Success Rate': f"{count/total_attempts*100:.1f}%" if total_attempts > 0 else "0%"
                })
                
                # Update progress bar to show current count
                pbar.n = count
                pbar.refresh()
            
            logger.info(f"✅ Dataset generation completed!")
            logger.info(f"   Generated: {count} unique positions")
            logger.info(f"   Skipped: {skipped_count} positions")
            logger.info(f"   Duplicates found: {duplicate_count} positions")
            logger.info(f"   Total attempts: {total_attempts}")
            logger.info(f"   Success rate: {count/total_attempts*100:.1f}%" if total_attempts > 0 else "0%")
            logger.info(f"   Saved to: {out_csv}")
            
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during dataset generation: {e}")

if __name__ == "__main__":
    # Generate dataset with default parameters
    # Use os.path to create correct path relative to script location
    output_path = os.path.join(PROJECT_ROOT, "data", "generated_data.csv")
    generate_dataset(out_csv=output_path, n_samples=DEFAULT_SAMPLES)
