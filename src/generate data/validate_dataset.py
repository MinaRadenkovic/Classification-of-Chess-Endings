import chess
import csv

def validate_dataset(csv_file):
    """Validate FEN positions in the dataset"""
    invalid_count = 0
    total_count = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print('Header:', header)
        
        for i, row in enumerate(reader):
            if i >= 250000:  # Check first 20 rows
                break
                
            fen = row[0]
            total_count += 1
            
            try:
                board = chess.Board(fen)
                if not board.is_valid():
                    print(f'Invalid FEN: {fen}')
                    invalid_count += 1
                else:
                    print(f'Valid FEN: {fen}')
            except Exception as e:
                print(f'Error parsing FEN: {fen} - {e}')
                invalid_count += 1
    
    print(f'\nSummary: {invalid_count}/{total_count} invalid positions')
    return invalid_count == 0

if __name__ == "__main__":
    validate_dataset('data/generated_data.csv')
