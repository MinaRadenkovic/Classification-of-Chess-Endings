# Chess Endgame Classification using Deep Learning

A comprehensive deep learning system for classifying chess endgame positions and predicting game outcomes using CNN+RNN architecture.

## 🎯 Project Overview

This project implements an intelligent chess endgame classifier that:
- **Classifies endgame types** (e.g., "King+Pawn vs King", "Rook vs Pawn", "Two Rooks vs King")
- **Predicts game outcomes** (Win/Draw/Loss) with probability estimates
- **Provides educational interface** for chess players and trainers

The system uses Syzygy tablebase data and implements a multi-task learning approach with CNN+RNN architecture for robust endgame analysis.

## 📋 Project Specification

**Project Title**: Chess Endgame Classification using Computer Intelligence Methods  
**Author**: Mina Radenković SV76/2022  
**Course**: Computer Intelligence Fundamentals  

### Problem Definition
The problem we solve is automatic recognition and classification of chess endgames. Input is a chess position in FEN (Forsyth–Edwards Notation) format, and output is:
- Classification of endgame type (e.g., "Rook vs Pawn", "King and Pawn vs King", "Two Rooks vs King", "Rook + Bishop vs Rook")
- Assessment of win/draw probability

This way the system helps players understand what type of endgame is being played and how to approach the position.

### Motivation
Endgames are one of the most difficult segments of chess to learn. Although theoretical tablebases exist (with completely accurate solutions), they are not user-friendly for players who want explanations and training.

### Practical Applications
- Help amateurs and club players practice endgames
- Education in chess schools and online tutorials
- System can serve for game analysis on platforms like Lichess or Chess.com

### Dataset
We use Lichess open database (https://database.lichess.org/), which contains millions of games in PGN format.
Also, there are endgame tablebases (e.g., Syzygy, https://syzygy-tables.info/) with positions and accurate outcomes.

### Methodology
1. Position extraction → PGN → FEN
2. Preprocessing and labeling of endgame types
3. Modeling:
   - CNN/RNN for chess board representation (as images or 8×8×N matrices)
   - Endgame type classification
   - Outcome prediction (win/draw/loss)
4. Training and evaluation
5. Interface development – simple tutor (enter FEN position → get endgame class, evaluation and advice)

### Technologies
- Python
- scikit-learn and PyTorch (for classification and prediction)
- python-chess library (https://python-chess.readthedocs.io/) for FEN/PGN work
- pgn-extract for data extraction
- pandas, NumPy for data work
- Matplotlib/Seaborn for visualizations

## 🚀 Features

### Core Functionality
- **Endgame Type Classification**: Identifies the specific type of endgame position
- **Outcome Prediction**: Predicts Win/Draw/Loss with confidence scores
- **Multi-task Learning**: Simultaneous classification and outcome prediction
- **Educational Interface**: Interactive CLI for chess players

### Technical Features
- **CNN+RNN Architecture**: Convolutional layers for spatial features + LSTM for sequential patterns
- **FEN to Tensor Conversion**: Efficient representation of chess positions
- **Position Normalization**: Consistent representation regardless of player perspective
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1-score, confusion matrices

## 📊 Dataset

- **Source**: Syzygy endgame tablebase (3-4-5 piece positions)
- **Size**: ~250,000 unique endgame positions
- **Format**: FEN strings with tablebase evaluations
- **Features**: Position, endgame type, WDL outcome, DTZ, piece count, turn information

### Dataset Statistics
- **Endgame Types**: 100+ different endgame classifications
- **Outcomes**: Balanced distribution of Win/Draw/Loss
- **Pieces**: 2-6 pieces per position (including kings)
- **Validation**: All positions validated against Syzygy tablebase

## 🏗️ Architecture

### Model Architecture
```
Input: FEN String → 8×8×12 Tensor
    ↓
CNN Layers (Spatial Feature Extraction)
    ↓
RNN Layers (Sequential Pattern Recognition)
    ↓
Multi-task Output:
├── Endgame Type Classification
└── Win/Draw/Loss Prediction
```

### Key Components
- **ChessEndgameCNN**: Convolutional layers with batch normalization
- **ChessEndgameRNN**: Bidirectional LSTM for sequence processing
- **Multi-task Loss**: Combined classification and outcome prediction
- **Position Normalization**: Consistent representation across perspectives

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Syzygy tablebase files (3-4-5 piece positions)

### Setup
```bash

# Install dependencies
pip install -r requirements.txt


### Dependencies
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
python-chess>=1.9.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.60.0
```

## 🚀 Usage

### 1. Generate Dataset
```bash
python src/generate\ data/generate_dataset.py
```

### 2. Train Model
```bash
python main.py train --csv-path data/generated_data.csv --save-dir checkpoints --epochs 50
```

### 3. Evaluate Model
```bash
python main.py evaluate --model-path checkpoints/final_model.pth --csv-path data/generated_data.csv
```

### 4. Interactive Prediction
```bash
python main.py interactive --model-path checkpoints/final_model.pth --encoders-path checkpoints/encoders.pkl
```

### 5. Single Position Prediction
```bash
python main.py predict --model-path checkpoints/final_model.pth --encoders-path checkpoints/encoders.pkl --fen "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
```

## 📁 Project Structure

```
chess-endgame-classifier/
├── src/
│   ├── dataset.py              # PyTorch Dataset implementation
│   ├── model.py                # CNN+RNN model architecture
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Model evaluation
│   ├── interface.py             # CLI interface
│   ├── fen_to_tensor.py        # FEN to tensor conversion
│   └── generate data/
│       ├── generate_dataset.py # Dataset generation
│       └── classify_type.py    # Endgame type classification
├── data/
│   ├── generated_data.csv      # Generated dataset
│   └── syzygy/                 # Syzygy tablebase files
├── main.py                     # Main entry point
├── analyze_dataset.py          # Dataset analysis
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔧 Configuration

### Training Parameters
- **Batch Size**: 32 (configurable)
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Epochs**: 50 (with early stopping)
- **Train/Val/Test Split**: 70%/15%/15%
- **Optimizer**: Adam with weight decay

### Model Parameters
- **Input Channels**: 12 (6 piece types × 2 colors) + optional turn channel
- **CNN Filters**: 64 → 128 → 256
- **RNN Hidden Size**: 128 (bidirectional)
- **Dropout**: 0.2-0.3 for regularization

## 📈 Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted and macro F1 scores
- **Precision/Recall**: Per-class metrics
- **Top-k Accuracy**: Top-3 and Top-5 accuracy

### Outcome Prediction Metrics
- **WDL Accuracy**: Win/Draw/Loss prediction accuracy
- **Confusion Matrix**: Detailed error analysis
- **Error Analysis**: Most common misclassifications

## 🎓 Educational Value

This project serves as an excellent educational resource for:
- **Chess Players**: Understanding endgame patterns and outcomes
- **ML Students**: Learning multi-task learning and CNN+RNN architectures
- **Researchers**: Exploring chess AI and game theory applications

### Use Cases
- **Chess Training**: Analyze endgame positions and learn optimal play
- **Game Analysis**: Understand endgame theory and patterns
- **Educational Tools**: Interactive learning for chess students
- **Research Platform**: Foundation for advanced chess AI research

## 🔬 Technical Details

### FEN to Tensor Conversion
- **Input**: FEN string (e.g., "4k3/8/8/8/8/8/8/4K3 w - - 0 1")
- **Output**: 8×8×12 tensor representing piece positions
- **Normalization**: Consistent representation regardless of perspective
- **Turn Information**: Optional additional channel for turn data

### Model Architecture Details
- **CNN**: 3 convolutional layers with batch normalization and max pooling
- **RNN**: 2-layer bidirectional LSTM with dropout
- **Multi-task**: Separate heads for classification and outcome prediction
- **Regularization**: Dropout, weight decay, and early stopping

## 🚧 Future Enhancements

### Potential Improvements
- **Web Interface**: Browser-based interactive tool
- **Mobile App**: Chess training application
- **Advanced Features**: Move suggestions and tactical analysis
- **Extended Dataset**: More complex endgame positions
- **Real-time Analysis**: Live game position analysis

### Research Directions
- **Transformer Architecture**: Attention-based position analysis
- **Reinforcement Learning**: Self-play endgame training
- **Multi-modal Learning**: Combining position and move history
- **Explainable AI**: Understanding model decision-making

## 📚 References

- **Syzygy Tablebase**: [https://syzygy-tables.info/](https://syzygy-tables.info/)
- **Python-Chess Library**: [https://python-chess.readthedocs.io/](https://python-chess.readthedocs.io/)
- **DeepChess Paper**: [https://arxiv.org/abs/1711.09667](https://arxiv.org/abs/1711.09667)
- **AlphaZero Paper**: [https://arxiv.org/abs/1712.01815](https://arxiv.org/abs/1712.01815)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Format code
black src/
```

## 👥 Authors

- **Mina Radenković**
- **Course**: Computer Intelligence
- **Institution**: FTN, University of Novi Sad

## 🙏 Acknowledgments

- Syzygy tablebase developers for providing accurate endgame data
- Python-Chess community for excellent chess library
- PyTorch team for deep learning framework
- Chess community for inspiration and feedback

---

**Note**: This project is designed for educational purposes and chess training. For competitive play, always consult official chess engines and databases.