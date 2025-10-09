""" Main file for running the chess endgame classifier. """ 
import os 
import sys 
import argparse 
import torch 
from pathlib import Path 

# Add src to path 
sys.path.append('src')

from dataset import create_data_loaders, ChessEndgameDataset 
from model import create_model 
from train import ChessEndgameTrainer 
from evaluate import ChessEndgameEvaluator 
from interface import ChessEndgameInterface 

def train_model(args): 
    """Trains the model.""" 

    print("=== TRAINING MODEL ===") 

    # Create DataLoaders  
    print("Creating DataLoaders...") 
    train_loader, val_loader, test_loader, dataset = create_data_loaders( csv_path=args.csv_path, batch_size=args.batch_size, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, max_samples=args.max_samples, normalize=args.normalize, add_turn_channel=args.add_turn_channel, num_workers=args.num_workers )
    
    # Create model 
    print("Creating model...") 
    model, loss_fn = create_model( num_type_classes=len(dataset.type_encoder.classes_), num_wdl_classes=3, input_channels=12 if args.add_turn_channel else 13, device=args.device ) 
    
    # Create trainer 
    trainer = ChessEndgameTrainer( model=model, loss_fn=loss_fn, device=args.device, save_dir=args.save_dir ) 
    # Train 
    trainer.train( train_loader=train_loader, val_loader=val_loader, epochs=args.epochs, save_every=args.save_every, early_stopping_patience=args.early_stopping_patience ) 
    # Save encoders 
    dataset.save_encoders(os.path.join(args.save_dir, "encoders.pkl")) 
    print("Training completed!") 
    
    def evaluate_model(args): 
        """Evaluates the model.""" 

        print("=== MODEL EVALUATION ===") 
        
        # Load dataset 
        dataset = ChessEndgameDataset( csv_path=args.csv_path, max_samples=args.max_samples, normalize=args.normalize, add_turn_channel=args.add_turn_channel ) 
        
        # Create model 
        model, loss_fn = create_model( num_type_classes=len(dataset.type_encoder.classes_), num_wdl_classes=3, input_channels=12 if args.add_turn_channel else 13, device=args.device ) 
        
        # Load checkpoint 
        checkpoint = torch.load(args.model_path, map_location=args.device) 
        model.load_state_dict(checkpoint['model_state_dict']) 
        
        # Create evaluator 
        evaluator = ChessEndgameEvaluator(model, device=args.device) 
        
        # Create test DataLoader 
        _, _, test_loader, _ = create_data_loaders( csv_path=args.csv_path, batch_size=args.batch_size, max_samples=args.max_samples, normalize=args.normalize, add_turn_channel=args.add_turn_channel, num_workers=args.num_workers ) 
        
        # Evaluation 
        metrics = evaluator.evaluate( data_loader=test_loader, type_encoder=dataset.type_encoder, save_predictions=args.save_predictions, save_path=args.output_path ) 
        
        # Confusion matrices 
        if args.plot_confusion: evaluator.plot_confusion_matrices( data_loader=test_loader, type_encoder=dataset.type_encoder, save_path=args.confusion_path ) 
        
        # Error analysis 
        if args.analyze_errors: evaluator.analyze_errors( data_loader=test_loader, type_encoder=dataset.type_encoder, top_k=args.top_k_errors ) 
        print("Evaluation completed!") 
        
        def predict_single(args): 
            """Prediction for a single position.""" 
            
            print("=== PREDICTION ===") 

            # Create interface 
            interface = ChessEndgameInterface( model_path=args.model_path, encoders_path=args.encoders_path, device=args.device ) 

            # Prediction 
            type_pred, wdl_pred, probs = interface.predict(args.fen) 
            
            # Show results 
            print(f"FEN: {args.fen}") 
            print(f"Endgame type: {type_pred}") 
            print(f"Outcome: {wdl_pred}") 
            if 'type_top5' in probs: print("\nTop 5 predictions:") 
            for i, pred in enumerate(probs['type_top5'], 1): 
                print(f" {i}. {pred['type']}: {pred['probability']:.3f}") 
                if 'wdl_all' in probs: print("\nWDL probabilities:") 
                for outcome, prob in probs['wdl_all'].items(): print(f" {outcome}: {prob:.3f}") 
                
                def interactive_mode(args): 
                    """Interactive mode.""" 
                print("=== INTERACTIVE MODE ===") # Create interface interface = ChessEndgameInterface( model_path=args.model_path, encoders_path=args.encoders_path, device=args.device ) # Run interactive mode interface.interactive_mode() def main(): """Main function.""" parser = argparse.ArgumentParser(description='Chess Endgame Classifier') subparsers = parser.add_subparsers(dest='command', help='Available commands') # Train command train_parser = subparsers.add_parser('train', help='Train model') train_parser.add_argument('--csv-path', required=True, help='Path to CSV file') train_parser.add_argument('--save-dir', default='checkpoints', help='Directory for saving') train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size') train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs') train_parser.add_argument('--max-samples', type=int, help='Maximum number of samples') train_parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)') train_parser.add_argument('--num-workers', type=int, default=4, help='Number of worker processes') train_parser.add_argument('--train-ratio', type=float, default=0.7, help='Percentage for training') train_parser.add_argument('--val-ratio', type=float, default=0.15, help='Percentage for validation') train_parser.add_argument('--test-ratio', type=float, default=0.15, help='Percentage for testing') train_parser.add_argument('--normalize', action='store_true', help='Normalize positions') train_parser.add_argument('--add-turn-channel', action='store_true', help='Add turn channel') train_parser.add_argument('--save-every', type=int, default=10, help='Save model every N epochs') train_parser.add_argument('--early-stopping-patience', type=int, default=10, help='Patience for early stopping') # Evaluate command eval_parser = subparsers.add_parser('evaluate', help='Evaluate model') eval_parser.add_argument('--model-path', required=True, help='Path to model') eval_parser.add_argument('--csv-path', required=True, help='Path to CSV file') eval_parser.add_argument('--batch-size', type=int, default=32, help='Batch size') eval_parser.add_argument('--max-samples', type=int, help='Maximum number of samples') eval_parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)') eval_parser.add_argument('--num-workers', type=int, default=4, help='Number of worker processes') eval_parser.add_argument('--normalize', action='store_true', help='Normalize positions') eval_parser.add_argument('--add-turn-channel', action='store_true', help='Add turn channel') eval_parser.add_argument('--save-predictions', action='store_true', help='Save predictions') eval_parser.add_argument('--output-path', default='predictions.csv', help='Output file for predictions') eval_parser.add_argument('--plot-confusion', action='store_true', help='Show confusion matrices') eval_parser.add_argument('--confusion-path', default='confusion_matrices.png', help='Path for confusion matrices') eval_parser.add_argument('--analyze-errors', action='store_true', help='Analyze errors') eval_parser.add_argument('--top-k-errors', type=int, default=5, help='Number of top errors to show') # Predict command predict_parser = subparsers.add_parser('predict', help='Prediction for a single position') predict_parser.add_argument('--model-path', required=True, help='Path to model') predict_parser.add_argument('--encoders-path', required=True, help='Path to encoders') predict_parser.add_argument('--fen', required=True, help='FEN position') predict_parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)') # Interactive command interactive_parser = subparsers.add_parser('interactive', help='Interactive mode') interactive_parser.add_argument('--model-path', required=True, help='Path to model') interactive_parser.add_argument('--encoders-path', required=True, help='Path to encoders') interactive_parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)') args = parser.parse_args() if args.command == 'train': train_model(args) elif args.command == 'evaluate': evaluate_model(args) elif args.command == 'predict': predict_single(args) elif args.command == 'interactive': interactive_mode(args) else: parser.print_help() if __name__ == "__main__": main()