"""
Evaluation of chess endgame model.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, 
    confusion_matrix, precision_recall_fscore_support
)
from typing import Dict, List, Optional
import pandas as pd

from model import ChessEndgameModel


class ChessEndgameEvaluator:
    """
    Evaluator class for chess endgame model.
    """
    
    def __init__(self, model: ChessEndgameModel, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate(
        self, 
        data_loader: DataLoader,
        type_encoder: Optional[object] = None,
        save_predictions: bool = False,
        save_path: str = "predictions.csv"
    ) -> Dict[str, float]:
        """
        Evaluation of the model on the dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            type_encoder: Encoder for endgame types
            save_predictions: Whether to save predictions
            save_path: Path to save predictions
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        print("Starting evaluation...")
        
        all_type_preds = []
        all_type_labels = []
        all_wdl_preds = []
        all_wdl_labels = []
        all_type_probs = []
        all_wdl_probs = []
        all_fens = []
        
        with torch.no_grad():
            for batch_idx, (tensors, type_labels_batch, wdl_labels_batch) in enumerate(data_loader):
                # Move to device
                tensors = tensors.to(self.device)
                type_labels_batch = type_labels_batch.to(self.device)
                wdl_labels_batch = wdl_labels_batch.to(self.device)
                
                # Forward pass
                type_logits, wdl_logits = self.model(tensors)
                
                # Softmax for probabilities
                type_probs = torch.softmax(type_logits, dim=1)
                wdl_probs = torch.softmax(wdl_logits, dim=1)
                
                # Predictions
                type_preds = type_logits.argmax(dim=1)
                wdl_preds = wdl_logits.argmax(dim=1)
                
                # Save results
                all_type_preds.extend(type_preds.cpu().numpy())
                all_type_labels.extend(type_labels_batch.cpu().numpy())
                all_wdl_preds.extend(wdl_preds.cpu().numpy())
                all_wdl_labels.extend(wdl_labels_batch.cpu().numpy())
                all_type_probs.extend(type_probs.cpu().numpy())
                all_wdl_probs.extend(wdl_probs.cpu().numpy())
                
                # Save FEN strings if available
                if hasattr(data_loader.dataset, 'df'):
                    batch_start = batch_idx * data_loader.batch_size
                    batch_end = min(batch_start + data_loader.batch_size, len(data_loader.dataset))
                    batch_fens = data_loader.dataset.df.iloc[batch_start:batch_end]['fen'].tolist()
                    all_fens.extend(batch_fens)
        
        # Convert to numpy arrays
        all_type_preds = np.array(all_type_preds)
        all_type_labels = np.array(all_type_labels)
        all_wdl_preds = np.array(all_wdl_preds)
        all_wdl_labels = np.array(all_wdl_labels)
        all_type_probs = np.array(all_type_probs)
        all_wdl_probs = np.array(all_wdl_probs)
        
        # calculate metrics
        metrics = self._calculate_metrics(
            all_type_preds, all_type_labels,
            all_wdl_preds, all_wdl_labels
        )
        
        # show results
        self._print_results(metrics, type_encoder)
        
        # save predictions if needed
        if save_predictions:
            self._save_predictions(
                all_fens, all_type_preds, all_type_labels, all_wdl_preds, all_wdl_labels,
                all_type_probs, all_wdl_probs, type_encoder, save_path
            )
        
        return metrics
    
    def _calculate_metrics(
        self,
        type_preds: np.ndarray,
        type_labels: np.ndarray,
        wdl_preds: np.ndarray,
        wdl_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculates evaluation metrics.
        
        Args:
            type_preds: Predictions for endgame type
            type_labels: True labels for endgame type
            wdl_preds: Predictions for WDL

            wdl_labels: True labels for WDL
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Accuracy
        type_acc = accuracy_score(type_labels, type_preds)
        wdl_acc = accuracy_score(wdl_labels, wdl_preds)
        
        # F1 Score
        type_f1 = f1_score(type_labels, type_preds, average='weighted')
        wdl_f1 = f1_score(wdl_labels, wdl_preds, average='weighted')
        
        # Precision, Recall, F1
        type_precision, type_recall, type_f1_macro, _ = precision_recall_fscore_support(
            type_labels, type_preds, average='macro'
        )
        wdl_precision, wdl_recall, wdl_f1_macro, _ = precision_recall_fscore_support(
            wdl_labels, wdl_preds, average='macro'
        )
        
        # Top-k accuracy for type
        type_top3_acc = self._calculate_top_k_accuracy(type_labels, type_preds, k=3)
        type_top5_acc = self._calculate_top_k_accuracy(type_labels, type_preds, k=5)
        
        return {
            'type_accuracy': type_acc,
            'wdl_accuracy': wdl_acc,
            'type_f1_weighted': type_f1,
            'wdl_f1_weighted': wdl_f1,
            'type_f1_macro': type_f1_macro,
            'wdl_f1_macro': wdl_f1_macro,
            'type_precision': type_precision,
            'type_recall': type_recall,
            'wdl_precision': wdl_precision,
            'wdl_recall': wdl_recall,
            'type_top3_accuracy': type_top3_acc,
            'type_top5_accuracy': type_top5_acc
        }
    
    def _calculate_top_k_accuracy(
        self, 
        labels: np.ndarray, 
        preds: np.ndarray, 
        k: int = 3
    ) -> float:
        """
        Calculates top-k accuracy.
        
        Args:
            labels: True labels
            preds: Predictions
            k: Number of top predictions to consider
        
        Returns:
            float: Top-k accuracy
        """
        # Return basic accuracy
        return accuracy_score(labels, preds)
    
    def _print_results(self, metrics: Dict[str, float], type_encoder: Optional[object] = None):
        """
        Prints evaluation results.
        
        Args:
            metrics: Evaluation metrics
            type_encoder: Encoder for endgame type
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"\nENDGAME TYPE:")
        print(f"  Accuracy: {metrics['type_accuracy']:.4f}")
        print(f"  F1 (weighted): {metrics['type_f1_weighted']:.4f}")
        print(f"  F1 (macro): {metrics['type_f1_macro']:.4f}")
        print(f"  Precision: {metrics['type_precision']:.4f}")
        print(f"  Recall: {metrics['type_recall']:.4f}")
        print(f"  Top-3 Accuracy: {metrics['type_top3_accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {metrics['type_top5_accuracy']:.4f}")
        
        print(f"\nOUTCOME (WDL):")
        print(f"  Accuracy: {metrics['wdl_accuracy']:.4f}")
        print(f"  F1 (weighted): {metrics['wdl_f1_weighted']:.4f}")
        print(f"  F1 (macro): {metrics['wdl_f1_macro']:.4f}")
        print(f"  Precision: {metrics['wdl_precision']:.4f}")
        print(f"  Recall: {metrics['wdl_recall']:.4f}")
        
        print("="*50)
    
    def _save_predictions(
        self,
        fens: List[str],
        type_preds: np.ndarray,
        type_labels: np.ndarray,
        wdl_preds: np.ndarray,
        wdl_labels: np.ndarray,
        type_probs: np.ndarray,
        wdl_probs: np.ndarray,
        type_encoder: Optional[object] = None,
        save_path: str = "predictions.csv"
    ):
        """
        Saves predictions to a CSV file.
        
        Args:
            fens: FEN strings
            type_preds: Predictions for endgame type
            type_labels: True labels for endgame type
            wdl_preds: Predictions for WDL
            wdl_labels: True labels for WDL
            type_probs: Probabilities for endgame type
            wdl_probs: Probabilities for WDL
            type_encoder: Encoder for endgame type
            save_path: Path to save the CSV file
        """
        # Create DataFrame
        data = {
            'fen': fens,
            'type_pred': type_preds,
            'type_true': type_labels,
            'wdl_pred': wdl_preds,
            'wdl_true': wdl_labels
        }
        
        # Add probabilities
        for i in range(type_probs.shape[1]):
            data[f'type_prob_{i}'] = type_probs[:, i]
        
        for i in range(wdl_probs.shape[1]):
            data[f'wdl_prob_{i}'] = wdl_probs[:, i]
        
        # Add string predictions if we have an encoder
        if type_encoder is not None:
            data['type_pred_str'] = type_encoder.inverse_transform(type_preds)
            data['type_true_str'] = type_encoder.inverse_transform(type_labels)
        
        # WDL string predictions
        wdl_map = {0: 'Loss', 1: 'Draw', 2: 'Win'}
        data['wdl_pred_str'] = [wdl_map[p] for p in wdl_preds]
        data['wdl_true_str'] = [wdl_map[l] for l in wdl_labels]
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")
    
    def plot_confusion_matrices(
        self,
        data_loader: DataLoader,
        type_encoder: Optional[object] = None,
        save_path: str = "confusion_matrices.png"
    ):
        """
        Shows confusion matrices.
        
        Args:
            data_loader: DataLoader for evaluation
            type_encoder: Encoder for endgame type
            save_path: Path to save the confusion matrices image
        """
        print("Creating confusion matrices...")
        
        all_type_preds = []
        all_type_labels = []
        all_wdl_preds = []
        all_wdl_labels = []
        
        with torch.no_grad():
            for _, (tensors, type_labels_batch, wdl_labels_batch) in enumerate(data_loader):
                tensors = tensors.to(self.device)
                type_labels_batch = type_labels_batch.to(self.device)
                wdl_labels_batch = wdl_labels_batch.to(self.device)
                
                type_logits, wdl_logits = self.model(tensors)
                
                type_preds = type_logits.argmax(dim=1)
                wdl_preds = wdl_logits.argmax(dim=1)
                
                all_type_preds.extend(type_preds.cpu().numpy())
                all_type_labels.extend(type_labels_batch.cpu().numpy())
                all_wdl_preds.extend(wdl_preds.cpu().numpy())
                all_wdl_labels.extend(wdl_labels_batch.cpu().numpy())
        
        # Konvertuj u numpy arrays
        all_type_preds = np.array(all_type_preds)
        all_type_labels = np.array(all_type_labels)
        all_wdl_preds = np.array(all_wdl_preds)
        all_wdl_labels = np.array(all_wdl_labels)
        
        # Kreiraj confusion matrice
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # WDL confusion matrix
        wdl_cm = confusion_matrix(all_wdl_labels, all_wdl_preds)
        sns.heatmap(
            wdl_cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Loss', 'Draw', 'Win'],
            yticklabels=['Loss', 'Draw', 'Win'],
            ax=axes[0]
        )
        axes[0].set_title('WDL Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # Type confusion matrix (samo top 10 klasa)
        if type_encoder is not None:
            type_cm = confusion_matrix(all_type_labels, all_type_preds)
            
            # Uzmi top 10 najčešćih klasa
            unique_labels = np.unique(all_type_labels)
            label_counts = [(label, np.sum(all_type_labels == label)) for label in unique_labels]
            label_counts.sort(key=lambda x: x[1], reverse=True)
            top_labels = [label for label, _ in label_counts[:10]]
            
            # Filtriraj confusion matrix
            mask = np.isin(all_type_labels, top_labels) & np.isin(all_type_preds, top_labels)
            filtered_labels = all_type_labels[mask]
            filtered_preds = all_type_preds[mask]
            
            if len(filtered_labels) > 0:
                type_cm_filtered = confusion_matrix(filtered_labels, filtered_preds, labels=top_labels)
                
                # Konvertuj u string labele
                type_labels_str = [type_encoder.inverse_transform([label])[0] for label in top_labels]
                
                sns.heatmap(
                    type_cm_filtered,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=type_labels_str,
                    yticklabels=type_labels_str,
                    ax=axes[1]
                )
                axes[1].set_title('Type Confusion Matrix (Top 10)')
                axes[1].set_xlabel('Predicted')
                axes[1].set_ylabel('True')
                plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
                plt.setp(axes[1].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrices saved to {save_path}")
    
    def analyze_errors(
        self,
        data_loader: DataLoader,
        type_encoder: Optional[object] = None,
        top_k: int = 5
    ):
        """
        Analyze errors in model predictions.
        
        Args:
            data_loader: DataLoader for evaluation
            type_encoder: Encoder for endgame type
            top_k: Number of most common errors to display
        """
        print("Analyzing errors...")
        
        errors = []
        
        with torch.no_grad():
            for _, (tensors, type_labels_batch, wdl_labels_batch) in enumerate(data_loader):
                tensors = tensors.to(self.device)
                type_labels_batch = type_labels_batch.to(self.device)
                wdl_labels_batch = wdl_labels_batch.to(self.device)
                
                type_logits, wdl_logits = self.model(tensors)
                
                type_preds = type_logits.argmax(dim=1)
                wdl_preds = wdl_logits.argmax(dim=1)
                
                # Find errors
                type_errors = type_preds != type_labels_batch
                wdl_errors = wdl_preds != wdl_labels_batch
                
                # Save error details
                for i in range(len(type_labels_batch)):
                    if type_errors[i] or wdl_errors[i]:
                        error = {
                            'type_pred': type_preds[i].item(),
                            'type_true': type_labels_batch[i].item(),
                            'wdl_pred': wdl_preds[i].item(),
                            'wdl_true': wdl_labels_batch[i].item(),
                            'type_error': type_errors[i].item(),
                            'wdl_error': wdl_errors[i].item()
                        }
                        errors.append(error)
        
        # Analyze error
        if errors:
            print(f"\nTotal errors: {len(errors)}")
            
            type_errors = [e for e in errors if e['type_error']]
            wdl_errors = [e for e in errors if e['wdl_error']]
            
            print(f"Type errors: {len(type_errors)}")
            print(f"WDL errors: {len(wdl_errors)}")
            
            # Most common type errors
            if type_errors and type_encoder is not None:
                print(f"\nMost common type errors:")
                error_pairs = [(e['type_true'], e['type_pred']) for e in type_errors]
                error_counts = {}
                for true_label, pred_label in error_pairs:
                    key = (true_label, pred_label)
                    error_counts[key] = error_counts.get(key, 0) + 1
                
                sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
                for (true_label, pred_label), count in sorted_errors[:top_k]:
                    true_str = type_encoder.inverse_transform([true_label])[0]
                    pred_str = type_encoder.inverse_transform([pred_label])[0]
                    print(f"  {true_str} -> {pred_str}: {count} times")
            
            # Most common WDL errors
            if wdl_errors:
                print(f"\nMost common WDL errors:")
                wdl_error_pairs = [(e['wdl_true'], e['wdl_pred']) for e in wdl_errors]
                wdl_error_counts = {}
                for true_label, pred_label in wdl_error_pairs:
                    key = (true_label, pred_label)
                    wdl_error_counts[key] = wdl_error_counts.get(key, 0) + 1
                
                sorted_wdl_errors = sorted(wdl_error_counts.items(), key=lambda x: x[1], reverse=True)
                wdl_map = {0: 'Loss', 1: 'Draw', 2: 'Win'}
                for (true_label, pred_label), count in sorted_wdl_errors[:top_k]:
                    true_str = wdl_map[true_label]
                    pred_str = wdl_map[pred_label]
                    print(f"  {true_str} -> {pred_str}: {count} times")
