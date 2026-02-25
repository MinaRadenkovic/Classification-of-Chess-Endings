"""
Training pipeline for chess endgame model.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import time
import os
from typing import Dict
import json
from tqdm import tqdm

from model import ChessEndgameModel, ChessEndgameLoss


class ChessEndgameTrainer:
    """
    Trainer class for chess endgame model.
    """
    
    def __init__(
        self,
        model: ChessEndgameModel,
        loss_fn: ChessEndgameLoss,
        device: str = "cpu",
        save_dir: str = "checkpoints"
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = save_dir
        
        # Create directorium for saving
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_type_acc': [],
            'val_type_acc': [],
            'train_wdl_acc': [],
            'val_wdl_acc': [],
            'train_type_f1': [],
            'val_type_f1': [],
            'train_wdl_f1': [],
            'val_wdl_f1': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Trains one epoch.
        
        Args:
            train_loader: DataLoader for training
        
        Returns:
            Dict[str, float]: Metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_type_loss = 0.0
        total_wdl_loss = 0.0
        
        type_preds = []
        type_labels = []
        wdl_preds = []
        wdl_labels = []
        
        pbar = tqdm(train_loader, desc="Training")
        
        for _, (tensors, type_labels_batch, wdl_labels_batch) in enumerate(pbar):
            # Move dat to device
            tensors = tensors.to(self.device)
            type_labels_batch = type_labels_batch.to(self.device)
            wdl_labels_batch = wdl_labels_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            type_logits, wdl_logits = self.model(tensors)
            
            # Loss
            loss, type_loss, wdl_loss = self.loss_fn(
                type_logits, wdl_logits, type_labels_batch, wdl_labels_batch
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Acumulate loss
            total_loss += loss.item()
            total_type_loss += type_loss.item()
            total_wdl_loss += wdl_loss.item()
            
            # Acumulate predictions
            type_preds.extend(type_logits.argmax(dim=1).cpu().numpy())
            type_labels.extend(type_labels_batch.cpu().numpy())
            wdl_preds.extend(wdl_logits.argmax(dim=1).cpu().numpy())
            wdl_labels.extend(wdl_labels_batch.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Type': f'{type_loss.item():.4f}',
                'WDL': f'{wdl_loss.item():.4f}'
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        avg_type_loss = total_type_loss / len(train_loader)
        avg_wdl_loss = total_wdl_loss / len(train_loader)
        
        type_acc = accuracy_score(type_labels, type_preds)
        wdl_acc = accuracy_score(wdl_labels, wdl_preds)
        
        type_f1 = f1_score(type_labels, type_preds, average='weighted')
        wdl_f1 = f1_score(wdl_labels, wdl_preds, average='weighted')
        
        return {
            'loss': avg_loss,
            'type_loss': avg_type_loss,
            'wdl_loss': avg_wdl_loss,
            'type_acc': type_acc,
            'wdl_acc': wdl_acc,
            'type_f1': type_f1,
            'wdl_f1': wdl_f1
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validates one epoch.
        
        Args:
            val_loader: DataLoader for validation
        
        Returns:
            Dict[str, float]: Metrics for the epoch
        """
        self.model.eval()
        
        total_loss = 0.0
        total_type_loss = 0.0
        total_wdl_loss = 0.0
        
        type_preds = []
        type_labels = []
        wdl_preds = []
        wdl_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for _, (tensors, type_labels_batch, wdl_labels_batch) in enumerate(pbar):
                # Move to device
                tensors = tensors.to(self.device)
                type_labels_batch = type_labels_batch.to(self.device)
                wdl_labels_batch = wdl_labels_batch.to(self.device)
                
                # Forward pass
                type_logits, wdl_logits = self.model(tensors)
                
                # Loss
                loss, type_loss, wdl_loss = self.loss_fn(
                    type_logits, wdl_logits, type_labels_batch, wdl_labels_batch
                )
                
                # Acumulate loss
                total_loss += loss.item()
                total_type_loss += type_loss.item()
                total_wdl_loss += wdl_loss.item()
                
                # Acumulate predikcije
                type_preds.extend(type_logits.argmax(dim=1).cpu().numpy())
                type_labels.extend(type_labels_batch.cpu().numpy())
                wdl_preds.extend(wdl_logits.argmax(dim=1).cpu().numpy())
                wdl_labels.extend(wdl_labels_batch.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Type': f'{type_loss.item():.4f}',
                    'WDL': f'{wdl_loss.item():.4f}'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        avg_type_loss = total_type_loss / len(val_loader)
        avg_wdl_loss = total_wdl_loss / len(val_loader)
        
        type_acc = accuracy_score(type_labels, type_preds)
        wdl_acc = accuracy_score(wdl_labels, wdl_preds)
        
        type_f1 = f1_score(type_labels, type_preds, average='weighted')
        wdl_f1 = f1_score(wdl_labels, wdl_preds, average='weighted')
        
        return {
            'loss': avg_loss,
            'type_loss': avg_type_loss,
            'wdl_loss': avg_wdl_loss,
            'type_acc': type_acc,
            'wdl_acc': wdl_acc,
            'type_f1': type_f1,
            'wdl_f1': wdl_f1
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        save_every: int = 10,
        early_stopping_patience: int = 10
    ):
        """
        Main trainig function.
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            epochs: Number of epochs
            save_every: Saves model every N epochs
            early_stopping_patience: Patience for early stopping
        """
        print(f"Starting training on {self.device}...")
        print(f"Number of epochs: {epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        start_time = time.time()
        best_epoch = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            print(f"\n=== EPOCH {epoch+1}/{epochs} ===")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Save in history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_type_acc'].append(train_metrics['type_acc'])
            self.history['val_type_acc'].append(val_metrics['type_acc'])
            self.history['train_wdl_acc'].append(train_metrics['wdl_acc'])
            self.history['val_wdl_acc'].append(val_metrics['wdl_acc'])
            self.history['train_type_f1'].append(train_metrics['type_f1'])
            self.history['val_type_f1'].append(val_metrics['type_f1'])
            self.history['train_wdl_f1'].append(train_metrics['wdl_f1'])
            self.history['val_wdl_f1'].append(val_metrics['wdl_f1'])
            
            # Show metrics
            print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train Type Acc: {train_metrics['type_acc']:.4f} | Val Type Acc: {val_metrics['type_acc']:.4f}")
            print(f"Train WDL Acc: {train_metrics['wdl_acc']:.4f} | Val WDL Acc: {val_metrics['wdl_acc']:.4f}")
            print(f"Train Type F1: {train_metrics['type_f1']:.4f} | Val Type F1: {val_metrics['type_f1']:.4f}")
            print(f"Train WDL F1: {train_metrics['wdl_f1']:.4f} | Val WDL F1: {val_metrics['wdl_f1']:.4f}")
            
            # Check for best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                print(f"New best model! Val Loss: {val_metrics['loss']:.4f}")
            
            # Save model
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1, val_metrics['loss'])
            
            # Early stopping
            if epoch - best_epoch >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch duration: {epoch_time:.2f}s")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model from epoch {best_epoch+1}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed! Total time: {total_time:.2f}s")
        
        # Save final model
        self.save_checkpoint(epochs, val_metrics['loss'], is_final=True)
        
        # Save history
        self.save_history()
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_final: bool = False):
        """
        Saves model checkpoint.
        
        Args:
            epoch: Number of epoch
            val_loss: Validation loss
            is_final: Is it final model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        if is_final:
            filename = f"final_model.pth"
        else:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def save_history(self):
        """
        Save trainig history to JSON.
        """
        filepath = os.path.join(self.save_dir, "training_history.json")
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved: {filepath}")
    
    def plot_history(self):
        """
        Show graphics of treining history.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training History')
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Type Accuracy
        axes[0, 1].plot(self.history['train_type_acc'], label='Train')
        axes[0, 1].plot(self.history['val_type_acc'], label='Val')
        axes[0, 1].set_title('Type Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # WDL Accuracy
        axes[0, 2].plot(self.history['train_wdl_acc'], label='Train')
        axes[0, 2].plot(self.history['val_wdl_acc'], label='Val')
        axes[0, 2].set_title('WDL Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Type F1
        axes[1, 0].plot(self.history['train_type_f1'], label='Train')
        axes[1, 0].plot(self.history['val_type_f1'], label='Val')
        axes[1, 0].set_title('Type F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # WDL F1
        axes[1, 1].plot(self.history['train_wdl_f1'], label='Train')
        axes[1, 1].plot(self.history['val_wdl_f1'], label='Val')
        axes[1, 1].set_title('WDL F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Learning Rate
        axes[1, 2].plot(self.history['train_loss'], label='Train Loss')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.show()
