#!/usr/bin/env python3
"""
Test script za brzo treniranje sa vizuelnim progresom
"""

import sys
import os
sys.path.append('src')

print("=== CHESS ENDGAME TRAINING TEST ===")
print("Starting training with progress visualization...")

try:
    print("1. Importing modules...")
    from dataset import create_data_loaders, ChessEndgameDataset
    from model import create_model
    from train import ChessEndgameTrainer
    print("   ✓ All modules imported successfully")
    
    print("2. Creating dataset...")
    dataset = ChessEndgameDataset(
        csv_path='data/generated_data.csv',
        max_samples=500,  # Mali broj za brzo testiranje
        normalize=True,
        add_turn_channel=True
    )
    print(f"   ✓ Dataset created with {len(dataset)} samples")
    print(f"   ✓ Number of endgame types: {len(dataset.type_encoder.classes_)}")
    
    print("3. Creating DataLoaders...")
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        csv_path='data/generated_data.csv',
        batch_size=16,
        max_samples=500,
        num_workers=0,  # Za Windows
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    print(f"   ✓ Train batches: {len(train_loader)}")
    print(f"   ✓ Val batches: {len(val_loader)}")
    print(f"   ✓ Test batches: {len(test_loader)}")
    
    print("4. Creating model...")
    model, loss_fn = create_model(
        num_type_classes=len(dataset.type_encoder.classes_),
        num_wdl_classes=3,
        input_channels=13,  # 12 + 1 turn channel
        device='cpu'
    )
    print("   ✓ Model created successfully")
    
    print("5. Creating trainer...")
    trainer = ChessEndgameTrainer(
        model=model,
        loss_fn=loss_fn,
        device='cpu',
        save_dir='checkpoints'
    )
    print("   ✓ Trainer created successfully")
    
    print("6. Starting training...")
    print("   This will show progress with tqdm bars...")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,  # Mali broj epoha za brzo testiranje
        save_every=1,
        early_stopping_patience=5
    )
    
    print("7. Saving encoders...")
    dataset.save_encoders(os.path.join('checkpoints', "encoders.pkl"))
    
    print("=== TRAINING COMPLETED SUCCESSFULLY! ===")
    print("Check the 'checkpoints' folder for saved models")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
