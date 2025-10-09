"""
Jednostavan CLI interfejs za testiranje šahovskog endgame modela.
"""

import torch
import argparse
import sys
import os
from typing import Optional, Tuple
import json

# Dodaj src u path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import ChessEndgameModel, ChessEndgameLoss
from dataset import ChessEndgameDataset
from fen_to_tensor import fen_to_tensor, add_turn_channel
import chess


class ChessEndgameInterface:
    """
    CLI interfejs za šahovski endgame model.
    """
    
    def __init__(self, model_path: str, encoders_path: str, device: str = "cpu"):
        """
        Args:
            model_path: Putanja do modela
            encoders_path: Putanja do enkodera
            device: Device za model
        """
        self.device = device
        
        # Učitaj enkodere
        self.encoders = self._load_encoders(encoders_path)
        
        # Kreiraj model
        self.model = self._load_model(model_path)
        
        print(f"Model učitán sa {model_path}")
        print(f"Device: {device}")
        print(f"Broj tipova završnica: {self.encoders['num_type_classes']}")
        print(f"Broj WDL klasa: {self.encoders['num_wdl_classes']}")
    
    def _load_encoders(self, encoders_path: str) -> dict:
        """Učitava enkodere."""
        import pickle
        
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        
        return encoders
    
    def _load_model(self, model_path: str) -> ChessEndgameModel:
        """Učitava model."""
        model = ChessEndgameModel(
            input_channels=12,
            num_type_classes=self.encoders['num_type_classes'],
            num_wdl_classes=self.encoders['num_wdl_classes']
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model.to(self.device)
    
    def predict(self, fen: str) -> Tuple[str, str, dict]:
        """
        Predikcija za FEN poziciju.
        
        Args:
            fen: FEN string pozicije
        
        Returns:
            Tuple[str, str, dict]:
                - type_prediction: Predikcija tipa završnice
                - wdl_prediction: Predikcija ishoda
                - probabilities: Verovatnoće
        """
        try:
            # Validiraj FEN
            board = chess.Board(fen)
            
            # Konvertuj u tensor
            tensor = fen_to_tensor(fen, normalize=True)
            tensor = add_turn_channel(tensor, board.turn)
            
            # Konvertuj u PyTorch tensor
            tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)  # Dodaj batch dim
            tensor = tensor.to(self.device)
            
            # Predikcija
            with torch.no_grad():
                type_logits, wdl_logits = self.model(tensor)
                
                # Softmax za verovatnoće
                type_probs = torch.softmax(type_logits, dim=1)
                wdl_probs = torch.softmax(wdl_logits, dim=1)
                
                # Predikcije
                type_pred = type_logits.argmax(dim=1).item()
                wdl_pred = wdl_logits.argmax(dim=1).item()
                
                # Konvertuj u string
                type_prediction = self.encoders['type_classes'][type_pred]
                wdl_prediction = self._wdl_to_string(wdl_pred)
                
                # Verovatnoće
                probabilities = {
                    'type_probs': type_probs[0].cpu().numpy().tolist(),
                    'wdl_probs': wdl_probs[0].cpu().numpy().tolist(),
                    'type_top5': self._get_top5_predictions(type_probs[0]),
                    'wdl_all': self._get_wdl_probabilities(wdl_probs[0])
                }
                
                return type_prediction, wdl_prediction, probabilities
                
        except Exception as e:
            return f"Greška: {str(e)}", "Greška", {}
    
    def _wdl_to_string(self, wdl: int) -> str:
        """Konvertuje WDL u string."""
        wdl_map = {0: 'Loss (Crni pobeduje)', 1: 'Draw (Remi)', 2: 'Win (Beli pobeduje)'}
        return wdl_map.get(wdl, 'Nepoznato')
    
    def _get_top5_predictions(self, type_probs: torch.Tensor) -> list:
        """Vraća top 5 predikcija za tip završnice."""
        top5_probs, top5_indices = torch.topk(type_probs, 5)
        
        results = []
        for prob, idx in zip(top5_probs, top5_indices):
            type_name = self.encoders['type_classes'][idx.item()]
            results.append({
                'type': type_name,
                'probability': prob.item()
            })
        
        return results
    
    def _get_wdl_probabilities(self, wdl_probs: torch.Tensor) -> dict:
        """Vraća verovatnoće za WDL."""
        return {
            'Loss': wdl_probs[0].item(),
            'Draw': wdl_probs[1].item(),
            'Win': wdl_probs[2].item()
        }
    
    def interactive_mode(self):
        """Interaktivni režim."""
        print("\n" + "="*60)
        print("ŠAHOVSKI ENDGAME KLASIFIKATOR")
        print("="*60)
        print("Unesite FEN poziciju ili 'quit' za izlaz")
        print("Primer: 4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        print("="*60)
        
        while True:
            try:
                fen = input("\nFEN: ").strip()
                
                if fen.lower() in ['quit', 'exit', 'q']:
                    print("Doviđenja!")
                    break
                
                if not fen:
                    continue
                
                # Predikcija
                type_pred, wdl_pred, probs = self.predict(fen)
                
                # Prikaži rezultate
                print(f"\n{'='*40}")
                print(f"TIP ZAVRŠNICE: {type_pred}")
                print(f"ISHOD: {wdl_pred}")
                print(f"{'='*40}")
                
                # Top 5 predikcija za tip
                if 'type_top5' in probs:
                    print("\nTop 5 predikcija za tip:")
                    for i, pred in enumerate(probs['type_top5'], 1):
                        print(f"  {i}. {pred['type']}: {pred['probability']:.3f}")
                
                # WDL verovatnoće
                if 'wdl_all' in probs:
                    print(f"\nWDL verovatnoće:")
                    for outcome, prob in probs['wdl_all'].items():
                        print(f"  {outcome}: {prob:.3f}")
                
                print(f"{'='*40}")
                
            except KeyboardInterrupt:
                print("\nDoviđenja!")
                break
            except Exception as e:
                print(f"Greška: {e}")
    
    def batch_predict(self, fens: list) -> list:
        """
        Batch predikcija za listu FEN pozicija.
        
        Args:
            fens: Lista FEN stringova
        
        Returns:
            list: Lista rezultata
        """
        results = []
        
        for i, fen in enumerate(fens):
            print(f"Procesiranje {i+1}/{len(fens)}: {fen[:50]}...")
            type_pred, wdl_pred, probs = self.predict(fen)
            
            results.append({
                'fen': fen,
                'type_prediction': type_pred,
                'wdl_prediction': wdl_pred,
                'probabilities': probs
            })
        
        return results
    
    def save_predictions(self, results: list, output_path: str):
        """
        Čuva predikcije u JSON fajl.
        
        Args:
            results: Lista rezultata
            output_path: Putanja za čuvanje
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Predikcije sačuvane u {output_path}")


def main():
    """Glavna funkcija."""
    parser = argparse.ArgumentParser(description='Šahovski Endgame Klasifikator')
    parser.add_argument('--model', required=True, help='Putanja do modela')
    parser.add_argument('--encoders', required=True, help='Putanja do enkodera')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--fen', help='FEN pozicija za predikciju')
    parser.add_argument('--batch', help='Fajl sa FEN pozicijama (jedna po liniji)')
    parser.add_argument('--output', help='Output fajl za batch predikcije')
    
    args = parser.parse_args()
    
    # Kreiraj interfejs
    interface = ChessEndgameInterface(
        model_path=args.model,
        encoders_path=args.encoders,
        device=args.device
    )
    
    if args.fen:
        # Predikcija za jednu poziciju
        type_pred, wdl_pred, probs = interface.predict(args.fen)
        
        print(f"FEN: {args.fen}")
        print(f"Tip završnice: {type_pred}")
        print(f"Ishod: {wdl_pred}")
        
        if 'type_top5' in probs:
            print("\nTop 5 predikcija:")
            for i, pred in enumerate(probs['type_top5'], 1):
                print(f"  {i}. {pred['type']}: {pred['probability']:.3f}")
        
        if 'wdl_all' in probs:
            print("\nWDL verovatnoće:")
            for outcome, prob in probs['wdl_all'].items():
                print(f"  {outcome}: {prob:.3f}")
    
    elif args.batch:
        # Batch predikcija
        with open(args.batch, 'r') as f:
            fens = [line.strip() for line in f if line.strip()]
        
        results = interface.batch_predict(fens)
        
        if args.output:
            interface.save_predictions(results, args.output)
        else:
            # Prikaži rezultate
            for result in results:
                print(f"FEN: {result['fen']}")
                print(f"Tip: {result['type_prediction']}")
                print(f"Ishod: {result['wdl_prediction']}")
                print("-" * 40)
    
    else:
        # Interaktivni režim
        interface.interactive_mode()


if __name__ == "__main__":
    main()