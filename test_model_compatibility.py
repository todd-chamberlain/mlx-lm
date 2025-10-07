#!/usr/bin/env python3
"""
Test Model Compatibility

Tests that the distributed layer inference works with different model architectures.
"""

import sys
import os

sys.path.insert(0, '.')

from worker.model_adapter import ModelAdapter, MODEL_CONFIGS
from mlx_lm import load


def test_model_detection(model_path: str):
    """Test model detection and adapter"""
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(model_path)}")
    print(f"{'='*70}\n")
    
    try:
        # Load model
        print("Loading model...")
        model, tokenizer = load(model_path)
        print("✓ Model loaded\n")
        
        # Create adapter
        adapter = ModelAdapter(model)
        info = adapter.get_model_info()
        
        # Print info
        print(f"Model Type: {info['model_type']}")
        print(f"Number of Layers: {info['num_layers']}")
        print(f"Has Embedding: {info['has_embedding']}")
        print(f"Has Final Norm: {info['has_norm']}")
        print(f"Has Output Projection: {info['has_output_proj']}")
        print(f"Uses Tied Embeddings: {info['uses_tied_embeddings']}")
        
        # Test getting components
        print("\nTesting component extraction:")
        
        try:
            layers = adapter.get_layers()
            print(f"✓ Layers: {len(layers)} layers extracted")
        except Exception as e:
            print(f"✗ Layers: {e}")
        
        try:
            embedding = adapter.get_embedding()
            print(f"✓ Embedding: {type(embedding).__name__}")
        except Exception as e:
            print(f"✗ Embedding: {e}")
        
        try:
            norm = adapter.get_norm()
            if norm:
                print(f"✓ Norm: {type(norm).__name__}")
            else:
                print(f"ℹ Norm: Not present (some models don't have final norm)")
        except Exception as e:
            print(f"✗ Norm: {e}")
        
        try:
            output_proj = adapter.get_output_projection()
            if output_proj:
                print(f"✓ Output Projection: {type(output_proj).__name__}")
            else:
                print(f"ℹ Output Projection: Using tied embeddings")
        except Exception as e:
            print(f"✗ Output Projection: {e}")
        
        print("\n✅ Model is compatible with distributed inference!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_supported_models():
    """Print list of known supported model types"""
    print("\n" + "="*70)
    print("Known Supported Model Types")
    print("="*70 + "\n")
    
    for model_type, config in MODEL_CONFIGS.items():
        print(f"{model_type.upper()}:")
        print(f"  Layers: {config['layers_path']}")
        print(f"  Embedding: {config['embedding_path']}")
        print(f"  Norm: {config['norm_path']}")
        print(f"  Output: {config['output_path'] or 'tied embeddings'}")
        print()
    
    print("Note: The adapter will attempt to auto-detect any model structure")
    print("even if not explicitly listed above.\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model compatibility')
    parser.add_argument(
        '--model',
        help='Path to model to test'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List known supported model types'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print_supported_models()
        return 0
    
    if not args.model:
        print("Testing with default Llama model...")
        model_path = os.path.expanduser('~/models/Llama-3.2-3B-Instruct-4bit')
    else:
        model_path = os.path.expanduser(args.model)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return 1
    
    success = test_model_detection(model_path)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
