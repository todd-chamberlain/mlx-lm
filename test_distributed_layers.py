#!/usr/bin/env python3
"""
Test Distributed Layer Inference

Loads a Llama model, splits it across nodes, and tests layer processing.
"""

import asyncio
import sys
import logging
import time
import mlx.core as mx

sys.path.insert(0, '.')

from worker.model_partitioner import ModelPartitioner, PartitionStrategy, create_pipeline_for_llama
from worker.layer_worker import LayerWorker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text:^70}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")


def print_test(name: str, status: str, details: str = ""):
    if status == "PASS":
        symbol = f"{Colors.GREEN}✓{Colors.NC}"
        status_text = f"{Colors.GREEN}PASS{Colors.NC}"
    elif status == "FAIL":
        symbol = f"{Colors.RED}✗{Colors.NC}"
        status_text = f"{Colors.RED}FAIL{Colors.NC}"
    else:
        symbol = f"{Colors.CYAN}ℹ{Colors.NC}"
        status_text = f"{Colors.CYAN}INFO{Colors.NC}"
    
    print(f"  {symbol} {name:.<50} {status_text}")
    if details:
        print(f"     {Colors.CYAN}{details}{Colors.NC}")


async def test_model_partitioner():
    """Test model partitioning"""
    print_header("TEST 1: Model Partitioner")
    
    try:
        partitioner = ModelPartitioner()
        
        # Create 2-node configuration
        nodes = [
            {'node_id': 'node-1', 'address': '10.201.20.10:50052'},
            {'node_id': 'node-2', 'address': '10.201.20.12:50052'}
        ]
        
        # Partition 32 layers (Llama-3.2-3B)
        config = partitioner.partition_model(
            model_name="Llama-3.2-3B-Instruct-4bit",
            total_layers=32,
            nodes=nodes,
            strategy=PartitionStrategy.BALANCED
        )
        
        print_test("Create Pipeline Config", "PASS", f"Pipeline ID: {config.pipeline_id}")
        print_test("Total Layers", "INFO", f"{config.total_layers} layers")
        
        # Check assignments
        for assignment in config.assignments:
            print_test(
                f"Node {assignment.node_id}",
                "INFO",
                f"Layers {assignment.start_layer}-{assignment.end_layer} ({assignment.num_layers} layers)"
            )
        
        # Verify coverage
        total_assigned = sum(a.num_layers for a in config.assignments)
        if total_assigned == config.total_layers:
            print_test("Layer Coverage", "PASS", f"All {config.total_layers} layers assigned")
        else:
            print_test("Layer Coverage", "FAIL", f"Only {total_assigned}/{config.total_layers} assigned")
            return False
        
        return config
        
    except Exception as e:
        print_test("Model Partitioner", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return None


async def test_layer_loading(model_path: str, config):
    """Test loading layers on each node"""
    print_header("TEST 2: Layer Loading")
    
    workers = {}
    
    for assignment in config.assignments:
        try:
            print(f"\n{Colors.YELLOW}Loading layers for {assignment.node_id}...{Colors.NC}")
            
            worker = LayerWorker(
                node_id=assignment.node_id,
                model_path=model_path,
                assignment=assignment
            )
            
            # Load layers
            start_time = time.time()
            success = await worker.load_layers()
            load_time = time.time() - start_time
            
            if success:
                print_test(
                    f"{assignment.node_id} Load",
                    "PASS",
                    f"Loaded in {load_time:.2f}s"
                )
                
                # Get memory usage
                mem_info = worker.get_memory_usage()
                print_test(
                    f"{assignment.node_id} Memory",
                    "INFO",
                    f"{mem_info['memory_gb']:.2f}GB, {mem_info['num_layers']} layers"
                )
                
                if mem_info.get('has_embedding'):
                    print_test(f"{assignment.node_id} Embedding", "INFO", "Has embedding layer")
                
                if mem_info.get('has_norm'):
                    print_test(f"{assignment.node_id} Norm", "INFO", "Has final norm (last node)")
                
                if mem_info.get('model_type'):
                    print_test(f"{assignment.node_id} Model Type", "INFO", mem_info['model_type'])
                
                workers[assignment.node_id] = worker
            else:
                print_test(f"{assignment.node_id} Load", "FAIL", "Failed to load layers")
                return None
                
        except Exception as e:
            print_test(f"{assignment.node_id} Load", "FAIL", str(e))
            import traceback
            traceback.print_exc()
            return None
    
    return workers


async def test_forward_pass(workers: dict, config):
    """Test forward pass through distributed layers"""
    print_header("TEST 3: Forward Pass Through Pipeline")
    
    try:
        # Get first and last workers
        first_assignment = config.assignments[0]
        last_assignment = config.assignments[-1]
        
        first_worker = workers[first_assignment.node_id]
        last_worker = workers[last_assignment.node_id]
        
        # Create dummy input tokens (add batch dimension)
        input_ids = mx.array([[1, 2, 3, 4, 5]])  # Shape: (batch=1, seq_len=5)
        print_test("Input Tokens", "INFO", f"Shape: {input_ids.shape}")
        
        # Step 1: Embed tokens (first node)
        print(f"\n{Colors.YELLOW}Step 1: Embedding (Node 1){Colors.NC}")
        hidden_states = first_worker.embed_tokens(input_ids)
        print_test("Embedding", "PASS", f"Shape: {hidden_states.shape}")
        
        # Step 2: Process through each node's layers
        for i, assignment in enumerate(config.assignments):
            worker = workers[assignment.node_id]
            print(f"\n{Colors.YELLOW}Step {i+2}: Processing Layers {assignment.start_layer}-{assignment.end_layer} ({assignment.node_id}){Colors.NC}")
            
            start_time = time.time()
            hidden_states, _ = worker.forward(hidden_states)
            process_time = (time.time() - start_time) * 1000
            
            print_test(
                f"Layer Processing",
                "PASS",
                f"Shape: {hidden_states.shape}, Time: {process_time:.2f}ms"
            )
        
        # Step 3: Compute logits (last node)
        print(f"\n{Colors.YELLOW}Step {len(config.assignments)+2}: Computing Logits (Last Node){Colors.NC}")
        logits = last_worker.compute_logits(hidden_states)
        print_test("Logits", "PASS", f"Shape: {logits.shape}")
        
        # Get top token
        next_token = mx.argmax(logits[0, -1], axis=-1)  # [batch, seq, vocab] -> last token
        print_test("Next Token", "INFO", f"Token ID: {int(next_token.item())}")
        
        return True
        
    except Exception as e:
        print_test("Forward Pass", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_generation(workers: dict, config, prompt: str = "Hello"):
    """Test token generation through pipeline"""
    print_header("TEST 4: Token Generation")
    
    try:
        first_worker = workers[config.assignments[0].node_id]
        last_worker = workers[config.assignments[-1].node_id]
        
        # Tokenize prompt
        print_test("Prompt", "INFO", f'"{prompt}"')
        tokens = first_worker.tokenizer.encode(prompt)
        print_test("Tokenized", "INFO", f"{len(tokens)} tokens")
        
        # Generate a few tokens
        max_tokens = 10
        generated_tokens = []
        
        # Add batch dimension
        input_ids = mx.array([tokens])
        
        for i in range(max_tokens):
            # Embed
            hidden_states = first_worker.embed_tokens(input_ids)
            
            # Process through pipeline
            for assignment in config.assignments:
                worker = workers[assignment.node_id]
                hidden_states, _ = worker.forward(hidden_states)
            
            # Get logits and next token
            logits = last_worker.compute_logits(hidden_states)
            next_token = int(mx.argmax(logits[0, -1], axis=-1).item())  # [batch, seq, vocab] -> get last token
            
            generated_tokens.append(next_token)
            
            # Append to input for next iteration
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)
            
            # Decode and print
            decoded = first_worker.tokenizer.decode([next_token])
            print(f"    Token {i+1}: {next_token} → '{decoded}'")
        
        # Decode full output
        full_output = first_worker.tokenizer.decode(tokens + generated_tokens)
        print_test("Generated Text", "PASS", f'"{full_output}"')
        
        return True
        
    except Exception as e:
        print_test("Token Generation", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test distributed layer inference')
    parser.add_argument(
        '--model-path',
        default='~/models/Llama-3.2-3B-Instruct-4bit',
        help='Path to model'
    )
    parser.add_argument(
        '--prompt',
        default='Hello',
        help='Prompt for generation test'
    )
    
    args = parser.parse_args()
    
    # Expand path
    import os
    model_path = os.path.expanduser(args.model_path)
    
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     Distributed Layer Inference Test                              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.NC}")
    
    print(f"Model: {model_path}")
    print(f"Prompt: {args.prompt}")
    
    # Test 1: Model Partitioner
    config = await test_model_partitioner()
    if not config:
        print(f"\n{Colors.RED}❌ Model partitioner test failed{Colors.NC}")
        return 1
    
    # Test 2: Layer Loading
    workers = await test_layer_loading(model_path, config)
    if not workers:
        print(f"\n{Colors.RED}❌ Layer loading test failed{Colors.NC}")
        return 1
    
    # Test 3: Forward Pass
    if not await test_forward_pass(workers, config):
        print(f"\n{Colors.RED}❌ Forward pass test failed{Colors.NC}")
        return 1
    
    # Test 4: Token Generation
    if not await test_generation(workers, config, args.prompt):
        print(f"\n{Colors.RED}❌ Token generation test failed{Colors.NC}")
        return 1
    
    # Summary
    print_header("TEST SUMMARY")
    print(f"{Colors.GREEN}✅ All tests passed!{Colors.NC}\n")
    print(f"{Colors.CYAN}Results:{Colors.NC}")
    print(f"  • Model successfully partitioned across {len(config.assignments)} nodes")
    print(f"  • Each node loaded its assigned layers")
    print(f"  • Forward pass works through the pipeline")
    print(f"  • Token generation works end-to-end")
    print()
    print(f"{Colors.CYAN}Next Steps:{Colors.NC}")
    print(f"  1. Implement activation transfer over gRPC")
    print(f"  2. Deploy to actual distributed nodes")
    print(f"  3. Add streaming support")
    print(f"  4. Optimize with micro-batching")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
