#!/usr/bin/env python
"""Main training script for hierarchical MARL supply chain system."""

import sys
import os
import numpy as np
from pathlib import Path
import torch
from config import get_config
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from runners.separated.runner import CRunner as Runner


def make_train_env(all_args):
    """Create training environment(s)."""
    return SubprocVecEnv(all_args)


def make_eval_env(all_args):
    """Create evaluation environment(s)."""
    return DummyVecEnv(all_args)


def parse_args(args, parser):
    """Parse command line arguments."""
    all_args = parser.parse_known_args(args)[0]
    return all_args


def print_config_summary(all_args):
    """Print configuration summary."""
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Environment: {getattr(all_args, 'env_name', 'N/A')}")
    print(f"Algorithm: {getattr(all_args, 'algorithm_name', 'happo')}")
    print(f"Experiment: {getattr(all_args, 'experiment_name', 'default')}")
    print(f"Scenario: {getattr(all_args, 'scenario_name', 'default')}")
    print(f"Agents: {getattr(all_args, 'num_agents', 3)}")
    print(f"Total steps: {getattr(all_args, 'num_env_steps', 0):,}")
    print(f"Episode length: {getattr(all_args, 'episode_length', 200)}")
    print(f"Seed(s): {getattr(all_args, 'seed', 1)}")
    
    # Hierarchical-specific
    if getattr(all_args, 'use_hierarchical', False):
        print(f"\n🎯 HIERARCHICAL SYSTEM:")
        print(f"  Discovery steps: {getattr(all_args, 'discovery_steps', 20)}")
        print(f"  Analysis steps: {getattr(all_args, 'analysis_steps', 1)}")
        print(f"  Cooldown period: {getattr(all_args, 'cooldown_period', 15)}")
        print(f"  Switching threshold: {getattr(all_args, 'switching_threshold', -100.0)}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Parse arguments
    parser = get_config()
    all_args = parse_args(sys.argv[1:], parser)
    
    # Handle seed
    if isinstance(all_args.seed, int):
        seeds = [all_args.seed]
    else:
        seeds = all_args.seed
    
    # Print configuration
    print_config_summary(all_args)
    
    # Set device
    if all_args.cuda and torch.cuda.is_available():
        print("✅ Using GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("✅ Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    # Run for each seed
    for seed_idx, seed in enumerate(seeds):
        print("\n" + "="*70)
        print(f"TRAINING SEED {seed_idx + 1}/{len(seeds)}: {seed}")
        print("="*70 + "\n")
        
        # Create directory structure
        base_path = Path(os.path.dirname(os.path.abspath(__file__)))
        env_name = getattr(all_args, 'env_name', 'hierarchical')
        scenario_name = getattr(all_args, 'scenario_name', 'supply_chain')
        algorithm_name = getattr(all_args, 'algorithm_name', 'happo')
        experiment_name = getattr(all_args, 'experiment_name', 'hierarchical_exp')
        
        # ✅ FIXED: Parent directory for seed_results.txt
        results_parent = base_path.parent / "results" / env_name / scenario_name / algorithm_name / experiment_name
        
        # Create parent directory
        if not results_parent.exists():
            os.makedirs(str(results_parent))
        
        # seed_results.txt goes in PARENT folder (not in run_seed_X)
        seed_res_record_file = results_parent / "seed_results.txt"
        
        # Individual run directory for logs and models
        curr_run = f'run_seed_{seed + 1}'
        run_dir = results_parent / curr_run
        
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        
        # Create seed_results.txt with header if doesn't exist
        if not seed_res_record_file.exists():
            with open(seed_res_record_file, 'w') as f:
                f.write("# seed reward bullwhip_indices\n")
        
        # Set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # Update args with current seed
        all_args.seed = seed
        
        # Create environments
        print("Creating environments...")
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None
        num_agents = getattr(all_args, 'num_agents', envs.num_agent)
        
        print(f"✅ Training environments created")
        if eval_envs:
            print(f"✅ Evaluation environments created")
        
        # Create runner config
        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": run_dir
        }
        
        # Create runner and train
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")
        
        try:
            runner = Runner(config)
            reward, bw = runner.run()
            
            # Save results to seed_results.txt
            with open(seed_res_record_file, 'a') as f:
                f.write(f"{seed} {reward:.4f}")
                if bw:
                    for fluc in bw:
                        f.write(f" {fluc:.4f}")
                f.write('\n')
            
            print("\n" + "="*70)
            print(f"✅ TRAINING COMPLETE - SEED {seed}")
            print("="*70)
            print(f"Final reward: {reward:.4f}")
            if bw:
                print(f"Bullwhip indices: {[f'{b:.4f}' for b in bw]}")
            print("="*70 + "\n")
            
        except Exception as e:
            print("\n" + "="*70)
            print(f"❌ ERROR DURING TRAINING - SEED {seed}")
            print("="*70)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("="*70 + "\n")
        
        finally:
            # Clean up environments
            print("Closing environments...")
            envs.close()
            if all_args.use_eval and eval_envs is not None and eval_envs is not envs:
                eval_envs.close()
            print("✅ Environments closed\n")
    
    print("\n" + "="*70)
    print("🎉 ALL SEEDS COMPLETE!")
    print("="*70)
    print(f"Results saved to: {seed_res_record_file}")
    print("="*70 + "\n")
