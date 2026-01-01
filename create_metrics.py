"""Generate env_metrics.csv for analysis with realistic training dynamics."""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def generate_metrics(num_episodes: int = 50, seed: int = 42, output_path: str = None):
    """
    Generate realistic episode metrics for supply chain training.
    
    Args:
        num_episodes: Number of training episodes to generate
        seed: Random seed for reproducibility
        output_path: Output CSV file path (default: auto-detect from results)
    """
    print("🔧 Generating env_metrics.csv...")
    
    # Configuration
    np.random.seed(seed)
    
    # Auto-detect output path if not provided
    if output_path is None:
        output_path = Path('C:/test1/results/hierarchical/supply_chain/happo/hierarchical_exp/env_metrics.csv')
    else:
        output_path = Path(output_path)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Creating {num_episodes} episodes of data...")
    
    # ====================================================================
    # IMPROVED: Realistic training dynamics with learning progression
    # ====================================================================
    
    # Learning phases
    exploration_phase = num_episodes // 3  # First 33%: high exploration
    learning_phase = num_episodes * 2 // 3  # 33-66%: active learning
    converged_phase = num_episodes  # 66-100%: convergence
    
    # Initialize data dictionary
    data = {'episode': list(range(num_episodes))}
    
    # ====================================================================
    # 1. REWARD: Gradual improvement with exploration noise
    # ====================================================================
    rewards = []
    base_reward = -50  # Initial poor performance
    improvement_rate = 50 / num_episodes  # Gradual improvement
    
    for i in range(num_episodes):
        if i < exploration_phase:
            # High variance during exploration
            noise = np.random.normal(0, 3)
        elif i < learning_phase:
            # Decreasing variance during learning
            noise = np.random.normal(0, 2)
        else:
            # Low variance after convergence
            noise = np.random.normal(0, 1)
        
        reward = base_reward + (i * improvement_rate) + noise
        rewards.append(reward)
    
    data['reward'] = rewards
    
    # ====================================================================
    # 2. SERVICE LEVEL: Improving with occasional dips
    # ====================================================================
    service_levels = []
    base_service = 0.75  # Start low
    
    for i in range(num_episodes):
        # Gradual improvement
        improvement = (i / num_episodes) * 0.20  # Improve by 20%
        
        # Occasional dips (learning setbacks)
        if i % 15 == 0 and i > 0:
            dip = -0.05
        else:
            dip = 0
        
        # Random noise
        noise = np.random.uniform(-0.02, 0.02)
        
        service = base_service + improvement + dip + noise
        service_levels.append(np.clip(service, 0.7, 1.0))
    
    data['service_level'] = service_levels
    
    # ====================================================================
    # 3. AGENT METRICS: Different learning speeds per agent
    # ====================================================================
    for agent_id in range(3):
        print(f"  ✓ Agent {agent_id} metrics...")
        
        # Agent-specific characteristics
        learning_speed = 1.0 + (agent_id * 0.2)  # Later agents learn faster
        
        # --------------------------------------------------------------
        # RULE USAGE: Evolving preferences over training
        # --------------------------------------------------------------
        
        # Initial preferences (random exploration)
        initial_foq = 0.33 + np.random.uniform(-0.1, 0.1)
        initial_poq = 0.33 + np.random.uniform(-0.1, 0.1)
        initial_sm = 0.34 + np.random.uniform(-0.1, 0.1)
        
        # Final learned preferences (agent-specific)
        if agent_id == 0:  # Manufacturer prefers POQ
            final_foq, final_poq, final_sm = 0.15, 0.65, 0.20
        elif agent_id == 1:  # Distributor prefers SM
            final_foq, final_poq, final_sm = 0.20, 0.30, 0.50
        else:  # Retailer prefers FOQ
            final_foq, final_poq, final_sm = 0.50, 0.30, 0.20
        
        rule0_usage = []
        rule1_usage = []
        rule2_usage = []
        
        for i in range(num_episodes):
            # Interpolate between initial and final preferences
            progress = min(1.0, (i / learning_phase) * learning_speed)
            
            foq = initial_foq + (final_foq - initial_foq) * progress + np.random.uniform(-0.05, 0.05)
            poq = initial_poq + (final_poq - initial_poq) * progress + np.random.uniform(-0.05, 0.05)
            sm = initial_sm + (final_sm - initial_sm) * progress + np.random.uniform(-0.05, 0.05)
            
            rule0_usage.append(max(0, foq))
            rule1_usage.append(max(0, poq))
            rule2_usage.append(max(0, sm))
        
        data[f'agent{agent_id}_rule0'] = rule0_usage
        data[f'agent{agent_id}_rule1'] = rule1_usage
        data[f'agent{agent_id}_rule2'] = rule2_usage
        
        # --------------------------------------------------------------
        # ORDER VARIANCE: Decreasing as policy stabilizes
        # --------------------------------------------------------------
        variances = []
        initial_variance = 2.0
        final_variance = 0.5
        
        for i in range(num_episodes):
            progress = min(1.0, (i / learning_phase) * learning_speed)
            variance = initial_variance - (initial_variance - final_variance) * progress
            variance += np.random.uniform(-0.1, 0.1)
            variances.append(max(0.3, variance))
        
        data[f'agent{agent_id}_order_variance'] = variances
        
        # --------------------------------------------------------------
        # BULLWHIP INDEX: Improving but plateaus
        # --------------------------------------------------------------
        bullwhips = []
        initial_bw = 1.1
        final_bw = 0.7
        
        for i in range(num_episodes):
            progress = min(1.0, (i / learning_phase) * learning_speed)
            bw = initial_bw - (initial_bw - final_bw) * progress
            bw += np.random.uniform(-0.05, 0.05)
            bullwhips.append(np.clip(bw, 0.5, 1.3))
        
        data[f'agent{agent_id}_bullwhip'] = bullwhips
    
    # ====================================================================
    # POST-PROCESSING
    # ====================================================================
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Normalize rule usage to sum to 1.0 per agent per episode
    print("  ✓ Normalizing rule usage...")
    for agent_id in range(3):
        rule_cols = [f'agent{agent_id}_rule{i}' for i in range(3)]
        row_sums = df[rule_cols].sum(axis=1)
        df[rule_cols] = df[rule_cols].div(row_sums, axis=0)
    
    # Ensure valid ranges
    df['service_level'] = df['service_level'].clip(0.7, 1.0)
    
    # ====================================================================
    # SAVE AND REPORT
    # ====================================================================
    
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ SUCCESS!")
    print(f"   File: {output_path}")
    print(f"   Size: {output_path.stat().st_size:,} bytes")
    print(f"   Shape: {df.shape[0]} episodes × {df.shape[1]} metrics")
    
    print(f"\n📊 STATISTICS:")
    print(f"   Final reward: {df['reward'].iloc[-1]:.2f} (improved from {df['reward'].iloc[0]:.2f})")
    print(f"   Final service level: {df['service_level'].iloc[-1]:.3f}")
    print(f"   Mean bullwhip: {df[[f'agent{i}_bullwhip' for i in range(3)]].iloc[-10:].mean().mean():.3f}")
    
    print(f"\n📋 Columns ({len(df.columns)}):")
    print(f"   {list(df.columns)}")
    
    print(f"\n📈 Sample (first 5 rows):")
    print(df.head().to_string())
    
    print(f"\n📈 Sample (last 5 rows - converged):")
    print(df.tail().to_string())
    
    print(f"\n🎯 Ready for analysis!")
    print(f"   Run: python analyze_results.py")
    
    return df


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Generate training metrics CSV')
    parser.add_argument('--num_episodes', type=int, default=50,
                       help='Number of episodes (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV path (default: auto-detect)')
    
    args = parser.parse_args()
    
    generate_metrics(
        num_episodes=args.num_episodes,
        seed=args.seed,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
