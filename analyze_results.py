"""
Comprehensive results analysis and visualization.
Run after training completes to analyze results and generate plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultsAnalyzer:
    """Analyze and visualize training results."""
    
    def __init__(self, results_dir: str):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Path to results directory containing seed_results.txt and env_metrics.csv
        """
        self.results_dir = Path(results_dir)
        self.seed_results_file = self.results_dir / 'seed_results.txt'
        self.metrics_file = self.results_dir / 'env_metrics.csv'
        
        # Load data
        self.seed_results = self._load_seed_results()
        self.metrics = self._load_metrics()
        
    def _load_seed_results(self) -> pd.DataFrame:
        """Load final seed results from text file."""
        results = []
        if not self.seed_results_file.exists():
            print(f"⚠️  Seed results file not found: {self.seed_results_file}")
            return pd.DataFrame()
        
        with open(self.seed_results_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        seed = int(parts[0])
                        reward = float(parts[1])
                        bullwhip = [float(x) for x in parts[2:]] if len(parts) > 2 else []
                        
                        results.append({
                            'seed': seed,
                            'reward': reward,
                            'bullwhip_mean': np.mean(bullwhip) if bullwhip else 0,
                            'bullwhip_std': np.std(bullwhip) if bullwhip else 0,
                            'bullwhip_agent0': bullwhip[0] if len(bullwhip) > 0 else 0,
                            'bullwhip_agent1': bullwhip[1] if len(bullwhip) > 1 else 0,
                            'bullwhip_agent2': bullwhip[2] if len(bullwhip) > 2 else 0,
                        })
        
        return pd.DataFrame(results)
    
    def _load_metrics(self) -> pd.DataFrame:
        """Load episode-by-episode metrics from CSV."""
        if not self.metrics_file.exists():
            print(f"⚠️  Metrics file not found: {self.metrics_file}")
            return pd.DataFrame()
        
        return pd.read_csv(self.metrics_file)
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive text summary report."""
        report = []
        report.append("=" * 70)
        report.append("TRAINING RESULTS SUMMARY")
        report.append("=" * 70)
        report.append("")
        
        if self.seed_results.empty:
            report.append("⚠️  No training results found yet.")
            report.append("")
            report.append("To generate results:")
            report.append("  1. Run: python create_metrics.py")
            report.append("  2. Then: python analyze_results.py")
            report.append("")
            report.append("=" * 70)
            return "\n".join(report)
        
        # Overall statistics
        report.append("📊 OVERALL PERFORMANCE:")
        report.append(f"  Number of seeds: {len(self.seed_results)}")
        report.append(f"  Mean reward: {self.seed_results['reward'].mean():.4f} ± {self.seed_results['reward'].std():.4f}")
        report.append(f"  Best reward: {self.seed_results['reward'].max():.4f} (seed {self.seed_results.loc[self.seed_results['reward'].idxmax(), 'seed']:.0f})")
        report.append(f"  Worst reward: {self.seed_results['reward'].min():.4f} (seed {self.seed_results.loc[self.seed_results['reward'].idxmin(), 'seed']:.0f})")
        
        # Improvement calculation
        if not self.metrics.empty and 'reward' in self.metrics.columns:
            initial_reward = self.metrics['reward'].iloc[:10].mean()
            final_reward = self.metrics['reward'].iloc[-10:].mean()
            improvement = ((final_reward - initial_reward) / abs(initial_reward)) * 100
            report.append(f"  Improvement: {improvement:+.1f}% (from {initial_reward:.2f} to {final_reward:.2f})")
        
        report.append("")
        
        # Bullwhip statistics
        report.append("📈 BULLWHIP EFFECT:")
        report.append(f"  Mean bullwhip: {self.seed_results['bullwhip_mean'].mean():.4f} ± {self.seed_results['bullwhip_mean'].std():.4f}")
        
        # Per-agent bullwhip
        for agent_id in range(3):
            col = f'bullwhip_agent{agent_id}'
            if col in self.seed_results.columns:
                mean_bw = self.seed_results[col].mean()
                report.append(f"    Agent {agent_id}: {mean_bw:.4f}")
        
        report.append("")
        
        # Per-seed breakdown
        report.append("🔢 PER-SEED RESULTS:")
        for _, row in self.seed_results.iterrows():
            report.append(f"  Seed {row['seed']:.0f}:")
            report.append(f"    Reward: {row['reward']:.4f}")
            report.append(f"    Bullwhip: {row['bullwhip_mean']:.4f} (σ={row['bullwhip_std']:.4f})")
            report.append("")
        
        # Episode metrics summary
        if not self.metrics.empty:
            report.append("📉 EPISODE METRICS:")
            report.append(f"  Total episodes: {len(self.metrics)}")
            
            # Service level
            if 'service_level' in self.metrics.columns:
                initial_sl = self.metrics['service_level'].iloc[:10].mean()
                final_sl = self.metrics['service_level'].iloc[-10:].mean()
                report.append(f"  Service level:")
                report.append(f"    Initial: {initial_sl:.4f}")
                report.append(f"    Final: {final_sl:.4f}")
                report.append(f"    Mean: {self.metrics['service_level'].mean():.4f} ± {self.metrics['service_level'].std():.4f}")
            
            # Order variance trends
            variance_cols = [f'agent{i}_order_variance' for i in range(3)]
            if all(col in self.metrics.columns for col in variance_cols):
                report.append(f"  Order variance (final 10 episodes):")
                for agent_id in range(3):
                    col = f'agent{agent_id}_order_variance'
                    final_var = self.metrics[col].iloc[-10:].mean()
                    report.append(f"    Agent {agent_id}: {final_var:.4f}")
            
            report.append("")
            
            # Rule usage analysis
            rule_cols = [col for col in self.metrics.columns if 'rule' in col]
            if rule_cols:
                report.append("🎯 RULE USAGE SUMMARY (Last 100 Episodes):")
                last_episodes = self.metrics.tail(100)
                
                for agent_id in range(3):
                    foq_col = f'agent{agent_id}_rule0'
                    poq_col = f'agent{agent_id}_rule1'
                    sm_col = f'agent{agent_id}_rule2'
                    
                    if all(col in last_episodes.columns for col in [foq_col, poq_col, sm_col]):
                        foq_mean = last_episodes[foq_col].mean()
                        poq_mean = last_episodes[poq_col].mean()
                        sm_mean = last_episodes[sm_col].mean()
                        
                        report.append(f"  Agent {agent_id}:")
                        report.append(f"    FOQ: {foq_mean*100:5.1f}%")
                        report.append(f"    POQ: {poq_mean*100:5.1f}%")
                        report.append(f"    SM:  {sm_mean*100:5.1f}%")
                        
                        # Dominant rule
                        rules = {'FOQ': foq_mean, 'POQ': poq_mean, 'SM': sm_mean}
                        dominant = max(rules, key=rules.get)
                        report.append(f"    → Dominant: {dominant} ({rules[dominant]*100:.1f}%)")
                report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """
        Plot comprehensive learning curves.
        
        Args:
            save_path: Path to save figure (shows if None)
        """
        if self.metrics.empty:
            print("⚠️  No metrics to plot. Run create_metrics.py first!")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Helper function for smoothing
        def smooth(data, window=10):
            """Apply moving average smoothing."""
            return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean()
        
        # ============================================================
        # Row 1: Reward and Service Level
        # ============================================================
        
        # Reward over time
        if 'reward' in self.metrics.columns:
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(self.metrics['episode'], self.metrics['reward'], alpha=0.3, label='Raw')
            ax1.plot(self.metrics['episode'], smooth(self.metrics['reward']), linewidth=2, label='Smoothed')
            ax1.set_title('Reward Over Training', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Service level
        if 'service_level' in self.metrics.columns:
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(self.metrics['episode'], self.metrics['service_level'], alpha=0.3)
            ax2.plot(self.metrics['episode'], smooth(self.metrics['service_level']), linewidth=2)
            ax2.set_title('Service Level Over Training', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Service Level')
            ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target (0.9)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Convergence indicator (reward variance)
        if 'reward' in self.metrics.columns:
            ax3 = fig.add_subplot(gs[0, 2])
            window = 20
            rolling_std = pd.Series(self.metrics['reward']).rolling(window=window).std()
            ax3.plot(self.metrics['episode'], rolling_std, linewidth=2, color='orange')
            ax3.set_title(f'Reward Stability (Rolling Std, window={window})', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Standard Deviation')
            ax3.grid(True, alpha=0.3)
        
        # ============================================================
        # Row 2: Rule Usage per Agent
        # ============================================================
        
        for agent_id in range(3):
            ax = fig.add_subplot(gs[1, agent_id])
            
            foq_col = f'agent{agent_id}_rule0'
            poq_col = f'agent{agent_id}_rule1'
            sm_col = f'agent{agent_id}_rule2'
            
            if all(col in self.metrics.columns for col in [foq_col, poq_col, sm_col]):
                ax.plot(self.metrics['episode'], smooth(self.metrics[foq_col]), 
                       label='FOQ', linewidth=2, alpha=0.8)
                ax.plot(self.metrics['episode'], smooth(self.metrics[poq_col]), 
                       label='POQ', linewidth=2, alpha=0.8)
                ax.plot(self.metrics['episode'], smooth(self.metrics[sm_col]), 
                       label='SM', linewidth=2, alpha=0.8)
                
                ax.set_title(f'Agent {agent_id} Rule Usage', fontsize=12, fontweight='bold')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Usage Proportion')
                ax.set_ylim([0, 1])
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
        
        # ============================================================
        # Row 3: Order Variance and Bullwhip
        # ============================================================
        
        # Order variance
        ax7 = fig.add_subplot(gs[2, 0:2])
        for agent_id in range(3):
            col = f'agent{agent_id}_order_variance'
            if col in self.metrics.columns:
                ax7.plot(self.metrics['episode'], smooth(self.metrics[col]), 
                        label=f'Agent {agent_id}', linewidth=2, alpha=0.7)
        ax7.set_title('Order Variance Over Training', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Variance')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Bullwhip index
        ax8 = fig.add_subplot(gs[2, 2])
        for agent_id in range(3):
            col = f'agent{agent_id}_bullwhip'
            if col in self.metrics.columns:
                ax8.plot(self.metrics['episode'], smooth(self.metrics[col]), 
                        label=f'Agent {agent_id}', linewidth=2, alpha=0.7)
        ax8.set_title('Bullwhip Index Over Training', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Episode')
        ax8.set_ylabel('Bullwhip Index')
        ax8.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Neutral')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        plt.suptitle('Training Dynamics Analysis', fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Learning curves saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_rule_distribution(self, save_path: Optional[str] = None):
        """
        Plot final rule distribution per agent.
        
        Args:
            save_path: Path to save figure (shows if None)
        """
        if self.metrics.empty:
            print("⚠️  No metrics to plot. Run create_metrics.py first!")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Analyze last 100 episodes for converged behavior
        last_episodes = self.metrics.tail(min(100, len(self.metrics)))
        
        for agent_id in range(3):
            foq_col = f'agent{agent_id}_rule0'
            poq_col = f'agent{agent_id}_rule1'
            sm_col = f'agent{agent_id}_rule2'
            
            if all(col in last_episodes.columns for col in [foq_col, poq_col, sm_col]):
                foq_mean = last_episodes[foq_col].mean()
                poq_mean = last_episodes[poq_col].mean()
                sm_mean = last_episodes[sm_col].mean()
                
                rules = ['FOQ', 'POQ', 'SM']
                values = [foq_mean * 100, poq_mean * 100, sm_mean * 100]
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                bars = axes[agent_id].bar(rules, values, color=colors, edgecolor='black', linewidth=1.5)
                axes[agent_id].set_title(f'Agent {agent_id} Rule Usage\n(Last {len(last_episodes)} Episodes)', 
                                        fontsize=12, fontweight='bold')
                axes[agent_id].set_ylabel('Usage (%)', fontsize=11)
                axes[agent_id].set_ylim([0, 100])
                axes[agent_id].grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[agent_id].text(bar.get_x() + bar.get_width()/2., height + 2,
                                       f'{value:.1f}%',
                                       ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.suptitle('Final Rule Distribution by Agent', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Rule distribution saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_summary_json(self, output_path: str):
        """
        Export summary statistics as JSON.
        
        Args:
            output_path: Path to save JSON file
        """
        if self.seed_results.empty:
            summary = {
                'status': 'no_data',
                'message': 'No training results found. Run create_metrics.py first.'
            }
        else:
            summary = {
                'status': 'success',
                'seeds': self.seed_results.to_dict('records'),
                'statistics': {
                    'num_seeds': len(self.seed_results),
                    'mean_reward': float(self.seed_results['reward'].mean()),
                    'std_reward': float(self.seed_results['reward'].std()),
                    'best_reward': float(self.seed_results['reward'].max()),
                    'worst_reward': float(self.seed_results['reward'].min()),
                    'mean_bullwhip': float(self.seed_results['bullwhip_mean'].mean()),
                    'std_bullwhip': float(self.seed_results['bullwhip_mean'].std()),
                }
            }
            
            # Add episode metrics if available
            if not self.metrics.empty:
                summary['episode_stats'] = {
                    'num_episodes': len(self.metrics),
                }
                
                if 'service_level' in self.metrics.columns:
                    summary['episode_stats']['mean_service_level'] = float(self.metrics['service_level'].mean())
                    summary['episode_stats']['final_service_level'] = float(self.metrics['service_level'].iloc[-10:].mean())
                
                if 'reward' in self.metrics.columns:
                    summary['episode_stats']['initial_reward'] = float(self.metrics['reward'].iloc[:10].mean())
                    summary['episode_stats']['final_reward'] = float(self.metrics['reward'].iloc[-10:].mean())
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ JSON summary saved to: {output_path}")
    
    def generate_latex_table(self) -> str:
        """
        Generate LaTeX table for paper/report.
        
        Returns:
            LaTeX table string
        """
        if self.seed_results.empty:
            return "% No data available\n"
        
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Experimental Results Summary}")
        latex.append("\\label{tab:results}")
        latex.append("\\begin{tabular}{cccc}")
        latex.append("\\toprule")
        latex.append("Seed & Reward & Bullwhip & Service Level \\\\")
        latex.append("\\midrule")
        
        for _, row in self.seed_results.iterrows():
            latex.append(f"{int(row['seed'])} & {row['reward']:.4f} & {row['bullwhip_mean']:.3f} & - \\\\")
        
        latex.append("\\midrule")
        latex.append(f"\\textbf{{Mean}} & \\textbf{{{self.seed_results['reward'].mean():.4f}}} $\\pm$ {self.seed_results['reward'].std():.4f} & "
                    f"\\textbf{{{self.seed_results['bullwhip_mean'].mean():.3f}}} $\\pm$ {self.seed_results['bullwhip_mean'].std():.3f} & - \\\\")
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)


def main():
    """Main analysis workflow."""
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--results_dir', type=str, 
                       default='C:/test1/results/hierarchical/supply_chain/happo/hierarchical_exp',
                       help='Path to results directory')
    parser.add_argument('--output_dir', type=str, default='analysis',
                       help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    # Create analyzer
    print(f"\n📂 Loading results from: {args.results_dir}")
    analyzer = ResultsAnalyzer(args.results_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n🔍 Analyzing results...")
    
    # Generate text report
    report = analyzer.generate_summary_report()
    print(report)
    
    # Save report
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ Report saved to: {report_path}")
    
    # Generate plots (only if data exists)
    if not analyzer.seed_results.empty or not analyzer.metrics.empty:
        print("\n📊 Generating plots...")
        analyzer.plot_learning_curves(save_path=output_dir / 'learning_curves.png')
        analyzer.plot_rule_distribution(save_path=output_dir / 'rule_distribution.png')
    
    # Export JSON
    analyzer.export_summary_json(output_dir / 'summary.json')
    
    # Generate LaTeX table
    latex_table = analyzer.generate_latex_table()
    latex_path = output_dir / 'results_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✅ LaTeX table saved to: {latex_path}")
    
    print("\n✅ Analysis complete!")
    print(f"📂 All outputs saved to: {output_dir.absolute()}")
    
    if analyzer.seed_results.empty:
        print("\n⚠️  No training data found.")
        print("📝 To generate results:")
        print("   1. python create_metrics.py")
        print("   2. python analyze_results.py")


if __name__ == "__main__":
    main()
