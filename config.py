"""Configuration for hierarchical MARL supply chain system - COMPLETE VERSION."""

import argparse

def get_config():
    """Get configuration parser with ALL parameters."""
    parser = argparse.ArgumentParser(
        description='Hierarchical MARL for Supply Chain Management',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ========================================
    # ENVIRONMENT
    # ========================================
    parser.add_argument("--env_name", type=str, default='MyEnv')
    parser.add_argument("--scenario_name", type=str, default='Ineventory Management')
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--lead_time", type=int, default=4)
    parser.add_argument("--episode_length", type=int, default=200)
    
    # ========================================
    # ALGORITHM
    # ========================================
    parser.add_argument("--algorithm_name", type=str, default='happo')
    parser.add_argument("--experiment_name", type=str, default='check')
    parser.add_argument("--use_centralized_V", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--use_obs_instead_of_state", type=lambda x: bool(str(x).lower() == 'true'), default=False)
    
    # ========================================
    # LEARNING
    # ========================================
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--critic_lr", type=float, default=5e-4)
    parser.add_argument("--opti_eps", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--use_gae", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--use_proper_time_limits", type=lambda x: bool(str(x).lower() == 'true'), default=False)
    
    parser.add_argument("--num_mini_batch", type=int, default=1)
    parser.add_argument("--ppo_epoch", type=int, default=15)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--use_max_grad_norm", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_loss_coef", type=float, default=1.0)
    parser.add_argument("--use_clipped_value_loss", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--use_huber_loss", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--huber_delta", type=float, default=10.0)
    parser.add_argument("--use_value_active_masks", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--use_policy_active_masks", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    
    # ========================================
    # NETWORK
    # ========================================
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--layer_N", type=int, default=2)
    parser.add_argument("--act_hidden_size", type=int, default=128)
    parser.add_argument("--recurrent_N", type=int, default=1)
    parser.add_argument("--data_chunk_length", type=int, default=10)
    parser.add_argument("--stacked_frames", type=int, default=1)
    
    # Recurrent options
    parser.add_argument("--use_naive_recurrent_policy", type=lambda x: bool(str(x).lower() == 'true'), default=False)
    parser.add_argument("--use_recurrent_policy", type=lambda x: bool(str(x).lower() == 'true'), default=False)
    parser.add_argument("--use_influence_policy", type=lambda x: bool(str(x).lower() == 'true'), default=False)
    parser.add_argument("--use_policy_vhead", type=lambda x: bool(str(x).lower() == 'true'), default=False)
    
    # Normalization
    parser.add_argument("--use_ReLU", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--use_feature_normalization", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--use_orthogonal", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--use_popart", type=lambda x: bool(str(x).lower() == 'true'), default=False)
    parser.add_argument("--use_valuenorm", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--use_single_network", type=lambda x: bool(str(x).lower() == 'true'), default=False)
    parser.add_argument("--gain", type=float, default=0.01)
    
    # ========================================
    # TRAINING
    # ========================================
    parser.add_argument("--num_env_steps", type=int, default=10000000)
    parser.add_argument("--n_rollout_threads", type=int, default=5)
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1)
    parser.add_argument("--use_linear_lr_decay", type=lambda x: bool(str(x).lower() == 'true'), default=False)
    
    # ========================================
    # LOGGING & EVALUATION
    # ========================================
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--use_eval", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--n_warmup_evaluations", type=int, default=5)
    parser.add_argument("--n_no_improvement_thres", type=int, default=20)
    parser.add_argument("--use_render", type=lambda x: bool(str(x).lower() == 'true'), default=False)
    
    # ========================================
    # HIERARCHICAL SYSTEM
    # ========================================
    parser.add_argument("--use_hierarchical", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--discovery_steps", type=int, default=20)
    parser.add_argument("--analysis_steps", type=int, default=1)
    parser.add_argument("--switching_threshold", type=float, default=-100.0)
    parser.add_argument("--cooldown_period", type=int, default=15)
    parser.add_argument("--evaluation_window", type=int, default=10)
    
    # Coordination
    parser.add_argument("--coordination_weight", type=float, default=0.05)
    parser.add_argument("--inventory_balance_weight", type=float, default=0.01)
    parser.add_argument("--order_stability_weight", type=float, default=0.005)
    parser.add_argument("--bullwhip_penalty_weight", type=float, default=0.02)
    
    # ========================================
    # RULE PARAMETERS
    # ========================================
    # FOQ
    parser.add_argument("--foq_reorder_point", type=float, default=10.0)
    parser.add_argument("--foq_order_quantity", type=float, default=20.0)
    
    # POQ
    parser.add_argument("--poq_lead_time", type=int, default=4)
    parser.add_argument("--poq_target_periods", type=int, default=2)
    parser.add_argument("--poq_forecast_window", type=int, default=3)
    
    # SM
    parser.add_argument("--sm_setup_cost", type=float, default=50.0)
    parser.add_argument("--sm_holding_cost", type=float, default=1.0)
    parser.add_argument("--sm_forecast_horizon", type=int, default=10)
    parser.add_argument("--sm_forecast_window", type=int, default=3)
    
    # ========================================
    # PATHS
    # ========================================
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="./results/logs")
    parser.add_argument("--save_dir", type=str, default="./results")
    
    # ========================================
    # MISC
    # ========================================
    parser.add_argument("--seed", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--cuda", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--cuda_deterministic", type=lambda x: bool(str(x).lower() == 'true'), default=True)
    parser.add_argument("--n_training_threads", type=int, default=1)
    
    return parser


if __name__ == "__main__":
    parser = get_config()
    args = parser.parse_args()
    print(f"✅ Config loaded successfully!")
    print(f"Total parameters: {len(vars(args))}")
