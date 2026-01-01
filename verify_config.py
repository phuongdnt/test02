# Test file: tests/test_config.py
from config import get_config, print_config

parser = get_config()
args = parser.parse_args(['--use_hierarchical', 'true'])

print("✅ Config loaded successfully")
print(f"   - use_hierarchical: {args.use_hierarchical}")
print(f"   - discovery_steps: {args.discovery_steps}")
print(f"   - foq_reorder_point: {args.foq_reorder_point}")

print_config(args)
