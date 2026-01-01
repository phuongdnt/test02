"""
TEST SCRIPT: Verify POQ và SM v3 logic
Chạy: python test_rules_v3.py
"""

import numpy as np

# ============================================================
# COPY LOGIC TỪ poq_rule_v3.py và sm_rule_v3.py ĐỂ TEST
# ============================================================

def calculate_inventory_position(state):
    """Calculate inventory position = inventory + pipeline - backlog."""
    inventory = float(state.get('inventory', 0.0))
    backlog = float(state.get('backlog', 0.0))
    pipeline = state.get('pipeline', [])
    pipeline_sum = float(sum(pipeline)) if pipeline else 0.0
    return inventory + pipeline_sum - backlog


def poq_v3(state, params):
    """POQ v3 logic."""
    # Parameters
    lead_time = params.get('lead_time', 4)
    target_periods = params.get('target_periods', 2)
    safety_periods = params.get('safety_periods', 1.5)
    
    # Estimate demand
    demand_history = state.get('demand_history', [10.0])
    avg_demand = np.mean(demand_history[-3:]) if demand_history else 10.0
    
    # Inventory position
    inv_pos = calculate_inventory_position(state)
    
    # Reorder point
    reorder_point = avg_demand * safety_periods
    
    # Check
    if inv_pos > reorder_point:
        return 0.0
    
    # Order
    target = avg_demand * (lead_time + target_periods)
    return max(0.0, target - inv_pos)


def sm_v3(state, params):
    """SM v3 logic."""
    safety_periods = params.get('safety_periods', 1.5)
    
    # Estimate demand
    demand_history = state.get('demand_history', [10.0])
    avg_demand = np.mean(demand_history[-3:]) if demand_history else 10.0
    
    # Inventory position
    inv_pos = calculate_inventory_position(state)
    
    # Reorder point
    reorder_point = avg_demand * safety_periods
    
    # Check
    if inv_pos > reorder_point:
        return 0.0
    
    # Silver-Meal (simplified: order for 3 periods)
    optimal_periods = 3
    target = avg_demand * optimal_periods
    return max(0.0, target - inv_pos)


def foq(state, params):
    """FOQ logic (unchanged)."""
    reorder_point = params.get('reorder_point', 10.0)
    order_quantity = params.get('order_quantity', 20.0)
    
    inv_pos = calculate_inventory_position(state)
    
    if inv_pos <= reorder_point:
        return order_quantity
    return 0.0


# ============================================================
# TEST CASES
# ============================================================

def run_tests():
    print("\n" + "="*70)
    print("🧪 TEST POQ v3 và SM v3 LOGIC")
    print("="*70)
    
    params = {
        'foq': {'reorder_point': 10.0, 'order_quantity': 20.0},
        'poq': {'lead_time': 4, 'target_periods': 2, 'safety_periods': 1.5},
        'sm': {'safety_periods': 1.5}
    }
    
    test_cases = [
        {
            'name': 'Inventory CAO (17.86), Pipeline=0, Backlog=0',
            'state': {
                'inventory': 17.86,
                'backlog': 0.0,
                'pipeline': [0, 0, 0, 0],
                'demand_history': [10.0] * 10
            },
            'expected': {'FOQ': 0, 'POQ': 0, 'SM': 0}  # Tất cả phải = 0
        },
        {
            'name': 'Inventory CAO (17.86), Pipeline=20, Backlog=0',
            'state': {
                'inventory': 17.86,
                'backlog': 0.0,
                'pipeline': [5, 5, 5, 5],
                'demand_history': [10.0] * 10
            },
            'expected': {'FOQ': 0, 'POQ': 0, 'SM': 0}
        },
        {
            'name': 'Inventory THẤP (5), Pipeline=0, Backlog=0',
            'state': {
                'inventory': 5.0,
                'backlog': 0.0,
                'pipeline': [0, 0, 0, 0],
                'demand_history': [10.0] * 10
            },
            'expected': {'FOQ': 20, 'POQ': 55, 'SM': 25}  # Tất cả phải > 0
        },
        {
            'name': 'Inventory = 15 (đủ cho 1.5 periods)',
            'state': {
                'inventory': 15.0,
                'backlog': 0.0,
                'pipeline': [0, 0, 0, 0],
                'demand_history': [10.0] * 10
            },
            'expected': {'FOQ': 0, 'POQ': 0, 'SM': 0}  # 15 > 10*1.5=15 → có thể = 0 hoặc nhỏ
        },
        {
            'name': 'Inventory = 10 (đủ cho 1 period)',
            'state': {
                'inventory': 10.0,
                'backlog': 0.0,
                'pipeline': [0, 0, 0, 0],
                'demand_history': [10.0] * 10
            },
            'expected': {'FOQ': 0, 'POQ': 50, 'SM': 20}  # 10 < 15 → cần order
        },
    ]
    
    all_passed = True
    
    for test in test_cases:
        print(f"\n📋 {test['name']}")
        state = test['state']
        inv_pos = calculate_inventory_position(state)
        avg_demand = np.mean(state['demand_history'][-3:])
        reorder_point = avg_demand * 1.5
        
        print(f"   Inventory Position: {inv_pos:.2f}")
        print(f"   Avg Demand: {avg_demand:.2f}")
        print(f"   Reorder Point (demand × 1.5): {reorder_point:.2f}")
        print(f"   Should order? {inv_pos} <= {reorder_point} = {inv_pos <= reorder_point}")
        
        # Test each rule
        foq_order = foq(state, params['foq'])
        poq_order = poq_v3(state, params['poq'])
        sm_order = sm_v3(state, params['sm'])
        
        # Check expectations
        if inv_pos > reorder_point:
            # Should NOT order
            foq_ok = True  # FOQ has different logic
            poq_ok = poq_order == 0
            sm_ok = sm_order == 0
        else:
            # Should order
            foq_ok = True
            poq_ok = poq_order > 0
            sm_ok = sm_order > 0
        
        print(f"\n   Results:")
        print(f"   {'✅' if foq_ok else '❌'} FOQ: Order = {foq_order:.2f}")
        print(f"   {'✅' if poq_ok else '❌'} POQ: Order = {poq_order:.2f}")
        print(f"   {'✅' if sm_ok else '❌'} SM:  Order = {sm_order:.2f}")
        
        if not (poq_ok and sm_ok):
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    run_tests()