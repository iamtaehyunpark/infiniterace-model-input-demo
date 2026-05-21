import sys
sys.path.insert(0, 'demo')
from loader import load_scene
try:
    nodes = load_scene('gsv_data')
    print(f"Loaded {len(nodes)} nodes successfully.")
except Exception as e:
    print(f"Failed: {e}")
