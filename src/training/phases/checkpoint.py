import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
# print(f"Verified Project Root: {project_root}")  # Should NOT be "/"


def save_checkpoint(epoch: int, algo, config):
    """Save algorithm checkpoint"""
    if algo is not None:
        checkpoint_path = f"{config['checkpointdir']}/epoch{epoch}_marl"
        algo.save_checkpoint(checkpoint_path)
        print(f"MARL checkpoint saved to {checkpoint_path}")
        return checkpoint_path
        # return None
    
def restore_checkpoint(checkpoint_path: str, algo):
    """Restore algorithm from checkpoint"""
    if algo is not None:
        algo.restore(checkpoint_path)
        print(f"MARL checkpoint restored from {checkpoint_path}")