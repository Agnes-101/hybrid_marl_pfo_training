import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
print(f"Verified Project Root: {project_root}")  # Should NOT be "/"

from ray.rllib.algorithms.callbacks import DefaultCallbacks
import json

class VizCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """Called at end of each training iteration."""
        it = result["training_iteration"]
        if it % 5 != 0:
            return

        env = algorithm.workers.local_worker().env
        policy = algorithm.get_policy()
        obs, _ = env.reset()
        associations = []

        for _ in range(env.episode_length):
            acts = {}
            for i in range(env.num_ue):
                o = obs[f"ue_{i}"]
                logits, _ = policy.model({"obs": o})
                acts[f"ue_{i}"] = int(logits.argmax(dim=-1).item())

            obs, _, done, _, _ = env.step(acts)
            associations.append([ue.associated_bs for ue in env.ues])
            if done["__all__"]:
                break

        # Dump JSON for Streamlit to pick up
        os.makedirs("/tmp/assoc", exist_ok=True)
        with open(f"/tmp/assoc/assoc_iter{it}.json", "w") as f:
            json.dump({"iteration": it, "associations": associations}, f)
        print(f"[VizCallback] wrote /tmp/assoc/assoc_iter{it}.json")