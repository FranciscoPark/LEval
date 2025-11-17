# hooking.py

import torch
from transformers.models.llama.modeling_llama import LlamaAttention

def calculate_head_outliers(attn_weights):
    avg = attn_weights.mean(-1).unsqueeze(-1)
    spike_mask = (attn_weights > 3 * avg).float()
    spike_score = spike_mask.mean(-1)[:, -1]
    return (-spike_score).argsort().tolist()

class LlamaQKCollector:
    """
    Collects Q/K for each forward pass.
    - Stores per-step:
         step -> layer_idx -> {"q": Tensor, "k": Tensor}
    - Safe with FlashAttention: hooks BEFORE FA2, on q_proj/k_proj.
    """
    def __init__(self, max_steps=100000, store_dtype=torch.bfloat16):
        self.max_steps = max_steps
        self.store_dtype = store_dtype

        self.current_step = 0
        self.step_cache = {}      # temporary: layer → q/k
        self.storage = {}         # persistent: step → layer → q/k

        # internal buffers
        self._q_buffer = {}
        self._k_buffer = {}

    def save_step(self):
        """Move current self.step_cache → persistent storage"""
        self.storage[self.current_step] = self.step_cache
        self.step_cache = {}
        self.current_step += 1

    # -----------------------------
    # Hooks on q_proj / k_proj
    # -----------------------------
    def q_proj_hook(self, module, input, output):
        layer_id = module._layer_id
        self._q_buffer[layer_id] = output.detach().to("cpu", dtype=self.store_dtype)

    def k_proj_hook(self, module, input, output):
        layer_id = module._layer_id
        self._k_buffer[layer_id] = output.detach().to("cpu", dtype=self.store_dtype)

    def attach(self, model):
        """Register hooks on each LlamaAttention.q_proj/k_proj"""
        for name, module in model.named_modules():
            if isinstance(module, LlamaAttention):
                layer_idx = int(name.split(".")[2])
                module.q_proj._layer_id = layer_idx
                module.k_proj._layer_id = layer_idx

                module.q_proj.register_forward_hook(self.q_proj_hook)
                module.k_proj.register_forward_hook(self.k_proj_hook)

                print(f"[QKCollector] hooked layer {layer_idx}")

    def finalize_step(self):
        """After a model.generate(), combine Q/K into structured form"""
        for layer_id in sorted(self._q_buffer.keys()):
            self.step_cache[layer_id] = {
                "q": self._q_buffer[layer_id],
                "k": self._k_buffer[layer_id],
            }

        self._q_buffer.clear()
        self._k_buffer.clear()
        self.save_step()

    def export(self, path):
        """Save entire dataset Q/K cache"""
        print(f"[QKCollector] saving Q/K cache to {path}")
        torch.save(self.storage, path)

class LlamaAttentionHook:
    def __init__(self):
        self.records = []  # stores per-layer stats for ONE forward

    def hook(self, module, input, output):
        _, attn_weights = output   # B,H,Q,K
        if attn_weights.size(0) == 1:
            attn_weights = attn_weights.squeeze(0)

        head_rank = calculate_head_outliers(attn_weights.detach().cpu())

        self.records.append({
            "layer": module.layer_idx,
            "head_ranking": head_rank,
        })

    def attach(self, model):
        for name, module in model.named_modules():
            if isinstance(module, LlamaAttention):
                layer_idx = int(name.split(".")[2])
                module.layer_idx = layer_idx
                module.register_forward_hook(self.hook)
                print(f"[Info] Hooked LlamaAttention layer {layer_idx}")

    def get_and_clear(self):
        out = self.records.copy()
        self.records.clear()
        return out