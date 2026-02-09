import torch
from transformers import AutoTokenizer
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights


class Engine:
    """
    A class to manage the generation engine.
    """
    def __init__(self):
        ########################################
        # Model Configuration Parameters
        ########################################
        self.weight_path = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
        self.head_dim = 64         # Dimensionality of each attention head
        self.num_qo_heads = 32      # Total number of query/output heads
        self.num_kv_heads = 8       # Total number of key/value heads
        self.layers = 16            # Number of transformer layers

        # Load the tokenizer for text processing
        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weight_path,
            local_files_only=True
        )

        # Initialize and load model weights using the helper module
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)

        # Extract all required model weights from the weight_map
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)
        
        self.kv_cache = {}
        self.cache_lens = None
    
    def rms_norm(self, x, weight, eps=1e-5):
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    def _prepare_padded_batch(self, input_ids):
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.to("cuda")
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            lengths = torch.full(
                (ids.shape[0],), ids.shape[1], dtype=torch.long, device=ids.device
            )
            return ids, lengths

        seq_tensors = []
        for seq in input_ids:
            if isinstance(seq, torch.Tensor):
                seq_tensors.append(seq.to("cuda").long())
            else:
                seq_tensors.append(torch.tensor(seq, device="cuda", dtype=torch.long))

        batch_size = len(seq_tensors)
        lengths = torch.tensor([seq.shape[0] for seq in seq_tensors], device="cuda", dtype=torch.long)
        max_len = int(lengths.max().item())

        # Pad with token id 0; masked positions are excluded from attention/logits.
        padded = torch.zeros((batch_size, max_len), dtype=torch.long, device="cuda")
        for i, seq in enumerate(seq_tensors):
            padded[i, : seq.shape[0]] = seq
        return padded, lengths

    def _apply_rope_with_offsets(self, x, offsets):
        # x: [batch_size, seq_len, hidden_dim], offsets: [batch_size]
        for b in range(x.shape[0]):
            apply_rope(x[b], x[b], self.head_dim, int(offsets[b].item()))
        return x

    def run(self, input_ids, prefill = True):
        ########################################
        # Complete this function
        ########################################
        input_ids, input_lens = self._prepare_padded_batch(input_ids)
        batch_size, seq_len = input_ids.shape

        if prefill:
            self.cache_lens = input_lens.clone()
        else:
            if self.cache_lens is None:
                raise RuntimeError("KV cache is empty; call run(..., prefill=True) first.")
            if torch.any(input_lens != 1):
                raise ValueError("Decode step expects one token per sequence.")
            if batch_size != self.cache_lens.shape[0]:
                raise ValueError("Batch size mismatch with cached prompts.")

        x = torch.nn.functional.embedding(input_ids, self.weights["embedding"])

        if prefill:
            query_valid = (
                torch.arange(seq_len, device="cuda")[None, :] < input_lens[:, None]
            )

        for i in range(self.layers):
            residual = x

            # Pre-attention Norm
            x = self.rms_norm(x, self.weights["layernormAttn_weight"][i])

            # QKV
            q = torch.nn.functional.linear(x, self.weights["self_attn_q_proj_weight"][i])
            k = torch.nn.functional.linear(x, self.weights["self_attn_k_proj_weight"][i])
            v = torch.nn.functional.linear(x, self.weights["self_attn_v_proj_weight"][i])

            # RoPE with per-sequence offsets
            if prefill:
                rope_offsets = torch.zeros(batch_size, dtype=torch.long, device="cuda")
            else:
                rope_offsets = self.cache_lens
            q = self._apply_rope_with_offsets(q, rope_offsets)
            k = self._apply_rope_with_offsets(k, rope_offsets)

            # Reshape to heads
            q = q.view(batch_size, seq_len, self.num_qo_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # KV Cache
            if prefill:
                k_cache = k
                v_cache = v
                key_lens = input_lens
            else:
                k_prev, v_prev = self.kv_cache[i]
                k_cache = torch.cat(
                    [
                        k_prev,
                        torch.zeros(
                            (batch_size, self.num_kv_heads, 1, self.head_dim),
                            dtype=k_prev.dtype,
                            device=k_prev.device,
                        ),
                    ],
                    dim=2,
                )
                v_cache = torch.cat(
                    [
                        v_prev,
                        torch.zeros(
                            (batch_size, self.num_kv_heads, 1, self.head_dim),
                            dtype=v_prev.dtype,
                            device=v_prev.device,
                        ),
                    ],
                    dim=2,
                )
                batch_index = torch.arange(batch_size, device="cuda")
                write_pos = self.cache_lens
                k_cache[batch_index, :, write_pos, :] = k[:, :, 0, :]
                v_cache[batch_index, :, write_pos, :] = v[:, :, 0, :]
                key_lens = self.cache_lens + 1

            self.kv_cache[i] = (k_cache, v_cache)

            # GQA Repeat
            k_expanded = k_cache.repeat_interleave(self.num_qo_heads // self.num_kv_heads, dim=1)
            v_expanded = v_cache.repeat_interleave(self.num_qo_heads // self.num_kv_heads, dim=1)

            # Attention
            scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / (self.head_dim ** 0.5)

            total_k = k_expanded.shape[2]
            key_pos = torch.arange(total_k, device="cuda")
            key_valid = key_pos[None, :] < key_lens[:, None]  # [B, K]

            if prefill:
                query_pos = torch.arange(seq_len, device="cuda")
                causal = query_pos[:, None] >= key_pos[None, :]  # [T, K]
                query_valid_expanded = query_valid[:, :, None]  # [B, T, 1]
                allowed = key_valid[:, None, :] & causal[None, :, :] & query_valid_expanded
            else:
                query_abs = rope_offsets[:, None] + torch.arange(seq_len, device="cuda")[None, :]
                causal = query_abs[:, :, None] >= key_pos[None, None, :]
                allowed = key_valid[:, None, :] & causal

            scores = scores.masked_fill(~allowed[:, None, :, :], float("-inf"))
            attn_probs = torch.nn.functional.softmax(scores, dim=-1)
            attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

            attn_out = torch.matmul(attn_probs, v_expanded)
            attn_out = attn_out.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.num_qo_heads * self.head_dim
            )

            # Output Projection
            o = torch.nn.functional.linear(attn_out, self.weights["o_proj_weight"][i])
            x = residual + o

            # FFN Block
            residual = x
            x = self.rms_norm(x, self.weights["layernormFFN_weight"][i])

            gate = torch.nn.functional.linear(x, self.weights["gate_proj_weight"][i])
            up = torch.nn.functional.linear(x, self.weights["up_proj_weight"][i])
            down_input = torch.nn.functional.silu(gate) * up
            down = torch.nn.functional.linear(down_input, self.weights["down_proj_weight"][i])
            x = residual + down

            # Keep padded query rows inert across layers in prefill.
            if prefill:
                x = x * query_valid.unsqueeze(-1).to(x.dtype)

        if not prefill:
            self.cache_lens = self.cache_lens + 1

        # Final Norm + logits from each sequence's last valid token
        x = self.rms_norm(x, self.weights["model_layernorm_weight"])
        last_indices = input_lens - 1 if prefill else torch.zeros_like(input_lens)
        last_hidden = x[torch.arange(batch_size, device="cuda"), last_indices]
        logits = torch.nn.functional.linear(last_hidden, self.weights["lm_head_weight"])
        next_token = torch.argmax(logits, dim=-1)
        return next_token.cpu()
    
    def generate_batched(self, input_string, rounds=20):
        input_ids_list = []
        for input_string in input_string:
            input_ids = self.tokenizer(input_string, return_tensors="pt").input_ids[0]
            input_ids_list.append(input_ids)
            
        output_ids_list = input_ids_list  
        new_token = self.run(input_ids_list)
        for i in range(len(input_ids_list)):
            output_ids_list[i] = torch.cat((output_ids_list[i], new_token[i:i+1]), dim=0)

        for round in range(rounds - 1):
            print(f"Round {round}")
            input_ids_list = []
            for output_ids in output_ids_list:
                input_ids_list.append(output_ids[-1:])
            new_token = self.run(input_ids_list, prefill=False)
            
            for i in range(len(input_ids_list)):
                output_ids_list[i] = torch.cat((output_ids_list[i], new_token[i:i+1]), dim=0)
        output_text_list = []
        for output_ids in output_ids_list:
            output_text_list.append(self.tokenizer.decode(output_ids, skip_special_tokens=True))
        return output_text_list

########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    input_string_list = [input_string for _ in range(10)]
    another_input_string = "The University of Washington is located in"
    for _ in range(10):
        input_string_list.append(another_input_string)
    engine = Engine()
    output_text = engine.generate_batched(input_string_list, rounds=20)
    print("Generated Text:", output_text)
