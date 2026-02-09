import sys

import torch
from transformers import AutoTokenizer

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
        self.head_dim = 64  # Dimensionality of each attention head
        self.num_qo_heads = 32  # Total number of query/output heads
        self.num_kv_heads = 8  # Total number of key/value heads
        self.layers = 16  # Number of transformer layers

        # Load the tokenizer for text processing
        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weight_path, local_files_only=True
        )

        # Initialize and load model weights using the helper module
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)

        # Extract all required model weights from the weight_map
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)

        self.kv_cache = {}

    def rms_norm(self, x, weight, eps=1e-5):
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    def run(self, input_ids, prefill=True):
        ########################################
        # Complete this function
        ########################################
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, device="cuda")
        else:
            input_ids = input_ids.to("cuda")

        # Embeddings
        x = torch.nn.functional.embedding(input_ids, self.weights["embedding"])

        if prefill:
            self.kv_cache = [None] * self.layers

        for i in range(self.layers):
            residual = x

            # Pre-attention Norm
            x = self.rms_norm(x, self.weights["layernormAttn_weight"][i])

            # QKV
            q = torch.nn.functional.linear(
                x, self.weights["self_attn_q_proj_weight"][i]
            )
            k = torch.nn.functional.linear(
                x, self.weights["self_attn_k_proj_weight"][i]
            )
            v = torch.nn.functional.linear(
                x, self.weights["self_attn_v_proj_weight"][i]
            )

            # RoPE
            if prefill:
                offset = 0
            else:
                offset = self.kv_cache[i][0].shape[0]

            apply_rope(q, q, self.head_dim, offset)
            apply_rope(k, k, self.head_dim, offset)

            # Reshape to heads
            q = q.view(-1, self.num_qo_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)

            # KV Cache
            if prefill:
                self.kv_cache[i] = (k, v)
            else:
                k_prev, v_prev = self.kv_cache[i]
                k = torch.cat([k_prev, k], dim=0)
                v = torch.cat([v_prev, v], dim=0)
                self.kv_cache[i] = (k, v)

            # GQA Repeat
            k_expanded = k.repeat_interleave(
                self.num_qo_heads // self.num_kv_heads, dim=1
            )
            v_expanded = v.repeat_interleave(
                self.num_qo_heads // self.num_kv_heads, dim=1
            )

            # Attention
            q_t = q.transpose(0, 1)
            k_t = k_expanded.transpose(0, 1).transpose(1, 2)

            scores = torch.matmul(q_t, k_t) / (self.head_dim**0.5)

            if prefill:
                seq_len = x.shape[0]
                mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device="cuda"),
                    diagonal=1,
                )
                scores = scores + mask

            attn_probs = torch.nn.functional.softmax(scores, dim=-1)

            v_t = v_expanded.transpose(0, 1)
            attn_out = torch.matmul(attn_probs.to(v_t.dtype), v_t)

            attn_out = attn_out.transpose(0, 1).contiguous()
            attn_out = attn_out.view(-1, self.num_qo_heads * self.head_dim)

            # Output Projection
            o = torch.nn.functional.linear(attn_out, self.weights["o_proj_weight"][i])
            x = residual + o

            # FFN Block
            residual = x
            x = self.rms_norm(x, self.weights["layernormFFN_weight"][i])

            gate = torch.nn.functional.linear(x, self.weights["gate_proj_weight"][i])
            up = torch.nn.functional.linear(x, self.weights["up_proj_weight"][i])
            down_input = torch.nn.functional.silu(gate) * up
            down = torch.nn.functional.linear(
                down_input, self.weights["down_proj_weight"][i]
            )

            x = residual + down

        # Final Norm
        x = self.rms_norm(x, self.weights["model_layernorm_weight"])

        # Logits
        logits = torch.nn.functional.linear(x[-1:], self.weights["lm_head_weight"])
        next_token = torch.argmax(logits, dim=-1)
        return next_token.item()

    def generate(self, input_string, rounds=20):
        input_ids = self.tokenizer.encode(input_string)

        print("Token IDs:", input_ids)
        output_ids = input_ids.copy()

        new_token = self.run(output_ids)
        output_ids.append(new_token)

        for round in range(rounds - 1):
            print(f"Round {round}")
            new_token = self.run(output_ids[-1:], prefill=False)
            output_ids.append(new_token)

        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text


########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    engine = Engine()
    output_text = engine.generate(input_string, rounds=20)
    print("Generated Text:", output_text)
