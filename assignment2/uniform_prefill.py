import torch
from transformers import AutoTokenizer
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights, rotate_half


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
    
    def _apply_rope_batched(self, x, head_dim, offset=0):
        # x: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape
        num_heads = hidden_dim // head_dim
        device = x.device
        dtype = x.dtype

        # Create positions: [seq_len]
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=dtype)
        
        base = 500000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim))
        
        # Expand dimensions for broadcasting
        inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1)
        position_ids_expanded = positions[None, None, :].float()
        
        with torch.autocast(device_type=device.type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            
        # Reshape cos/sin: [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        # Reshape x: [batch_size, num_heads, seq_len, head_dim]
        x_reshaped = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Apply RoPE
        x_rotated = x_reshaped * cos + rotate_half(x_reshaped) * sin
        
        # Reshape back to [batch_size, seq_len, hidden_dim]
        return x_rotated.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim).to(dtype=dtype)

    def run(self, input_ids, prefill = True):
        ########################################
        # Complete this function
        ########################################
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, device='cuda')
        else:
            input_ids = input_ids.to('cuda')
        
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = torch.nn.functional.embedding(input_ids, self.weights["embedding"])
        
        if prefill:
            self.kv_cache = [None] * self.layers
            
        for i in range(self.layers):
            residual = x
            
            # Pre-attention Norm
            x = self.rms_norm(x, self.weights["layernormAttn_weight"][i])
            
            # QKV
            q = torch.nn.functional.linear(x, self.weights["self_attn_q_proj_weight"][i])
            k = torch.nn.functional.linear(x, self.weights["self_attn_k_proj_weight"][i])
            v = torch.nn.functional.linear(x, self.weights["self_attn_v_proj_weight"][i])
            
            # RoPE
            if prefill:
                offset = 0
            else:
                offset = self.kv_cache[i][0].shape[2]
            
            q = self._apply_rope_batched(q, self.head_dim, offset)
            k = self._apply_rope_batched(k, self.head_dim, offset)
            
            # Reshape to heads and transpose
            # q: [batch_size, num_heads, seq_len, head_dim]
            q = q.view(batch_size, seq_len, self.num_qo_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            # KV Cache
            if prefill:
                self.kv_cache[i] = (k, v)
            else:
                k_prev, v_prev = self.kv_cache[i]
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
                self.kv_cache[i] = (k, v)
            
            # GQA Repeat
            k_expanded = k.repeat_interleave(self.num_qo_heads // self.num_kv_heads, dim=1)
            v_expanded = v.repeat_interleave(self.num_qo_heads // self.num_kv_heads, dim=1)
            
            # Attention
            # scores: [batch_size, num_qo_heads, seq_len, total_seq_len]
            scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if prefill:
                mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device='cuda'), diagonal=1)
                scores = scores + mask
            
            attn_probs = torch.nn.functional.softmax(scores, dim=-1)
            
            # attn_out: [batch_size, num_qo_heads, seq_len, head_dim]
            attn_out = torch.matmul(attn_probs, v_expanded)
            
            # Reshape back
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_qo_heads * self.head_dim)
            
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
            
        # Final Norm
        x = self.rms_norm(x, self.weights["model_layernorm_weight"])
        
        # Logits
        logits = torch.nn.functional.linear(x[:, -1:, :], self.weights["lm_head_weight"])
        next_token = torch.argmax(logits, dim=-1)
        return next_token
    
    def generate_batched(self, input_string, rounds=20):
        input_ids_list = self.tokenizer(input_string, return_tensors="pt", padding=False).input_ids
        print("Input String:", input_string)

        print("Token IDs:", input_ids_list)
        output_ids_list = input_ids_list  

        new_token = self.run(output_ids_list)
        print("New Token Shape:", new_token.shape)
        output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        for round in range(rounds - 1):
            print(f"Round {round}")
            new_token = self.run(output_ids_list[:, -1:], prefill=False)
            output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        output_text = self.tokenizer.batch_decode(output_ids_list, skip_special_tokens=True)
        return output_text

########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    input_string_list = [input_string for _ in range(10)]
    another_input_string = "Hi, how are you?"
    for _ in range(10):
        input_string_list.append(another_input_string)
    engine = Engine()
    output_text = engine.generate_batched(input_string_list, rounds=20)
    print("Generated Text:", output_text)