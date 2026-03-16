from __future__ import annotations

from sched import scheduler
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import flashinfer
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
#  Project utilities (local module)
# ---------------------------------------------------------------------------
# helper.py must live one directory above this file
sys.path.append(str(Path(__file__).resolve().parent.parent))
from helper import WeightManager, extract_model_weights  # noqa: E402


# ---------------------------------------------------------------------------
#  Low-level data structures: paged KV-cache & per-request view
# ---------------------------------------------------------------------------
class DistKVPool:
    """Global *paged* KV-cache ("HND" = head-page-dim layout)."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        capacity: int,
        page_size: int,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.page_size = page_size

        # Simple free-list allocator ------------------------------------------------
        self._free_pages: set[int] = set(range(capacity))

        # Backing storage tensors: (L, N, H, P, D) where
        #   L = num transformer layers
        #   N = total pages in the pool
        kv_shape = (
            num_layers,
            capacity,
            num_kv_heads,
            page_size,
            head_dim,
        )
        self.k_datas = torch.empty(kv_shape, dtype=torch.float16, device="cuda")
        self.v_datas = torch.empty_like(self.k_datas)

    # Free-list helpers -----------------------------------------------------------
    @property
    def num_free_pages(self) -> int:  # noqa: D401  (short property)
        """Number of unallocated pages left in the pool."""
        return len(self._free_pages)

    def alloc_page(self) -> int:
        """Pop a page index off the free list (O(1))."""
        return self._free_pages.pop()

    def free_page(self, idx: int) -> None:
        """Return *idx* back to the pool."""
        assert idx not in self._free_pages, "double-free detected"
        self._free_pages.add(idx)


class DistKVCache:
    """Light-weight *view* of a request's KV pages (no real storage)."""

    def __init__(self, pool: DistKVPool):
        self._pool = pool
        self._indices: list[int] = []  # page indices owned by this request
        self._seqlen: int = 0          # total tokens stored so far
        self.page_size = pool.page_size

    # Convenience properties -----------------------------------------------------
    @property
    def seqlen(self) -> int:
        return self._seqlen

    @property
    def indices(self) -> list[int]:
        return self._indices

    @property
    def last_page_offset(self) -> int:
        """Number of tokens already present in the *last* page (0-based)."""
        if self._seqlen == 0:
            return 0
        remainder = self._seqlen % self.page_size
        return self.page_size if remainder == 0 else remainder

    # Allocation / release -------------------------------------------------------
    def allocate_tokens(self, num_tokens: int) -> None:
        """Grow the cache so it can hold *num_tokens* additional tokens."""
        assert num_tokens > 0, "must allocate a positive number of tokens"

        # Tokens that still fit into the *current* (possibly partial) page --------
        room_in_last = (
            self.page_size - self.last_page_offset
        ) % self.page_size  # 0 when last page is full

        remaining = max(0, num_tokens - room_in_last)
        pages_needed = (remaining + self.page_size - 1) // self.page_size

        for _ in range(pages_needed):
            self._indices.append(self._pool.alloc_page())

        self._seqlen += num_tokens

    def release(self) -> None:
        """Return all pages back to the global pool (when request finishes)."""
        for idx in self._indices:
            self._pool.free_page(idx)
        self._indices.clear()
        self._seqlen = 0


# ---------------------------------------------------------------------------
#  Helpers to convert a *list* of DistKVCache into FlashInfer ragged metadata
# ---------------------------------------------------------------------------

def build_kv_metadata(kvs: List[DistKVCache]):
    """Return (indptr, indices, last_page_len) - all torch.cuda tensors."""
    kv_indptr: List[int] = [0]
    kv_indices: List[int] = []
    kv_last_page_len: List[int] = []

    for kv in kvs:
        kv_indptr.append(kv_indptr[-1] + len(kv.indices))
        kv_indices.extend(kv.indices)
        kv_last_page_len.append(kv.last_page_offset)

    device = "cuda"
    return (
        torch.tensor(kv_indptr, dtype=torch.int32, device=device),
        torch.tensor(kv_indices, dtype=torch.int32, device=device),
        torch.tensor(kv_last_page_len, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
#  Simple *request* wrapper (prompt + generation buffer)
# ---------------------------------------------------------------------------
class Request:
    def __init__(self, req_id: int, prompt_ids: torch.Tensor, target_len: int):
        self.request_id = req_id
        self.prompt_token_ids = prompt_ids  # (prompt_len,)
        self.output_length = target_len
        # History buffer (prompt + generated tokens will be appended here)
        self.output_token_ids = prompt_ids.clone()
        self.scheduling_pf_tokens = None
        self.last_chunk = False
        self.remaining_prefill_tokens = self.prompt_length

    # Convenience --------------------------------------------------------------
    @property
    def prompt_length(self) -> int:
        return self.prompt_token_ids.size(0)

    @property
    def current_length(self) -> int:
        return self.output_token_ids.size(0)
    
    @property
    def scheduling_length(self) -> int:
        return self.scheduling_pf_tokens.size(0) if self.scheduling_pf_tokens is not None else 0


# ---------------------------------------------------------------------------
#  Generation *engine*
# ---------------------------------------------------------------------------
class Engine:
    """A minimal Llama-3.2-1B engine using FlashInfer for attention."""

    # ---------------------------------------------------------------------
    #  Initialisation
    # ---------------------------------------------------------------------
    def __init__(self) -> None:
        # ---- model hyper-parameters --------------------------------------
        self.weight_path = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
        self.head_dim = 64    
        self.num_qo_heads = 32     
        self.num_kv_heads = 8       
        self.layers = 16            

        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_path)

        # ---- load weights -------------------------------------------------
        wm = WeightManager()
        wm.load_from_safe_tensor(self.weight_path)
        self.weights = extract_model_weights(wm.weight_map, self.layers)

        # ---- global paged KV-cache ---------------------------------------
        self.page_size = 16
        self.max_pages = 30_000  # total pages in the pool (across *all* layers)
        self.pool = DistKVPool(
            num_layers=self.layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            capacity=self.max_pages,
            page_size=self.page_size,
        )

        # Mapping: request-id -> DistKVCache
        self.kv_cache_map: Dict[int, DistKVCache] = {}

        # FlashInfer workspace (single allocation for the whole run)
        workspace_bytes = 128 << 20  # 128 MiB
        self._fi_workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device="cuda"
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._fi_workspace, "HND"
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._fi_workspace, "HND", use_tensor_cores=True)
        self.last_run_breakdown: Dict[str, float] = {}

    # ---------------------------------------------------------------------
    #  One *step* (mixed prefill + decode) over an *arbitrary* request batch
    # ---------------------------------------------------------------------
    def run(
        self,
        requests: List[Request],
        num_decode_req: int = 0,
        enable_timing: bool = False,
    ):
        """Run *one* transformer step for ``requests``.

        Parameters
        ----------
        requests : List[Request]
            Full list of requests to be processed this step.
        num_decode_req : int, default=0
            Number of *decode* requests (the **first** N in *requests*).
            Those will feed only their **last** token; the rest are prefills.
        """
        stage_times: Dict[str, float] = {}
        stage_start: Optional[float] = None

        def tic() -> None:
            nonlocal stage_start
            if enable_timing:
                torch.cuda.synchronize()
                stage_start = time.perf_counter()

        def toc(name: str) -> None:
            nonlocal stage_start
            if enable_timing and stage_start is not None:
                torch.cuda.synchronize()
                stage_times[name] = stage_times.get(name, 0.0) + (
                    (time.perf_counter() - stage_start) * 1000.0
                )
                stage_start = None

        tic()
        with torch.inference_mode():
            # ----------------------------------------------------------------
            # 1) Build ragged *input* tensor and its CSR *indptr*
            # ----------------------------------------------------------------
            pieces: List[torch.Tensor] = []
            indptr: List[int] = [0]

            for idx, req in enumerate(requests):
                if idx < num_decode_req:  # decode - feed only *last* token
                    pieces.append(req.output_token_ids[-1:])
                    indptr.append(indptr[-1] + 1)
                else:                     # prefill
                    pieces.append(req.scheduling_pf_tokens)
                    indptr.append(indptr[-1] + req.scheduling_length)
                    #########
                    # FIXME #
                    #########

            input_tensor = torch.cat(pieces).to("cuda")
            # print(f"batch size {len(input_tensor)}")
            indptr_tensor = torch.tensor(indptr, dtype=torch.int32, device="cuda")
            toc("build_input")

            # ----------------------------------------------------------------
            # 2) Ensure each request has a DistKVCache & collect pre-sizes
            # ----------------------------------------------------------------
            tic()
            seq_lens_before: List[int] = []
            for req in requests:
                cache = self.kv_cache_map.setdefault(req.request_id, DistKVCache(self.pool))
                seq_lens_before.append(cache.seqlen)

            seq_lens_before_t = torch.tensor(seq_lens_before, dtype=torch.int32, device="cuda")
            toc("prepare_kv_views")

            # ----------------------------------------------------------------
            # 3) Reserve pages for the *new* tokens just queued
            # ----------------------------------------------------------------
            tic()
            for idx, req in enumerate(requests):
                cache = self.kv_cache_map[req.request_id]
                #########
                # FIXME #
                #########
                tokens_to_add = req.scheduling_length if idx >= num_decode_req else 1
                cache.allocate_tokens(tokens_to_add)

            seq_lens_after = [self.kv_cache_map[r.request_id].seqlen for r in requests]
            seq_lens_after_t = torch.tensor(seq_lens_after, dtype=torch.int32, device="cuda")
            toc("allocate_kv_pages")

            # Build paged-KV metadata **after** the append -------------------
            tic()
            kv_indptr, kv_indices, kv_last_page_len = build_kv_metadata(
                [self.kv_cache_map[r.request_id] for r in requests]
            )
            toc("build_kv_metadata")

            # ----------------------------------------------------------------
            # 4) Plan FlashInfer execution for this micro-batch
            # ----------------------------------------------------------------
            #########
            # FIXME #
            #########
            tic()
            num_decode_tokens = indptr_tensor[num_decode_req]
            if not len(requests) - num_decode_req == 0:
                prefill_qo_indptr = indptr_tensor[num_decode_req:] - num_decode_tokens
                prefill_kv_indptr = kv_indptr[num_decode_req:] - kv_indptr[num_decode_req]
                
                prefill_kv_indices = kv_indices[kv_indptr[num_decode_req]:]
                
                self.prefill_wrapper.plan(
                    qo_indptr=prefill_qo_indptr,
                    paged_kv_indptr=prefill_kv_indptr,
                    paged_kv_indices=prefill_kv_indices,
                    paged_kv_last_page_len=kv_last_page_len[num_decode_req:],
                    num_qo_heads=self.num_qo_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim_qk=self.head_dim,
                    page_size=self.page_size,
                    causal=True
                )

            if num_decode_req > 0:
                decode_total_pages = kv_indptr[num_decode_req]
                self.decode_wrapper.plan(
                    indptr=kv_indptr[:num_decode_req + 1],
                    indices=kv_indices[:decode_total_pages],
                    last_page_len=kv_last_page_len[:num_decode_req],
                    num_qo_heads=self.num_qo_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    data_type=torch.float16,
                )
            toc("wrapper_plan")

            # ----------------------------------------------------------------
            # 5) Forward pass through all *transformer* layers
            # ----------------------------------------------------------------
            tic()
            hidden = self.weights["embedding"][input_tensor]
            toc("embedding_lookup")

            for layer in range(self.layers):
                # === Self-attention sub-layer ==================================
                tic()
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_attn_in = (hidden / rms).to(torch.float16) * self.weights["layernormAttn_weight"][layer]

                k = (
                    ln_attn_in
                    .matmul(self.weights["self_attn_k_proj_weight"][layer].T)
                    .view(-1, self.num_kv_heads, self.head_dim)
                )
                v = (
                    ln_attn_in
                    .matmul(self.weights["self_attn_v_proj_weight"][layer].T)
                    .view(-1, self.num_kv_heads, self.head_dim)
                )
                q = (
                    ln_attn_in
                    .matmul(self.weights["self_attn_q_proj_weight"][layer].T)
                    .view(-1, self.num_qo_heads, self.head_dim)
                )
                toc("attn_qkv_proj")
                # print("seq_lens_before_t", seq_lens_before_t)

                # ---- Rotary positional embedding ---------------------------
                tic()
                flashinfer.apply_rope_inplace(
                    q,
                    k,
                    indptr_tensor,
                    seq_lens_before_t,
                    rope_theta=500_000.0,
                )
                toc("apply_rope")

                # ---- Map ragged → (batch, pos) for the append --------------
                tic()
                batch_idx, pos = flashinfer.get_batch_indices_positions(
                    indptr_tensor, seq_lens_after_t, q.size(0)
                )

                # ---- Append new tokens to *paged* KV-cache ------------------
                flashinfer.append_paged_kv_cache(
                    append_key=k,  # token-wise keys   (T, Hkv, D)
                    append_value=v,  # token-wise values (T, Hkv, D)
                    batch_indices=batch_idx,  # request-id for each *token*
                    positions=pos,  # absolute position of each token
                    paged_kv_cache=(
                        self.pool.k_datas[layer],  # (L, N, Hkv, P, D) -> k
                        self.pool.v_datas[layer],  # (L, N, Hkv, P, D) -> v
                    ),
                    kv_indices=kv_indices,  # flattened page ids for *all* reqs
                    kv_indptr=kv_indptr,  # row pointer into kv_indices
                    kv_last_page_len=kv_last_page_len,  # tokens used in last page
                    kv_layout="HND",  # (Head, Page, Dim)
                )
                toc("copy_kv_cache")

                # ---- Attention itself --------------------------------------
                attn_out = None
                #########
                # FIXME #
                #########
                tic()
                if not len(requests) - num_decode_req == 0:
                    prefill_q = q[num_decode_tokens:]
                    prefill_att = self.prefill_wrapper.run(
                        q=prefill_q,
                        paged_kv_cache=(self.pool.k_datas[layer], self.pool.v_datas[layer]),
                    )
                
                if not num_decode_req == 0:
                    decode_q = q[:num_decode_tokens]
                    decode_att = self.decode_wrapper.run(
                        q=decode_q,
                        paged_kv_cache=(self.pool.k_datas[layer], self.pool.v_datas[layer]),
                    )
                toc("attention_kernel")

                if num_decode_req == 0:
                    attn_out = prefill_att
                elif len(requests) - num_decode_req == 0:
                    attn_out = decode_att
                else:
                    attn_out = torch.cat([decode_att, prefill_att], dim=0)
                tic()
                attn_out = attn_out.view(attn_out.size(0), -1)  # (T, Hq*D)
                
                # Residual connection
                hidden = attn_out.matmul(self.weights["o_proj_weight"][layer].T) + hidden
                toc("attn_output_proj")

                # === FFN sub-layer ==========================================
                tic()
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_ffn_in = (hidden / rms).to(torch.float16) * self.weights["layernormFFN_weight"][layer]

                up = ln_ffn_in.matmul(self.weights["up_proj_weight"][layer].T)
                gate = ln_ffn_in.matmul(self.weights["gate_proj_weight"][layer].T)
                hidden = (
                    (up * torch.nn.functional.silu(gate))
                    .matmul(self.weights["down_proj_weight"][layer].T)
                    + hidden
                )
                toc("ffn")

            # ----------------------------------------------------------------
            # 6) Final language-model head ----------------------------------
            tic()
            rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
            logits = (
                (hidden / rms).to(torch.float16) * self.weights["model_layernorm_weight"]
            ).matmul(self.weights["lm_head_weight"].T)
            toc("lm_head")

            sample_ids = torch.argmax(logits, dim=-1)

            # Extract *new* token for each request (last token of each row)
            last_token_indices = (indptr_tensor[1:] - 1).long()
            tic()
            output = sample_ids[last_token_indices].cpu()
            toc("output_select")
            if enable_timing:
                stage_times["total"] = sum(stage_times.values())
                self.last_run_breakdown = stage_times
            else:
                self.last_run_breakdown = {}
            return output
