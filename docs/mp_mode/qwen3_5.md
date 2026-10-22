# Qwen3.5 / Qwen3.6 / Qwen3.8 on Ascend

Hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8, architecture `Qwen3_5ForConditionalGeneration`) interleave Mamba / Gated-DeltaNet layers with full attention. Serve them with MP mode in `lmcache_driven`. `engine_driven` does not support this family.

Flags that match the LMCache recipe are not repeated here. Follow [Qwen3.5 / Qwen3.6 / Qwen3.8 series](https://docs.lmcache.ai/recipes/qwen3_5.html) for:

- `--mamba-cache-mode align` (required; `all` is not supported)
- `--enable-prefix-caching`
- `--separate-object-groups` on `lmcache server`
- `--chunk-size` equal to the attention block size `N` defined below
- `--max-num-batched-tokens` at least `N`, with `2N-1` as the default
- `LMCacheMPConnector` with `kv_role: kv_both`
- CacheBlend and CacheGen stay unsupported
- cached and fresh generations match at score level, not token level

The dense Qwen example in the [repository README](../../README.md) (`--chunk-size 128`, `--no-enable-prefix-caching`) is a different model family. Do not use it here.

## What differs on Ascend

`N` is the attention block size, in tokens, that vLLM selects for the hybrid model during startup. With `--mamba-cache-mode align`, the attention layers and the Mamba layers share that block, and `--chunk-size` has to equal it. `--max-num-batched-tokens` stays `2N-1`.

vLLM picks `N` in `vllm/platforms/interface.py` (`_align_hybrid_block_size`). One attention page has to be at least as large as one Mamba state page, and `N` is then rounded up to the attention kernel's block size. CUDA FlashAttention advertises `MultipleOf(16)` from `vllm/v1/attention/backends/flash_attn.py`, so the LMCache table (784 for Qwen3.8-27B and Qwen3.6-27B, 544 for Qwen3.5-0.8B) is that rule at a 16-token alignment. Ascend attention does not implement a 16-token block. `AscendAttentionBackend.get_supported_kernel_block_sizes` in `vllm_ascend/attention/attention_v1.py` returns `[128]`, which is the alignment vLLM then uses. The note on this patch in `vllm_ascend/patch/__init__.py` says why: a block size of 16 is not supported on Ascend, so the block size is 128. The same page-coverage rule, rounded to 128, is why Qwen3.8-27B moves from 784 to 896. For that model the attention KV heads and the Mamba heads both divide by the tensor-parallel size, so the page ratio is the same at TP1 and TP2. The 784 and 896 differ because of the kernel alignment.

vLLM prints the value it kept:

```text
Setting attention block size to 896 tokens to ensure that attention page size is >= mamba page size.
```

Copy that integer into `--chunk-size`. A later line, `Padding mamba page size by ...%`, pads the Mamba state until its page matches the attention page. It does not change `N`.

On vLLM-Ascend v0.30 this log comes from `vllm/platforms/interface.py`. `vllm_ascend/patch/platform/patch_mamba_config.py` on that release only resizes models that have a sparse indexer (`index_kpool`). Qwen hybrids do not take that branch.

vLLM-Ascend v0.23 uses that file for these models. It replaces `HybridAttentionMambaModelConfig.verify_and_update_config`, sets `kernel_block_size = 128`, and sizes the block so the K cache of one attention block equals the SSM state page. It then logs the same sentence. That match is stricter than the v0.30 rule, so the logged `N` is larger (1536 for Qwen3.8-27B, 1024 for Qwen3.5-4B on the runs below). Use the log from the version you start.

Do not pass `--block-size`. If the flag is omitted, or if it is smaller than this minimum, vLLM overwrites it and logs `N`. An explicit larger value is kept, and `--chunk-size` then has to match that larger value.

`LMCacheMPConnector` checks the server chunk against every KV group at startup. A `--chunk-size` that is not a multiple of `N` aborts with:

```text
LMCache chunk size <chunk> must be a multiple of group <i> tokens_per_block <N>
```

The LMCache recipe sets the chunk equal to `N`. Copying 784 from the CUDA table hits this error when the log says 896.

`N` also follows the model, because the Mamba state shape and the bytes per attention token are model fields. Measured on Ascend 910B3, TP2, with prefix caching and `mamba-cache-mode align`:

| Model | vLLM-Ascend | `N` | `2N-1` |
| --- | --- | ---: | ---: |
| Qwen3.8-27B | v0.30 | 896 | 1791 |
| Qwen3.8-27B | v0.23 | 1536 | 3071 |
| Qwen3.5-4B | v0.30 | 640 | 1279 |
| Qwen3.5-4B | v0.23 | 1024 | 2047 |

Qwen3.6-27B and Qwen3.5-0.8B follow the same code. Read `N` from the log. The CUDA table's 784 and 544 are the 16-token figures for those models.

The LMCache page's `--l1-size-gb 100` is an example, not an Ascend requirement. Size L1 for the workload. On Qwen3.8-27B TP2, LongBench qasper (20 samples, prompt plus 128 generation tokens kept inside 8192) evicted a 16 GB pool during the first pass. A 64 GB pool retained the second pass. Tensor parallel does not split that pool: every rank writes the same L1.

On NPU, an unset transfer mode resolves to `lmcache_driven`, and the server address defaults to `localhost:5555`. Pass `lmcache.mp.host`, `lmcache.mp.port`, and `lmcache.mp.mp_transfer_mode` only when those defaults are wrong.

## Example

Qwen3.8-27B, vLLM-Ascend v0.30, TP2, after the log reports `N = 896`. If the log reports a different `N`, use that value for `--chunk-size` and set `--max-num-batched-tokens` to `2N-1`.

```bash
lmcache server \
    --chunk-size 896 \
    --separate-object-groups \
    --l1-size-gb 64 \
    --eviction-policy LRU
```

```bash
vllm serve /path/to/Qwen3.8-27B \
    --tensor-parallel-size 2 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --max-num-batched-tokens 1791 \
    --kv-transfer-config '{
      "kv_connector": "LMCacheMPConnector",
      "kv_role": "kv_both"
    }'
```
