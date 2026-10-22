# Qwen3.5 / Qwen3.6 / Qwen3.8 on Ascend

Hybrid Qwen models (`Qwen3_5ForConditionalGeneration`) interleave Mamba / Gated-DeltaNet layers with full attention. Serve them with MP mode in `lmcache_driven`. `engine_driven` does not support this family.

Flags that match the LMCache recipe are not repeated here. Follow [Qwen3.5 / Qwen3.6 / Qwen3.8 series](https://docs.lmcache.ai/recipes/qwen3_5.html) for:

- `--mamba-cache-mode align` (required; `all` is not supported)
- `--enable-prefix-caching`
- `--separate-object-groups` on `lmcache server`
- `--chunk-size` equal to vLLM's unified attention block size `N`
- `--max-num-batched-tokens` at least `N`, with `2N-1` as the default
- `LMCacheMPConnector` with `kv_role: kv_both`
- CacheBlend and CacheGen stay unsupported
- cached and fresh generations match at score level, not token level

The dense Qwen example in the [repository README](../../README.md) (`--chunk-size 128`, `--no-enable-prefix-caching`) is a different model family. Do not use it here.

## What differs on Ascend

`N` is not the value in the LMCache table. That table is for one CUDA GPU (`784` for Qwen3.8-27B and Qwen3.6-27B, `544` for Qwen3.5-0.8B). On Ascend, vLLM chooses `N` so the attention page is at least as large as the Mamba page, and it logs:

```text
Setting attention block size to N tokens to ensure that attention page size is >= mamba page size.
```

Set `--chunk-size` to that `N`. Do not pass `--block-size`. A chunk size copied from the CUDA page, or left at the server default, will not match the scheduler block.

`N` depends on the model, the vLLM-Ascend version, and tensor parallel size. Measured on Ascend 910B3, TP2, with prefix caching and `mamba-cache-mode align`:

| Model | vLLM-Ascend | `N` | `2N-1` |
| --- | --- | ---: | ---: |
| Qwen3.8-27B | v0.30 | 896 | 1791 |
| Qwen3.8-27B | v0.23 | 1536 | 3071 |
| Qwen3.5-4B | v0.30 | 640 | 1279 |
| Qwen3.5-4B | v0.23 | 1024 | 2047 |

Qwen3.6-27B and Qwen3.5-0.8B use the same procedure. Read `N` from the log instead of copying a row above.

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
