# MP Mode (Multiprocess LMCache) on Ascend NPU

MP mode is the multiprocess deployment architecture of LMCache: the KV cache is owned by a standalone `lmcache server` process, and vLLM attaches to it through the `LMCacheMPConnector`. The server owns L1/L2 storage, eviction, and lookup; the engine talks to it over the MP protocol (ZMQ control plane + shared-memory / device-handle data plane). Compared with running LMCache inside the vLLM process, this keeps cache work (store, retrieve, chunking, eviction) out of the engine loop and allows one cache to be shared by several engines.

The data plane has two transfer modes, selected with the `lmcache.mp.mp_transfer_mode` key in `kv_connector_extra_config`:

| | `lmcache_driven` | `engine_driven` |
| :--- | :--- | :--- |
| Data path | The LMCache server pulls/pushes KV through device IPC handles | The vLLM workers gather/scatter the KV copies themselves, zero-copy from a named shared-memory pool |
| KV cache formats | All KV cache formats — standard GQA/MHA layouts as well as vLLM hybrid KV cache management formats (incl. the packed MLA/DSA `(latent, scale)` layout of DeepSeek-V4) | Single KV-cache format models only (standard GQA/MHA layouts, e.g. Qwen3) |
| Server requirements | `--chunk-size 4096` for DeepSeek-V4 (see [lmcache_driven.md](lmcache_driven.md)) | `--shm-name <name> --no-l1-use-lazy` are mandatory (see [engine_driven.md](engine_driven.md)) |
| Deployment guide | [lmcache_driven.md](lmcache_driven.md) | [engine_driven.md](engine_driven.md) |

On NPU, leaving `lmcache.mp.mp_transfer_mode` unset (or `auto`) resolves to `engine_driven`; for DeepSeek-V4 you must set `lmcache_driven` explicitly.

## Documents

- [lmcache_driven.md](lmcache_driven.md) — the `lmcache_driven` transfer mode full deployment guide: environment preparation, container startup, source installation, server/vLLM startup, and verification (DeepSeek-V4 example).
- [engine_driven.md](engine_driven.md) — deploy and run the `engine_driven` transfer mode (Qwen3 example).

## Relationship to the LMCache dev branch

The MP architecture (the `lmcache server` CLI, the `LMCacheMPConnector`, and the transfer-mode layer) is developed on the LMCache `dev` branch — the main line — and the matching Ascend backends (NPU platform routing, block-transfer ops, hccl/hixl bindings) live in the LMCache-Ascend repository. Install both from the official repositories only — see the installation section of [lmcache_driven.md](lmcache_driven.md) for the exact procedure. The classic in-process connector (`LMCacheAscendConnectorV1Dynamic`) is a separate integration that uses its own LMCache release line; its installation must not be mixed with the MP-mode checkout in one Python environment.
