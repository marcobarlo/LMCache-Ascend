/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "multi_layer_mem_kernels_v3.h"
#include <stdexcept>
#include <string>

template <typename scalar_t, typename slot_t, kvcache_ops::KVCacheFormat kvcache_fmt>
class MultiLayerPagedKVCopyV3 {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

public:
    __aicore__ inline MultiLayerPagedKVCopyV3() {}

    __aicore__ inline void init(GM_ADDR pagedKVCaches, GM_ADDR cacheTensor, GM_ADDR slotmappings,
                                const int64_t hiddenDims, const int32_t numLayers, const int64_t pageBuffSize,
                                const int32_t numTokensChunk, const int64_t perLoopBuffSize,
                                const int32_t maxTokensPerLoop, const bool page2L, AscendC::TPipe *pipe,
                                const int64_t kHiddenDims = 0, const int64_t vHiddenDims = 0,
                                const int64_t dsaHiddenDims = 0,
                                const int64_t blockStrideElems = 0, const int32_t blockSize = 0,
                                const int64_t lmcRowElems = 0)
    {
        this->pipe_ = pipe;
        this->numLayers_ = numLayers;
        this->hiddenDims_ = hiddenDims;
        this->pageBuffSize_ = pageBuffSize;
        this->numTokensChunk_ = numTokensChunk;
        this->maxTokensPerLoop_ = maxTokensPerLoop;
        this->perLoopBuffSize_ = perLoopBuffSize;
        this->page2L_ = page2L;
        this->valid_ = true;
        this->blockStride_ = blockStrideElems;
        this->blockSize_ = blockSize;
        this->lmcRowElems_ = lmcRowElems;

        if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::MLA_KV ||
                      kvcache_fmt == kvcache_ops::KVCacheFormat::DSA_KV) {
            this->kHiddenDims_ = kHiddenDims;
            this->vHiddenDims_ = vHiddenDims;
            this->dsaHiddenDims_ = dsaHiddenDims;
        }

        this->pipe_->InitBuffer(pagedTokenQue_, 2, this->perLoopBuffSize_);
    }

    __aicore__ inline int64_t GetHiddenDims(const int cacheIdx) {
        if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::MLA_KV) {
            return (cacheIdx == 0) ? this->kHiddenDims_ : this->vHiddenDims_;
        } else if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::DSA_KV) {
            if (cacheIdx == 0) return this->kHiddenDims_;
            else if (cacheIdx == 1) return this->vHiddenDims_;
            else return this->dsaHiddenDims_;
        } else {
            return this->hiddenDims_;
        }
    }

    __aicore__ inline int64_t GetLMCBaseOffset(const int cacheIdx) {
        if (this->lmcRowElems_ > 0) {
            return 0;
        }
        if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::MLA_KV) {
            if (cacheIdx == 0) return 0;
            else return this->numLayers_ * this->numTokensChunk_ * this->kHiddenDims_;
        } else if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::DSA_KV) {
            if (cacheIdx == 0) return 0;
            else if (cacheIdx == 1) return this->numLayers_ * this->numTokensChunk_ * this->kHiddenDims_;
            else return this->numLayers_ * this->numTokensChunk_ * (this->kHiddenDims_ + this->vHiddenDims_);
        } else {
            return static_cast<int64_t>(cacheIdx) * this->numLayers_ * this->numTokensChunk_ * this->hiddenDims_;
        }
    }

    __aicore__ inline int64_t GetPlaneRowOffset(const int cacheIdx) {
        if (this->lmcRowElems_ <= 0) {
            return 0;
        }
        if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::MLA_KV) {
            return (cacheIdx == 0) ? 0 : this->kHiddenDims_;
        } else if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::DSA_KV) {
            if (cacheIdx == 0) return 0;
            else if (cacheIdx == 1) return this->kHiddenDims_;
            else return this->kHiddenDims_ + this->vHiddenDims_;
        } else {
            return static_cast<int64_t>(cacheIdx) * this->hiddenDims_;
        }
    }

    __aicore__ inline int64_t PagedTokenOffset(const int64_t pagedOffset,
                                               const int64_t slot,
                                               const int64_t hiddenDims) {
        if (slot < 0 || slot >= this->pageBuffSize_) {
            return -1;
        }
        if (this->blockStride_ > 0 && this->blockSize_ > 0) {
            return pagedOffset + (slot / this->blockSize_) * this->blockStride_
                 + (slot % this->blockSize_) * hiddenDims;
        }
        return pagedOffset + slot * hiddenDims;
    }

    __aicore__ inline int64_t LmcTokenOffset(const int cacheIdx, const int layerIdx,
                                             const int64_t tokenIdx, const int64_t hiddenDims) {
        if (this->lmcRowElems_ > 0) {
            return static_cast<int64_t>(layerIdx) * this->numTokensChunk_ * this->lmcRowElems_
                 + tokenIdx * this->lmcRowElems_
                 + GetPlaneRowOffset(cacheIdx);
        }
        return GetLMCBaseOffset(cacheIdx)
             + static_cast<int64_t>(layerIdx) * this->numTokensChunk_ * hiddenDims
             + tokenIdx * hiddenDims;
    }

    // UB rows must sit on 32 B dataBlock boundaries (MTE). For a 2 B scale
    // plane this pads each token to 32 B in UB only; GM layout is unchanged.
    __aicore__ inline int64_t AlignUp32Bytes(const int64_t bytes) const {
        return (bytes + 31) & ~static_cast<int64_t>(31);
    }

    __aicore__ inline int64_t UbTokenStrideElems(const int64_t hiddenDims) const {
        const int64_t bytes = hiddenDims * static_cast<int64_t>(sizeof(scalar_t));
        return AlignUp32Bytes(bytes) / static_cast<int64_t>(sizeof(scalar_t));
    }

    __aicore__ inline bool IsNarrowPlane(const int64_t hiddenDims) const {
        const int64_t bytes = hiddenDims * static_cast<int64_t>(sizeof(scalar_t));
        return bytes > 0 && (bytes % 32) != 0;
    }

    __aicore__ inline int64_t LmcRowBytes(const int64_t hdBytes) const {
        if (this->lmcRowElems_ > 0) {
            return this->lmcRowElems_ * static_cast<int64_t>(sizeof(scalar_t));
        }
        return hdBytes;
    }

    __aicore__ inline void CopyGmToUb(local_scalar_t dst,
                                      const AscendC::GlobalTensor<scalar_t> &src,
                                      const int64_t srcOffset, const int64_t count) {
        const int64_t bytes = count * static_cast<int64_t>(sizeof(scalar_t));
        const int64_t addrBytes = srcOffset * static_cast<int64_t>(sizeof(scalar_t));
        if (bytes > 0 && (bytes % 32) == 0 && (addrBytes % 32) == 0) {
            AscendC::DataCopy(dst, src[srcOffset], count);
            return;
        }
        AscendC::DataCopyExtParams params{1u, static_cast<uint32_t>(bytes), 0u, 0u, 0u};
        AscendC::DataCopyPadExtParams<scalar_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, src[srcOffset], params, pad);
    }

    __aicore__ inline void CopyUbToGm(const AscendC::GlobalTensor<scalar_t> &dst,
                                      const int64_t dstOffset, local_scalar_t src,
                                      const int64_t count) {
        const int64_t bytes = count * static_cast<int64_t>(sizeof(scalar_t));
        const int64_t addrBytes = dstOffset * static_cast<int64_t>(sizeof(scalar_t));
        if (bytes > 0 && (bytes % 32) == 0 && (addrBytes % 32) == 0) {
            AscendC::DataCopy(dst[dstOffset], src, count);
            return;
        }
        AscendC::DataCopyExtParams params{1u, static_cast<uint32_t>(bytes), 0u, 0u, 0u};
        AscendC::DataCopyPad(dst[dstOffset], src, params);
    }

    // UB stride is in 32-byte dataBlocks (start-to-start). hd=2 -> 1.
    __aicore__ inline uint32_t UbRowStrideBlks(const int64_t hdBytes) const {
        return static_cast<uint32_t>(AlignUp32Bytes(hdBytes) / 32);
    }

    // blockCount-mode: dense paged GM rows -> UB padded to 32 B/row.
    // GM srcStride=0 bytes; UB dstStride = 1 dataBlock (see multi-plane CopyLmcChunkToUb).
    __aicore__ inline void CopyPagedBlockToUb(local_scalar_t dst,
                                              const int64_t srcByteOff,
                                              const int32_t nTok,
                                              const int64_t hdBytes) {
        AscendC::LocalTensor<uint8_t> dstU8 = dst.template ReinterpretCast<uint8_t>();
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(nTok), static_cast<uint32_t>(hdBytes),
            0u, UbRowStrideBlks(hdBytes), 0u};
        AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0u, 0u, 0u};
        AscendC::DataCopyPad(dstU8, this->pagedTokenGlobalU8_[srcByteOff], params, pad);
    }

    // UB padded rows -> token-major (or dense) LMC. dstStride is the GM gap in bytes.
    // UB srcStride skips pad (1 dataBlock for hd<32).
    __aicore__ inline void CopyUbToLmcBlock(const int64_t dstByteOff,
                                            local_scalar_t src,
                                            const int32_t nTok,
                                            const int64_t hdBytes,
                                            const int64_t lmcRowBytes) {
        AscendC::LocalTensor<uint8_t> srcU8 = src.template ReinterpretCast<uint8_t>();
        const uint32_t dstStride = static_cast<uint32_t>(lmcRowBytes - hdBytes);
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(nTok), static_cast<uint32_t>(hdBytes),
            UbRowStrideBlks(hdBytes), dstStride, 0u};
        AscendC::DataCopyPad(this->lmcBufferGlobalU8_[dstByteOff], srcU8, params);
    }

    // Token-major (or dense) LMC -> UB padded rows. srcStride is the GM gap in bytes.
    __aicore__ inline void CopyLmcBlockToUb(local_scalar_t dst,
                                            const int64_t srcByteOff,
                                            const int32_t nTok,
                                            const int64_t hdBytes,
                                            const int64_t lmcRowBytes) {
        AscendC::LocalTensor<uint8_t> dstU8 = dst.template ReinterpretCast<uint8_t>();
        const uint32_t srcStride = static_cast<uint32_t>(lmcRowBytes - hdBytes);
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(nTok), static_cast<uint32_t>(hdBytes),
            srcStride, UbRowStrideBlks(hdBytes), 0u};
        AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0u, 0u, 0u};
        AscendC::DataCopyPad(dstU8, this->lmcBufferGlobalU8_[srcByteOff], params, pad);
    }

    // UB padded rows -> dense paged GM.
    __aicore__ inline void CopyUbToPagedBlock(const int64_t dstByteOff,
                                              local_scalar_t src,
                                              const int32_t nTok,
                                              const int64_t hdBytes) {
        AscendC::LocalTensor<uint8_t> srcU8 = src.template ReinterpretCast<uint8_t>();
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(nTok), static_cast<uint32_t>(hdBytes),
            UbRowStrideBlks(hdBytes), 0u, 0u};
        AscendC::DataCopyPad(this->pagedTokenGlobalU8_[dstByteOff], srcU8, params);
    }

    __aicore__ inline bool CanBulkBlock(const int32_t nTok, const int64_t hdBytes,
                                        const int64_t pagedElemOff) const {
        if (nTok <= 0 || nTok > 4095) {
            return false;
        }
        const int64_t ubBytes = static_cast<int64_t>(nTok) * AlignUp32Bytes(hdBytes);
        if (ubBytes <= 0 || ubBytes > this->perLoopBuffSize_) {
            return false;
        }
        const int64_t addrBytes = pagedElemOff * static_cast<int64_t>(sizeof(scalar_t));
        return (addrBytes % 32) == 0;
    }

    // Per-token path with 32 B UB row stride (Option C). Used for aligned
    // planes and as the fallback when a narrow-plane run is not bulk-eligible.
    __aicore__ inline void CopyTokensPageToLmc(__gm__ slot_t *slotmappingPtr,
                                               const int cacheIdx, const int layerIdx,
                                               const int32_t startTokensIdx, const int32_t endTokensIdx,
                                               const int64_t pagedOffset, const int64_t hiddenDims) {
        const int64_t ubStride = UbTokenStrideElems(hiddenDims);
        local_scalar_t buf = this->pagedTokenQue_.template AllocTensor<scalar_t>();
        for (int64_t tokenIdx = startTokensIdx; tokenIdx < endTokensIdx; tokenIdx++) {
            const int64_t slot = static_cast<int64_t>(slotmappingPtr[tokenIdx]);
            const int64_t tmpPagedOffset = PagedTokenOffset(pagedOffset, slot, hiddenDims);
            if (tmpPagedOffset < 0) {
                continue;
            }
            const int64_t localOff = (tokenIdx - startTokensIdx) * ubStride;
            CopyGmToUb(buf[localOff], this->pagedTokenGlobal_, tmpPagedOffset, hiddenDims);
        }
        pagedTokenQue_.EnQue(buf);
        buf = pagedTokenQue_.DeQue<scalar_t>();
        for (int64_t tokenIdx = startTokensIdx; tokenIdx < endTokensIdx; tokenIdx++) {
            const int64_t slot = static_cast<int64_t>(slotmappingPtr[tokenIdx]);
            if (slot < 0 || slot >= this->pageBuffSize_) {
                continue;
            }
            const int64_t lmcOff = LmcTokenOffset(cacheIdx, layerIdx, tokenIdx, hiddenDims);
            const int64_t localOff = (tokenIdx - startTokensIdx) * ubStride;
            CopyUbToGm(this->lmcBufferGlobal_, lmcOff, buf[localOff], hiddenDims);
        }
        pagedTokenQue_.FreeTensor(buf);
    }

    __aicore__ inline void CopyTokensLmcToPage(__gm__ slot_t *slotmappingPtr,
                                               const int cacheIdx, const int layerIdx,
                                               const int32_t startTokensIdx, const int32_t endTokensIdx,
                                               const int64_t pagedOffset, const int64_t hiddenDims) {
        const int64_t ubStride = UbTokenStrideElems(hiddenDims);
        local_scalar_t buf = this->pagedTokenQue_.template AllocTensor<scalar_t>();
        for (int64_t tokenIdx = startTokensIdx; tokenIdx < endTokensIdx; tokenIdx++) {
            const int64_t slot = static_cast<int64_t>(slotmappingPtr[tokenIdx]);
            if (slot < 0 || slot >= this->pageBuffSize_) {
                continue;
            }
            const int64_t lmcOff = LmcTokenOffset(cacheIdx, layerIdx, tokenIdx, hiddenDims);
            const int64_t localOff = (tokenIdx - startTokensIdx) * ubStride;
            CopyGmToUb(buf[localOff], this->lmcBufferGlobal_, lmcOff, hiddenDims);
        }
        pagedTokenQue_.EnQue(buf);
        buf = pagedTokenQue_.DeQue<scalar_t>();
        for (int64_t tokenIdx = startTokensIdx; tokenIdx < endTokensIdx; tokenIdx++) {
            const int64_t slot = static_cast<int64_t>(slotmappingPtr[tokenIdx]);
            const int64_t tmpPagedOffset = PagedTokenOffset(pagedOffset, slot, hiddenDims);
            if (tmpPagedOffset < 0) {
                continue;
            }
            const int64_t localOff = (tokenIdx - startTokensIdx) * ubStride;
            CopyUbToGm(this->pagedTokenGlobal_, tmpPagedOffset, buf[localOff], hiddenDims);
        }
        pagedTokenQue_.FreeTensor(buf);
    }

    // Dedicated narrow-plane path: one DataCopyPad pair per contiguous paged-block
    // run (blockCount mode keeps every UB row 32 B aligned).
    __aicore__ inline void TransferNarrowPlane(__gm__ slot_t *slotmappingPtr,
                                               const int cacheIdx, const int layerIdx,
                                               const int32_t startTokensIdx, const int32_t endTokensIdx,
                                               const int64_t pagedOffset, const int64_t hiddenDims,
                                               const bool page2L) {
        const int64_t hdBytes = hiddenDims * static_cast<int64_t>(sizeof(scalar_t));
        const int64_t lmcRowBytes = LmcRowBytes(hdBytes);
        int32_t i = startTokensIdx;
        while (i < endTokensIdx) {
            const int64_t firstSlot = static_cast<int64_t>(slotmappingPtr[i]);
            const int64_t firstPaged = PagedTokenOffset(pagedOffset, firstSlot, hiddenDims);
            if (firstPaged < 0) {
                i++;
                continue;
            }
            const int32_t runStart = i;
            int32_t nTok = 1;
            i++;
            while (i < endTokensIdx) {
                const int64_t s = static_cast<int64_t>(slotmappingPtr[i]);
                if (s != firstSlot + static_cast<int64_t>(i - runStart)) {
                    break;
                }
                if (this->blockSize_ > 0 &&
                    (s / this->blockSize_) != (firstSlot / this->blockSize_)) {
                    break;
                }
                if (PagedTokenOffset(pagedOffset, s, hiddenDims) < 0) {
                    break;
                }
                nTok++;
                i++;
            }
            if (!CanBulkBlock(nTok, hdBytes, firstPaged)) {
                if (page2L) {
                    CopyTokensPageToLmc(slotmappingPtr, cacheIdx, layerIdx,
                                        runStart, runStart + nTok, pagedOffset, hiddenDims);
                } else {
                    CopyTokensLmcToPage(slotmappingPtr, cacheIdx, layerIdx,
                                        runStart, runStart + nTok, pagedOffset, hiddenDims);
                }
                continue;
            }
            const int64_t pagedByteOff = firstPaged * static_cast<int64_t>(sizeof(scalar_t));
            const int64_t lmcElemOff = LmcTokenOffset(cacheIdx, layerIdx, runStart, hiddenDims);
            const int64_t lmcByteOff = lmcElemOff * static_cast<int64_t>(sizeof(scalar_t));
            local_scalar_t buf = this->pagedTokenQue_.template AllocTensor<scalar_t>();
            if (page2L) {
                CopyPagedBlockToUb(buf, pagedByteOff, nTok, hdBytes);
                pagedTokenQue_.EnQue(buf);
                buf = pagedTokenQue_.DeQue<scalar_t>();
                CopyUbToLmcBlock(lmcByteOff, buf, nTok, hdBytes, lmcRowBytes);
            } else {
                CopyLmcBlockToUb(buf, lmcByteOff, nTok, hdBytes, lmcRowBytes);
                pagedTokenQue_.EnQue(buf);
                buf = pagedTokenQue_.DeQue<scalar_t>();
                CopyUbToPagedBlock(pagedByteOff, buf, nTok, hdBytes);
            }
            pagedTokenQue_.FreeTensor(buf);
        }
    }

    __aicore__ inline void _page2LTransfer(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t* cacheTensor,
                                           __gm__ uint8_t *slotmappings, const int cacheIdx,
                                           const int layerIdx, const int32_t startTokensIdx,
                                           const int32_t endTokensIdx,
                                           const int32_t actualTokensPerInnerLoop,
                                           const int64_t pagedOffset) {
        (void)pagedKVCaches;
        (void)cacheTensor;
        (void)actualTokensPerInnerLoop;
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);
        const int64_t hiddenDims = GetHiddenDims(cacheIdx);
        if (IsNarrowPlane(hiddenDims)) {
            TransferNarrowPlane(slotmappingPtr, cacheIdx, layerIdx,
                                startTokensIdx, endTokensIdx, pagedOffset, hiddenDims, true);
            return;
        }
        CopyTokensPageToLmc(slotmappingPtr, cacheIdx, layerIdx,
                            startTokensIdx, endTokensIdx, pagedOffset, hiddenDims);
    }

    __aicore__ inline void _L2PageTransfer(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t* cacheTensor,
                                           __gm__ uint8_t *slotmappings, const int cacheIdx,
                                           const int layerIdx, const int32_t startTokensIdx,
                                           const int32_t endTokensIdx,
                                           const int32_t actualTokensPerInnerLoop,
                                           const int64_t pagedOffset) {
        (void)pagedKVCaches;
        (void)cacheTensor;
        (void)actualTokensPerInnerLoop;
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);
        const int64_t hiddenDims = GetHiddenDims(cacheIdx);
        if (IsNarrowPlane(hiddenDims)) {
            TransferNarrowPlane(slotmappingPtr, cacheIdx, layerIdx,
                                startTokensIdx, endTokensIdx, pagedOffset, hiddenDims, false);
            return;
        }
        CopyTokensLmcToPage(slotmappingPtr, cacheIdx, layerIdx,
                            startTokensIdx, endTokensIdx, pagedOffset, hiddenDims);
    }

    __aicore__ inline void processLayerCache(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t* cacheTensor,
                                             __gm__ uint8_t *slotmappings, const int cacheIdx,
                                             const int layerIdx, const bool page2L)
    {
        const int64_t hiddenDims = GetHiddenDims(cacheIdx);

        __gm__ uint8_t *pagedLayerKVCaches =
            kvcache_ops::GetLayerBasePtr<kvcache_fmt>(pagedKVCaches, layerIdx, cacheIdx);

        int64_t pagedOffset = 0;
        if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::MERGED_KV) {
            pagedOffset = cacheIdx * this->pageBuffSize_ * hiddenDims;
        }

        int64_t pagedViewElems = this->pageBuffSize_ * hiddenDims;
        if (this->blockStride_ > 0 && this->blockSize_ > 0) {
            const int64_t nBlocks =
                (this->pageBuffSize_ + this->blockSize_ - 1) / this->blockSize_;
            pagedViewElems = nBlocks * this->blockStride_ + hiddenDims;
        }
        this->pagedTokenGlobal_.SetGlobalBuffer(
            reinterpret_cast<__gm__ scalar_t*>(pagedLayerKVCaches), pagedViewElems);
        this->pagedTokenGlobalU8_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(pagedLayerKVCaches),
            pagedViewElems * static_cast<int64_t>(sizeof(scalar_t)));

        int64_t lmcViewElems = this->numTokensChunk_ * hiddenDims;
        if (this->lmcRowElems_ > 0) {
            lmcViewElems = static_cast<int64_t>(this->numLayers_)
                         * this->numTokensChunk_ * this->lmcRowElems_;
        }
        this->lmcBufferGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(cacheTensor),
                                               lmcViewElems);
        this->lmcBufferGlobalU8_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(cacheTensor),
            lmcViewElems * static_cast<int64_t>(sizeof(scalar_t)));

        int32_t startTokensIdx;
        int32_t endTokensIdx;
        int32_t actualTokensPerInnerLoop;

        for (startTokensIdx = 0; startTokensIdx < this->numTokensChunk_; startTokensIdx += this->maxTokensPerLoop_) {
            endTokensIdx = startTokensIdx + this->maxTokensPerLoop_;
            endTokensIdx = min(endTokensIdx, this->numTokensChunk_);
            actualTokensPerInnerLoop = endTokensIdx - startTokensIdx;

            if (page2L) {
                this->_page2LTransfer(pagedKVCaches, cacheTensor, slotmappings, cacheIdx, layerIdx,
                                     startTokensIdx, endTokensIdx, actualTokensPerInnerLoop, pagedOffset);
            } else {
                this->_L2PageTransfer(pagedKVCaches, cacheTensor, slotmappings, cacheIdx, layerIdx,
                                     startTokensIdx, endTokensIdx, actualTokensPerInnerLoop, pagedOffset);
            }
        }
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> pagedTokenQue_;

    AscendC::GlobalTensor<scalar_t> pagedTokenGlobal_;
    AscendC::GlobalTensor<scalar_t> lmcBufferGlobal_;
    AscendC::GlobalTensor<uint8_t> pagedTokenGlobalU8_;
    AscendC::GlobalTensor<uint8_t> lmcBufferGlobalU8_;
    int32_t numLayers_;
    int64_t pageBuffSize_;
    int64_t hiddenDims_;
    int32_t numTokensChunk_;
    int32_t maxTokensPerLoop_;
    int64_t perLoopBuffSize_;
    bool valid_;
    bool page2L_;

    int64_t kHiddenDims_{0};
    int64_t vHiddenDims_{0};
    int64_t dsaHiddenDims_{0};
    int64_t blockStride_{0};
    int32_t blockSize_{0};
    int64_t lmcRowElems_{0};
};

#define MULTI_LAYER_PAGED_KV_COPY_V3_KERNEL_NAME(TYPE, SLOTTYPE, FMT) \
    multi_layer_paged_kv_copy_v3_##TYPE##_##SLOTTYPE##_##FMT

#define MULTI_LAYER_PAGED_KV_COPY_V3_DECLARE(TYPE, SLOTTYPE, FMT)                                     \
    extern "C" __global__ __aicore__ void MULTI_LAYER_PAGED_KV_COPY_V3_KERNEL_NAME(TYPE, SLOTTYPE, FMT)( \
        __gm__ uint8_t* pagedKVCaches, __gm__ uint8_t* dstCacheTensor, __gm__ uint8_t* slotmappings,    \
        const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,                           \
        const int64_t pageBuffSize, const int32_t numTokensChunk,                                       \
        const int64_t perLoopBuffer, const int32_t maxTokensPerLoop, const bool page2L,                 \
        const int64_t kHiddenDims, const int64_t vHiddenDims, const int64_t dsaHiddenDims,              \
        const int64_t blockStrideElems, const int32_t blockSize, const int64_t lmcRowElems)             \
    {                                                                                                   \
        AscendC::TPipe pipe;                                                                            \
        MultiLayerPagedKVCopyV3<TYPE, SLOTTYPE, kvcache_ops::KVCacheFormat::FMT> op{};                  \
        int32_t bIdx = AscendC::GetBlockIdx();                                                          \
        int32_t launchedCores = AscendC::GetBlockNum();                                                 \
        int32_t layersPerCore = (numLayers + launchedCores - 1) / launchedCores;                        \
        int32_t startLayersIdx = bIdx * layersPerCore;                                                  \
        int32_t endLayersIdx = min(numLayers, startLayersIdx + layersPerCore);                          \
        op.init(pagedKVCaches, dstCacheTensor, slotmappings, hiddenDims,                                \
                numLayers, pageBuffSize, numTokensChunk, perLoopBuffer, maxTokensPerLoop, page2L, &pipe, \
                kHiddenDims, vHiddenDims, dsaHiddenDims,                                                \
                blockStrideElems, blockSize, lmcRowElems);                                              \
        for (int32_t layerIdx = startLayersIdx; layerIdx < endLayersIdx; layerIdx++) {                  \
            for (int32_t cacheIdx = 0; cacheIdx < kvs; cacheIdx++) {                                    \
                op.processLayerCache(pagedKVCaches, dstCacheTensor, slotmappings, cacheIdx, layerIdx, page2L); \
            }                                                                                           \
        }                                                                                               \
    }

#define EXPAND_FMT_V3(TYPE, SLOTTYPE) \
    MULTI_LAYER_PAGED_KV_COPY_V3_DECLARE(TYPE, SLOTTYPE, MERGED_KV) \
    MULTI_LAYER_PAGED_KV_COPY_V3_DECLARE(TYPE, SLOTTYPE, SEPARATE_KV) \
    MULTI_LAYER_PAGED_KV_COPY_V3_DECLARE(TYPE, SLOTTYPE, MLA_KV) \
    MULTI_LAYER_PAGED_KV_COPY_V3_DECLARE(TYPE, SLOTTYPE, DSA_KV)

#define EXPAND_SLOT_V3(TYPE) \
    EXPAND_FMT_V3(TYPE, int32_t) \
    EXPAND_FMT_V3(TYPE, int64_t)

EXPAND_SLOT_V3(half)
EXPAND_SLOT_V3(int8_t)
#if (__CCE_AICORE__ >= 220)
EXPAND_SLOT_V3(bfloat16_t)
#endif

namespace kvcache_ops {

#define SPECIALIZE_V3_LAUNCHER(TYPE, SLOTTYPE, FMT)                                                    \
template<>                                                                                             \
struct V3Launcher<TYPE, SLOTTYPE, KVCacheFormat::FMT> {                                                \
    static void Launch(uint32_t blockDim, void *stream,                                                \
                      uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, uint8_t *slotmappings,          \
                      const V3Config& config,                                                          \
                      int64_t kHiddenDims = 0, int64_t vHiddenDims = 0, int64_t dsaHiddenDims = 0)     \
    {                                                                                                  \
        MULTI_LAYER_PAGED_KV_COPY_V3_KERNEL_NAME(TYPE, SLOTTYPE, FMT)<<<blockDim, nullptr, stream>>>( \
            pagedKVCaches, dstCacheTensor, slotmappings,                                               \
            config.common.hiddenDims, config.common.kvs, config.common.numLayers,                      \
            config.common.pageBuffSize, config.common.numTokensChunk,                                  \
            config.perLoopBuffSize, config.maxTokensPerLoop, config.common.page2L,                     \
            kHiddenDims, vHiddenDims, dsaHiddenDims,                                                   \
            config.blockStrideElems, config.blockSize, config.lmcRowElems);                            \
    }                                                                                                  \
};

#define EXPAND_V3_LAUNCHER_FMT(TYPE, SLOTTYPE) \
    SPECIALIZE_V3_LAUNCHER(TYPE, SLOTTYPE, MERGED_KV) \
    SPECIALIZE_V3_LAUNCHER(TYPE, SLOTTYPE, SEPARATE_KV) \
    SPECIALIZE_V3_LAUNCHER(TYPE, SLOTTYPE, MLA_KV) \
    SPECIALIZE_V3_LAUNCHER(TYPE, SLOTTYPE, DSA_KV)

#define EXPAND_V3_LAUNCHER_SLOT(TYPE) \
    EXPAND_V3_LAUNCHER_FMT(TYPE, int32_t) \
    EXPAND_V3_LAUNCHER_FMT(TYPE, int64_t)

EXPAND_V3_LAUNCHER_SLOT(half)
EXPAND_V3_LAUNCHER_SLOT(int8_t)
#if (ASCEND_AICORE_ARCH >= 220)
EXPAND_V3_LAUNCHER_SLOT(bfloat16_t)
#endif

void multi_layer_kv_transfer_kernel_v3(kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
                                              const kvcache_ops::KVCacheFormat kvcacheFormat, uint32_t blockDim, void *stream,
                                              uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, uint8_t *slotmappings,
                                              const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,
                                              const int64_t pageBuffSize, const int32_t numTokensChunk,
                                              const int64_t perLoopBuffer, const int32_t maxTokensPerLoop,
                                              const bool page2L,
                                              const int64_t kHiddenDims, const int64_t vHiddenDims,
                                              const int64_t dsaHiddenDims,
                                              const int64_t blockStrideElems, const int32_t blockSize,
                                              const int64_t lmcRowElems)
{
    auto config = kvcache_ops::MakeV3Config(
        hiddenDims, numLayers, pageBuffSize, numTokensChunk, page2L, kvs,
        perLoopBuffer, maxTokensPerLoop, blockStrideElems, blockSize, lmcRowElems
    );

    switch(type) {
        case kvcache_ops::AscendType::FP16:
            kvcache_ops::dispatch_paged_kernel_on_slot_type<kvcache_ops::V3Launcher, half>(
                slotType, kvcacheFormat, blockDim, stream,
                pagedKVCaches, dstCacheTensor, slotmappings, config, kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
#if (ASCEND_AICORE_ARCH >= 220)
        case kvcache_ops::AscendType::BF16:
            kvcache_ops::dispatch_paged_kernel_on_slot_type<kvcache_ops::V3Launcher, bfloat16_t>(
                slotType, kvcacheFormat, blockDim, stream,
                pagedKVCaches, dstCacheTensor, slotmappings, config, kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
#endif
        case kvcache_ops::AscendType::INT8:
            kvcache_ops::dispatch_paged_kernel_on_slot_type<kvcache_ops::V3Launcher, int8_t>(
                slotType, kvcacheFormat, blockDim, stream,
                pagedKVCaches, dstCacheTensor, slotmappings, config, kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
        default:
            ASCENDC_REPORT_NOT_SUPPORT(false, std::to_string(static_cast<int>(type)) + " is not supported.")
            throw std::runtime_error("Scalar type: " + std::to_string(static_cast<int>(type)) + " not supported. This should not have happened.");
    }
}

} // namespace kvcache_ops
