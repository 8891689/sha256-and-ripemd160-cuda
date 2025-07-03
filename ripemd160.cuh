/*MIT License
Copyright (c) 2025 8891689
//author： https://github.com/8891689
*/
// ripemd160.cuh
#ifndef RIPEMD160_CUH_
#define RIPEMD160_CUH_

#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ===================================================================
//  宏定義
// ===================================================================
#define DIGEST_SIZE 20
#define BLOCK_SIZE 64

__constant__ uint32_t d_KL[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
__constant__ uint32_t d_KR[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};

#define ROL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define F(x, y, z) ((x) ^ (y) ^ (z))
#define G(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define H(x, y, z) (((x) | ~(y)) ^ (z))
#define I(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define J(x, y, z) ((x) ^ ((y) | ~(z)))

#define FF_L(a, b, c, d, e, k, s) { (a) += F((b), (c), (d)) + X[k] + d_KL[0]; (a) = ROL32((a), (s)) + (e); (c) = ROL32((c), 10); }
#define GG_L(a, b, c, d, e, k, s) { (a) += G((b), (c), (d)) + X[k] + d_KL[1]; (a) = ROL32((a), (s)) + (e); (c) = ROL32((c), 10); }
#define HH_L(a, b, c, d, e, k, s) { (a) += H((b), (c), (d)) + X[k] + d_KL[2]; (a) = ROL32((a), (s)) + (e); (c) = ROL32((c), 10); }
#define II_L(a, b, c, d, e, k, s) { (a) += I((b), (c), (d)) + X[k] + d_KL[3]; (a) = ROL32((a), (s)) + (e); (c) = ROL32((c), 10); }
#define JJ_L(a, b, c, d, e, k, s) { (a) += J((b), (c), (d)) + X[k] + d_KL[4]; (a) = ROL32((a), (s)) + (e); (c) = ROL32((c), 10); }

#define FF_R(a, b, c, d, e, k, s) { (a) += J((b), (c), (d)) + X[k] + d_KR[0]; (a) = ROL32((a), (s)) + (e); (c) = ROL32((c), 10); }
#define GG_R(a, b, c, d, e, k, s) { (a) += I((b), (c), (d)) + X[k] + d_KR[1]; (a) = ROL32((a), (s)) + (e); (c) = ROL32((c), 10); }
#define HH_R(a, b, c, d, e, k, s) { (a) += H((b), (c), (d)) + X[k] + d_KR[2]; (a) = ROL32((a), (s)) + (e); (c) = ROL32((c), 10); }
#define II_R(a, b, c, d, e, k, s) { (a) += G((b), (c), (d)) + X[k] + d_KR[3]; (a) = ROL32((a), (s)) + (e); (c) = ROL32((c), 10); }
#define JJ_R(a, b, c, d, e, k, s) { (a) += F((b), (c), (d)) + X[k] + d_KR[4]; (a) = ROL32((a), (s)) + (e); (c) = ROL32((c), 10); }
// ===================================================================
//  錯誤檢測
// ===================================================================
#define CUDA_CHECK(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t err, const char *file, const int line) {
    if (cudaSuccess != err) {
        const char* errName = cudaGetErrorName(err);
        const char* errString = cudaGetErrorString(err);
        char error_msg[512];
        snprintf(error_msg, sizeof(error_msg), "CUDA error in file '%s' line %d: %s (%d) - %s\n",
                 file, line, errName, err, errString);
        throw std::runtime_error(error_msg);
    }
}
// ===================================================================
//  內核
// ===================================================================
__device__ __forceinline__ void cuda_compress_from_X(uint32_t state[5], const uint32_t X[16]) {
    uint32_t a0 = state[0], b0 = state[1], c0 = state[2], d0 = state[3], e0 = state[4];
    uint32_t a1 = a0, b1 = b0, c1 = c0, d1 = d0, e1 = e0; 

    FF_L(a0, b0, c0, d0, e0, 0, 11); FF_L(e0, a0, b0, c0, d0, 1, 14); FF_L(d0, e0, a0, b0, c0, 2, 15); FF_L(c0, d0, e0, a0, b0, 3, 12);
    FF_L(b0, c0, d0, e0, a0, 4, 5);  FF_L(a0, b0, c0, d0, e0, 5, 8);  FF_L(e0, a0, b0, c0, d0, 6, 7);  FF_L(d0, e0, a0, b0, c0, 7, 9);
    FF_L(c0, d0, e0, a0, b0, 8, 11); FF_L(b0, c0, d0, e0, a0, 9, 13); FF_L(a0, b0, c0, d0, e0, 10, 14);FF_L(e0, a0, b0, c0, d0, 11, 15);
    FF_L(d0, e0, a0, b0, c0, 12, 6); FF_L(c0, d0, e0, a0, b0, 13, 7); FF_L(b0, c0, d0, e0, a0, 14, 9); FF_L(a0, b0, c0, d0, e0, 15, 8);
    GG_L(e0, a0, b0, c0, d0, 7, 7);  GG_L(d0, e0, a0, b0, c0, 4, 6);  GG_L(c0, d0, e0, a0, b0, 13, 8); GG_L(b0, c0, d0, e0, a0, 1, 13);
    GG_L(a0, b0, c0, d0, e0, 10, 11);GG_L(e0, a0, b0, c0, d0, 6, 9);  GG_L(d0, e0, a0, b0, c0, 15, 7); GG_L(c0, d0, e0, a0, b0, 3, 15);
    GG_L(b0, c0, d0, e0, a0, 12, 7); GG_L(a0, b0, c0, d0, e0, 0, 12); GG_L(e0, a0, b0, c0, d0, 9, 15); GG_L(d0, e0, a0, b0, c0, 5, 9);
    GG_L(c0, d0, e0, a0, b0, 2, 11); GG_L(b0, c0, d0, e0, a0, 14, 7); GG_L(a0, b0, c0, d0, e0, 11, 13);GG_L(e0, a0, b0, c0, d0, 8, 12);
    HH_L(d0, e0, a0, b0, c0, 3, 11); HH_L(c0, d0, e0, a0, b0, 10, 13);HH_L(b0, c0, d0, e0, a0, 14, 6); HH_L(a0, b0, c0, d0, e0, 4, 7);
    HH_L(e0, a0, b0, c0, d0, 9, 14); HH_L(d0, e0, a0, b0, c0, 15, 9); HH_L(c0, d0, e0, a0, b0, 8, 13); HH_L(b0, c0, d0, e0, a0, 1, 15);
    HH_L(a0, b0, c0, d0, e0, 2, 14); HH_L(e0, a0, b0, c0, d0, 7, 8);  HH_L(d0, e0, a0, b0, c0, 0, 13); HH_L(c0, d0, e0, a0, b0, 6, 6);
    HH_L(b0, c0, d0, e0, a0, 13, 5); HH_L(a0, b0, c0, d0, e0, 11, 12);HH_L(e0, a0, b0, c0, d0, 5, 7);  HH_L(d0, e0, a0, b0, c0, 12, 5);
    II_L(c0, d0, e0, a0, b0, 1, 11); II_L(b0, c0, d0, e0, a0, 9, 12); II_L(a0, b0, c0, d0, e0, 11, 14);II_L(e0, a0, b0, c0, d0, 10, 15);
    II_L(d0, e0, a0, b0, c0, 0, 14); II_L(c0, d0, e0, a0, b0, 8, 15); II_L(b0, c0, d0, e0, a0, 12, 9); II_L(a0, b0, c0, d0, e0, 4, 8);
    II_L(e0, a0, b0, c0, d0, 13, 9); II_L(d0, e0, a0, b0, c0, 3, 14); II_L(c0, d0, e0, a0, b0, 7, 5);  II_L(b0, c0, d0, e0, a0, 15, 6);
    II_L(a0, b0, c0, d0, e0, 14, 8); II_L(e0, a0, b0, c0, d0, 5, 6);  II_L(d0, e0, a0, b0, c0, 6, 5);  II_L(c0, d0, e0, a0, b0, 2, 12);
    JJ_L(b0, c0, d0, e0, a0, 4, 9);  JJ_L(a0, b0, c0, d0, e0, 0, 15); JJ_L(e0, a0, b0, c0, d0, 5, 5);  JJ_L(d0, e0, a0, b0, c0, 9, 11);
    JJ_L(c0, d0, e0, a0, b0, 7, 6);  JJ_L(b0, c0, d0, e0, a0, 12, 8); JJ_L(a0, b0, c0, d0, e0, 2, 13); JJ_L(e0, a0, b0, c0, d0, 10, 12);
    JJ_L(d0, e0, a0, b0, c0, 14, 5); JJ_L(c0, d0, e0, a0, b0, 1, 12); JJ_L(b0, c0, d0, e0, a0, 3, 13); JJ_L(a0, b0, c0, d0, e0, 8, 14);
    JJ_L(e0, a0, b0, c0, d0, 11, 11);JJ_L(d0, e0, a0, b0, c0, 6, 8);  JJ_L(c0, d0, e0, a0, b0, 15, 5); JJ_L(b0, c0, d0, e0, a0, 13, 6);
    FF_R(a1, b1, c1, d1, e1, 5, 8); FF_R(e1, a1, b1, c1, d1, 14, 9); FF_R(d1, e1, a1, b1, c1, 7, 9); FF_R(c1, d1, e1, a1, b1, 0, 11);
    FF_R(b1, c1, d1, e1, a1, 9, 13); FF_R(a1, b1, c1, d1, e1, 2, 15); FF_R(e1, a1, b1, c1, d1, 11, 15);FF_R(d1, e1, a1, b1, c1, 4, 5);
    FF_R(c1, d1, e1, a1, b1, 13, 7); FF_R(b1, c1, d1, e1, a1, 6, 7); FF_R(a1, b1, c1, d1, e1, 15, 8); FF_R(e1, a1, b1, c1, d1, 8, 11);
    FF_R(d1, e1, a1, b1, c1, 1, 14); FF_R(c1, d1, e1, a1, b1, 10, 14);FF_R(b1, c1, d1, e1, a1, 3, 12); FF_R(a1, b1, c1, d1, e1, 12, 6);
    GG_R(e1, a1, b1, c1, d1, 6, 9); GG_R(d1, e1, a1, b1, c1, 11, 13);GG_R(c1, d1, e1, a1, b1, 3, 15); GG_R(b1, c1, d1, e1, a1, 7, 7);
    GG_R(a1, b1, c1, d1, e1, 0, 12); GG_R(e1, a1, b1, c1, d1, 13, 8); GG_R(d1, e1, a1, b1, c1, 5, 9); GG_R(c1, d1, e1, a1, b1, 10, 11);
    GG_R(b1, c1, d1, e1, a1, 14, 7); GG_R(a1, b1, c1, d1, e1, 15, 7); GG_R(e1, a1, b1, c1, d1, 8, 12); GG_R(d1, e1, a1, b1, c1, 12, 7);
    GG_R(c1, d1, e1, a1, b1, 4, 6); GG_R(b1, c1, d1, e1, a1, 9, 15); GG_R(a1, b1, c1, d1, e1, 1, 13); GG_R(e1, a1, b1, c1, d1, 2, 11);
    HH_R(d1, e1, a1, b1, c1, 15, 9); HH_R(c1, d1, e1, a1, b1, 5, 7); HH_R(b1, c1, d1, e1, a1, 1, 15); HH_R(a1, b1, c1, d1, e1, 3, 11);
    HH_R(e1, a1, b1, c1, d1, 7, 8); HH_R(d1, e1, a1, b1, c1, 14, 6); HH_R(c1, d1, e1, a1, b1, 6, 6); HH_R(b1, c1, d1, e1, a1, 9, 14);
    HH_R(a1, b1, c1, d1, e1, 11, 12);HH_R(e1, a1, b1, c1, d1, 8, 13); HH_R(d1, e1, a1, b1, c1, 12, 5); HH_R(c1, d1, e1, a1, b1, 2, 14);
    HH_R(b1, c1, d1, e1, a1, 10, 13);HH_R(a1, b1, c1, d1, e1, 0, 13); HH_R(e1, a1, b1, c1, d1, 4, 7); HH_R(d1, e1, a1, b1, c1, 13, 5);
    II_R(c1, d1, e1, a1, b1, 8, 15); II_R(b1, c1, d1, e1, a1, 6, 5); II_R(a1, b1, c1, d1, e1, 4, 8); II_R(e1, a1, b1, c1, d1, 1, 11);
    II_R(d1, e1, a1, b1, c1, 3, 14); II_R(c1, d1, e1, a1, b1, 11, 14);II_R(b1, c1, d1, e1, a1, 15, 6); II_R(a1, b1, c1, d1, e1, 0, 14);
    II_R(e1, a1, b1, c1, d1, 5, 6); II_R(d1, e1, a1, b1, c1, 12, 9); II_R(c1, d1, e1, a1, b1, 2, 12); II_R(b1, c1, d1, e1, a1, 13, 9);
    II_R(a1, b1, c1, d1, e1, 9, 12); II_R(e1, a1, b1, c1, d1, 7, 5); II_R(d1, e1, a1, b1, c1, 10, 15);II_R(c1, d1, e1, a1, b1, 14, 8);
    JJ_R(b1, c1, d1, e1, a1, 12, 8); JJ_R(a1, b1, c1, d1, e1, 15, 5); JJ_R(e1, a1, b1, c1, d1, 10, 12);JJ_R(d1, e1, a1, b1, c1, 4, 9);
    JJ_R(c1, d1, e1, a1, b1, 1, 12); JJ_R(b1, c1, d1, e1, a1, 5, 5); JJ_R(a1, b1, c1, d1, e1, 8, 14); JJ_R(e1, a1, b1, c1, d1, 7, 6);
    JJ_R(d1, e1, a1, b1, c1, 6, 8); JJ_R(c1, d1, e1, a1, b1, 2, 13); JJ_R(b1, c1, d1, e1, a1, 13, 6); JJ_R(a1, b1, c1, d1, e1, 14, 5);
    JJ_R(e1, a1, b1, c1, d1, 0, 15); JJ_R(d1, e1, a1, b1, c1, 3, 13); JJ_R(c1, d1, e1, a1, b1, 9, 11); JJ_R(b1, c1, d1, e1, a1, 11, 11);

    uint32_t T = state[1] + c0 + d1;
    state[1] = state[2] + d0 + e1;
    state[2] = state[3] + e0 + a1;
    state[3] = state[4] + a0 + b1;
    state[4] = state[0] + b0 + c1;
    state[0] = T;
}

// ===================================================================
//  內核內循環以進行精確性能測試
// ===================================================================
__global__ void ripemd160_fixed8_kernel(
    const uint64_t* d_messages,
    uint8_t* d_digests,
    uint32_t num_lanes,
    uint32_t iterations)
{
    const int num_messages_per_thread = 4;
    int thread_base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * num_messages_per_thread;
    int stride = gridDim.x * blockDim.x * num_messages_per_thread;

    for (int base_idx = thread_base_idx; base_idx < num_lanes; base_idx += stride) {
        
        #pragma unroll
        for (int i = 0; i < num_messages_per_thread; i++) {
            int lane_idx = base_idx + i;
            if (lane_idx >= num_lanes) break;

            uint64_t message = d_messages[lane_idx];

            uint32_t state[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
            
            for (uint32_t iter = 0; iter < iterations; ++iter) {
                
                uint32_t X[16] = {0};
                
                if (iter == 0) {
                    X[0] = (uint32_t)message;
                    X[1] = (uint32_t)(message >> 32);
                    X[2] = 0x00000080;
                    X[14] = 64; 
                } else {
                    X[0] = state[0];
                    X[1] = state[1];
                    X[2] = state[2];
                    X[3] = state[3];
                    X[4] = state[4];
                    X[5] = 0x00000080; 
                    X[14] = 160; 
            
                    state[0] = 0x67452301; state[1] = 0xEFCDAB89; state[2] = 0x98BADCFE;
                    state[3] = 0x10325476; state[4] = 0xC3D2E1F0;
                }

                cuda_compress_from_X(state, X);
            }
            
            uint8_t* digest = d_digests + lane_idx * DIGEST_SIZE;
            #pragma unroll
            for (int j = 0; j < 5; ++j) {
                ((uint32_t*)digest)[j] = state[j];
            }
        }
    }
}

// ===================================================================
//  通用內核和類（保留用於驗證）
// ===================================================================
__global__ void ripemd160_kernel_grid_stride(
    const uint8_t* d_messages, const uint32_t* d_offsets, 
    const uint64_t* d_lengths, uint32_t num_lanes, uint8_t* d_digests) 
{
    int lane_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; lane_idx < num_lanes; lane_idx += stride) {
        const uint8_t* message = d_messages + d_offsets[lane_idx];
        uint64_t len = d_lengths[lane_idx];
        uint32_t state[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
        uint64_t full_blocks_count = len / BLOCK_SIZE;
        for (uint64_t i = 0; i < full_blocks_count; ++i) {
            uint32_t X[16]; for(int k=0; k<16; ++k) X[k] = ((uint32_t*)(message + i*BLOCK_SIZE))[k];
            cuda_compress_from_X(state, X);
        }
        uint8_t last_block[BLOCK_SIZE] = {0};
        uint64_t remaining = len % BLOCK_SIZE;
        if (remaining > 0) { for (uint64_t i = 0; i < remaining; i++) { last_block[i] = (message + full_blocks_count * BLOCK_SIZE)[i]; } }
        last_block[remaining] = 0x80;
        uint64_t total_bits = len << 3;
        if (remaining < 56) {
            for (int i = 0; i < 8; ++i) { last_block[56 + i] = (uint8_t)((total_bits >> (i * 8)) & 0xFF); }
            uint32_t X[16]; for(int k=0; k<16; ++k) X[k] = ((uint32_t*)last_block)[k];
            cuda_compress_from_X(state, X);
        } else {
            uint32_t X1[16]; for(int k=0; k<16; ++k) X1[k] = ((uint32_t*)last_block)[k];
            cuda_compress_from_X(state, X1);
            for (int i = 0; i < 56; ++i) { last_block[i] = 0; }
            for (int i = 0; i < 8; ++i) { last_block[56 + i] = (uint8_t)((total_bits >> (i * 8)) & 0xFF); }
            uint32_t X2[16]; for(int k=0; k<16; ++k) X2[k] = ((uint32_t*)last_block)[k];
            cuda_compress_from_X(state, X2);
        }
        for (int i = 0; i < 5; i++) { ((uint32_t*)(d_digests + lane_idx * DIGEST_SIZE))[i] = state[i]; }
    }
}
class Ripemd160BatchProcessor {
public:
    Ripemd160BatchProcessor(uint32_t max_lanes, uint64_t max_total_data_size) : max_lanes_(max_lanes), max_data_size_(max_total_data_size), d_messages_(nullptr), d_offsets_(nullptr), d_lengths_(nullptr), d_digests_(nullptr), stream_(0) {
        CUDA_CHECK(cudaMalloc(&d_messages_, max_data_size_)); CUDA_CHECK(cudaMalloc(&d_offsets_, max_lanes_ * sizeof(uint32_t))); CUDA_CHECK(cudaMalloc(&d_lengths_, max_lanes_ * sizeof(uint64_t))); CUDA_CHECK(cudaMalloc(&d_digests_, max_lanes_ * DIGEST_SIZE)); CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    ~Ripemd160BatchProcessor() noexcept { if(stream_) cudaStreamSynchronize(stream_); if(d_messages_) cudaFree(d_messages_); if(d_offsets_) cudaFree(d_offsets_); if(d_lengths_) cudaFree(d_lengths_); if(d_digests_) cudaFree(d_digests_); if(stream_) cudaStreamDestroy(stream_); }
    void process_batch_async(const uint8_t* h_messages, const uint32_t* h_offsets, const uint64_t* h_lengths, uint32_t num_lanes, uint8_t* h_digests) {
        if (num_lanes == 0) return; if (num_lanes > max_lanes_) throw std::length_error("Number of lanes exceeds the pre-allocated capacity.");
        uint64_t current_total_size = 0; if (num_lanes > 0) { current_total_size = (uint64_t)h_offsets[num_lanes-1] + h_lengths[num_lanes-1]; }
        if (current_total_size > max_data_size_) throw std::length_error("Total data size exceeds the pre-allocated capacity.");
        CUDA_CHECK(cudaMemcpyAsync(d_messages_, h_messages, current_total_size, cudaMemcpyHostToDevice, stream_)); CUDA_CHECK(cudaMemcpyAsync(d_offsets_, h_offsets, num_lanes * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_)); CUDA_CHECK(cudaMemcpyAsync(d_lengths_, h_lengths, num_lanes * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_));
        int deviceId; CUDA_CHECK(cudaGetDevice(&deviceId)); cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
        int optimal_block_size, min_grid_size; CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size, ripemd160_kernel_grid_stride, 0, num_lanes));
        int grid_size = (num_lanes + optimal_block_size - 1) / optimal_block_size;
        ripemd160_kernel_grid_stride<<<grid_size, optimal_block_size, 0, stream_>>>(d_messages_, d_offsets_, d_lengths_, num_lanes, d_digests_);
        CUDA_CHECK(cudaMemcpyAsync(h_digests, d_digests_, num_lanes * DIGEST_SIZE, cudaMemcpyDeviceToHost, stream_));
    }
    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }
private:
    uint32_t max_lanes_; uint64_t max_data_size_; uint8_t* d_messages_; uint32_t* d_offsets_; uint64_t* d_lengths_; uint8_t* d_digests_; cudaStream_t stream_;
};
#endif // RIPEMD160_CUH_
