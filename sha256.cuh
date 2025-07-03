/* MIT License
   Copyright (c) 2025 8891689
   author： https://github.com/8891689
*/
//sha256.cuh

#ifndef SHA256_CUH
#define SHA256_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <memory>

typedef unsigned char BYTE;
typedef uint32_t WORD;

struct CudaDeleter { void operator()(void* p) const { if (p) cudaFree(p); } };
template<typename T> using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

static void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)
                  << " in " << file << " at line " << line << std::endl;
        throw std::runtime_error("CUDA operation failed");
    }
}
#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

__constant__ WORD dev_k[64];

__device__ __forceinline__ WORD ROTRIGHT(WORD a, WORD b) { return (((a) >> (b)) | ((a) << (32 - (b)))); }
__device__ __forceinline__ WORD CH(WORD x, WORD y, WORD z) { return (((x) & (y)) ^ (~(x) & (z))); }
__device__ __forceinline__ WORD MAJ(WORD x, WORD y, WORD z) { return (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z))); }
__device__ __forceinline__ WORD EP0(WORD x) { return (ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22)); }
__device__ __forceinline__ WORD EP1(WORD x) { return (ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25)); }
__device__ __forceinline__ WORD SIG0(WORD x) { return (ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ ((x) >> 3)); }
__device__ __forceinline__ WORD SIG1(WORD x) { return (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ ((x) >> 10)); }

static const WORD host_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __inline__ void sha256_init(WORD state[8]) {
    state[0] = 0x6a09e667; state[1] = 0xbb67ae85; state[2] = 0x3c6ef372; state[3] = 0xa54ff53a;
    state[4] = 0x510e527f; state[5] = 0x9b05688c; state[6] = 0x1f83d9ab; state[7] = 0x5be0cd19;
}

__device__ __inline__ void hash_state_to_bytes_unrolled(const WORD st[8], BYTE hash_bytes[32]) {
    hash_bytes[0] = static_cast<BYTE>(st[0] >> 24); hash_bytes[1] = static_cast<BYTE>(st[0] >> 16);
    hash_bytes[2] = static_cast<BYTE>(st[0] >> 8);  hash_bytes[3] = static_cast<BYTE>(st[0]);
    hash_bytes[4] = static_cast<BYTE>(st[1] >> 24); hash_bytes[5] = static_cast<BYTE>(st[1] >> 16);
    hash_bytes[6] = static_cast<BYTE>(st[1] >> 8);  hash_bytes[7] = static_cast<BYTE>(st[1]);
    hash_bytes[8] = static_cast<BYTE>(st[2] >> 24); hash_bytes[9] = static_cast<BYTE>(st[2] >> 16);
    hash_bytes[10] = static_cast<BYTE>(st[2] >> 8); hash_bytes[11] = static_cast<BYTE>(st[2]);
    hash_bytes[12] = static_cast<BYTE>(st[3] >> 24); hash_bytes[13] = static_cast<BYTE>(st[3] >> 16);
    hash_bytes[14] = static_cast<BYTE>(st[3] >> 8); hash_bytes[15] = static_cast<BYTE>(st[3]);
    hash_bytes[16] = static_cast<BYTE>(st[4] >> 24); hash_bytes[17] = static_cast<BYTE>(st[4] >> 16);
    hash_bytes[18] = static_cast<BYTE>(st[4] >> 8); hash_bytes[19] = static_cast<BYTE>(st[4]);
    hash_bytes[20] = static_cast<BYTE>(st[5] >> 24); hash_bytes[21] = static_cast<BYTE>(st[5] >> 16);
    hash_bytes[22] = static_cast<BYTE>(st[5] >> 8); hash_bytes[23] = static_cast<BYTE>(st[5]);
    hash_bytes[24] = static_cast<BYTE>(st[6] >> 24); hash_bytes[25] = static_cast<BYTE>(st[6] >> 16);
    hash_bytes[26] = static_cast<BYTE>(st[6] >> 8); hash_bytes[27] = static_cast<BYTE>(st[6]);
    hash_bytes[28] = static_cast<BYTE>(st[7] >> 24); hash_bytes[29] = static_cast<BYTE>(st[7] >> 16);
    hash_bytes[30] = static_cast<BYTE>(st[7] >> 8); hash_bytes[31] = static_cast<BYTE>(st[7]);
}

__device__ __forceinline__ void sha256_transform_optimized(WORD state[8], const WORD* __restrict__ block)
{
    WORD W[16];
    WORD a, b, c, d, e, f, g, h;

    W[ 0] = block[ 0];
    W[ 1] = block[ 1];
    W[ 2] = block[ 2];
    W[ 3] = block[ 3];
    W[ 4] = block[ 4];
    W[ 5] = block[ 5];
    W[ 6] = block[ 6];
    W[ 7] = block[ 7];
    W[ 8] = block[ 8];
    W[ 9] = block[ 9];
    W[10] = block[10];
    W[11] = block[11];
    W[12] = block[12];
    W[13] = block[13];
    W[14] = block[14];
    W[15] = block[15];

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    #define SHA256_ROUND(a,b,c,d,e,f,g,h,k,w) \
        { \
            WORD T1 = h + EP1(e) + CH(e, f, g) + k + w; \
            WORD T2 = EP0(a) + MAJ(a, b, c); \
            d += T1; \
            h = T1 + T2; \
        }

    SHA256_ROUND(a,b,c,d,e,f,g,h, dev_k[0], W[0]); 
    SHA256_ROUND(h,a,b,c,d,e,f,g, dev_k[1], W[1]); 
    SHA256_ROUND(g,h,a,b,c,d,e,f, dev_k[2], W[2]); 
    SHA256_ROUND(f,g,h,a,b,c,d,e, dev_k[3], W[3]); 
    SHA256_ROUND(e,f,g,h,a,b,c,d, dev_k[4], W[4]); 
    SHA256_ROUND(d,e,f,g,h,a,b,c, dev_k[5], W[5]);
    SHA256_ROUND(c,d,e,f,g,h,a,b, dev_k[6], W[6]); 
    SHA256_ROUND(b,c,d,e,f,g,h,a, dev_k[7], W[7]); 
    SHA256_ROUND(a,b,c,d,e,f,g,h, dev_k[8], W[8]); 
    SHA256_ROUND(h,a,b,c,d,e,f,g, dev_k[9], W[9]); 
    SHA256_ROUND(g,h,a,b,c,d,e,f, dev_k[10], W[10]); 
    SHA256_ROUND(f,g,h,a,b,c,d,e, dev_k[11], W[11]); 
    SHA256_ROUND(e,f,g,h,a,b,c,d, dev_k[12], W[12]); 
    SHA256_ROUND(d,e,f,g,h,a,b,c, dev_k[13], W[13]); 
    SHA256_ROUND(c,d,e,f,g,h,a,b, dev_k[14], W[14]); 
    SHA256_ROUND(b,c,d,e,f,g,h,a, dev_k[15], W[15]); 
    SHA256_ROUND(a,b,c,d,e,f,g,h, dev_k[16], (W[0] = SIG1(W[14]) + W[9] + SIG0(W[1]) + W[0])); 
    SHA256_ROUND(h,a,b,c,d,e,f,g, dev_k[17], (W[1] = SIG1(W[15]) + W[10] + SIG0(W[2]) + W[1])); 
    SHA256_ROUND(g,h,a,b,c,d,e,f, dev_k[18], (W[2] = SIG1(W[0]) + W[11] + SIG0(W[3]) + W[2])); 
    SHA256_ROUND(f,g,h,a,b,c,d,e, dev_k[19], (W[3] = SIG1(W[1]) + W[12] + SIG0(W[4]) + W[3])); 
    SHA256_ROUND(e,f,g,h,a,b,c,d, dev_k[20], (W[4] = SIG1(W[2]) + W[13] + SIG0(W[5]) + W[4])); 
    SHA256_ROUND(d,e,f,g,h,a,b,c, dev_k[21], (W[5] = SIG1(W[3]) + W[14] + SIG0(W[6]) + W[5])); 
    SHA256_ROUND(c,d,e,f,g,h,a,b, dev_k[22], (W[6] = SIG1(W[4]) + W[15] + SIG0(W[7]) + W[6])); 
    SHA256_ROUND(b,c,d,e,f,g,h,a, dev_k[23], (W[7] = SIG1(W[5]) + W[0] + SIG0(W[8]) + W[7])); 
    SHA256_ROUND(a,b,c,d,e,f,g,h, dev_k[24], (W[8] = SIG1(W[6]) + W[1] + SIG0(W[9]) + W[8])); 
    SHA256_ROUND(h,a,b,c,d,e,f,g, dev_k[25], (W[9] = SIG1(W[7]) + W[2] + SIG0(W[10]) + W[9])); 
    SHA256_ROUND(g,h,a,b,c,d,e,f, dev_k[26], (W[10] = SIG1(W[8]) + W[3] + SIG0(W[11]) + W[10])); 
    SHA256_ROUND(f,g,h,a,b,c,d,e, dev_k[27], (W[11] = SIG1(W[9]) + W[4] + SIG0(W[12]) + W[11])); 
    SHA256_ROUND(e,f,g,h,a,b,c,d, dev_k[28], (W[12] = SIG1(W[10]) + W[5] + SIG0(W[13]) + W[12])); 
    SHA256_ROUND(d,e,f,g,h,a,b,c, dev_k[29], (W[13] = SIG1(W[11]) + W[6] + SIG0(W[14]) + W[13])); 
    SHA256_ROUND(c,d,e,f,g,h,a,b, dev_k[30], (W[14] = SIG1(W[12]) + W[7] + SIG0(W[15]) + W[14])); 
    SHA256_ROUND(b,c,d,e,f,g,h,a, dev_k[31], (W[15] = SIG1(W[13]) + W[8] + SIG0(W[0]) + W[15])); 
    SHA256_ROUND(a,b,c,d,e,f,g,h, dev_k[32], (W[0] = SIG1(W[14]) + W[9] + SIG0(W[1]) + W[0])); 
    SHA256_ROUND(h,a,b,c,d,e,f,g, dev_k[33], (W[1] = SIG1(W[15]) + W[10] + SIG0(W[2]) + W[1]));
    SHA256_ROUND(g,h,a,b,c,d,e,f, dev_k[34], (W[2] = SIG1(W[0]) + W[11] + SIG0(W[3]) + W[2])); 
    SHA256_ROUND(f,g,h,a,b,c,d,e, dev_k[35], (W[3] = SIG1(W[1]) + W[12] + SIG0(W[4]) + W[3])); 
    SHA256_ROUND(e,f,g,h,a,b,c,d, dev_k[36], (W[4] = SIG1(W[2]) + W[13] + SIG0(W[5]) + W[4])); 
    SHA256_ROUND(d,e,f,g,h,a,b,c, dev_k[37], (W[5] = SIG1(W[3]) + W[14] + SIG0(W[6]) + W[5])); 
    SHA256_ROUND(c,d,e,f,g,h,a,b, dev_k[38], (W[6] = SIG1(W[4]) + W[15] + SIG0(W[7]) + W[6])); 
    SHA256_ROUND(b,c,d,e,f,g,h,a, dev_k[39], (W[7] = SIG1(W[5]) + W[0] + SIG0(W[8]) + W[7])); 
    SHA256_ROUND(a,b,c,d,e,f,g,h, dev_k[40], (W[8] = SIG1(W[6]) + W[1] + SIG0(W[9]) + W[8])); 
    SHA256_ROUND(h,a,b,c,d,e,f,g, dev_k[41], (W[9] = SIG1(W[7]) + W[2] + SIG0(W[10]) + W[9])); 
    SHA256_ROUND(g,h,a,b,c,d,e,f, dev_k[42], (W[10] = SIG1(W[8]) + W[3] + SIG0(W[11]) + W[10])); 
    SHA256_ROUND(f,g,h,a,b,c,d,e, dev_k[43], (W[11] = SIG1(W[9]) + W[4] + SIG0(W[12]) + W[11])); 
    SHA256_ROUND(e,f,g,h,a,b,c,d, dev_k[44], (W[12] = SIG1(W[10]) + W[5] + SIG0(W[13]) + W[12])); 
    SHA256_ROUND(d,e,f,g,h,a,b,c, dev_k[45], (W[13] = SIG1(W[11]) + W[6] + SIG0(W[14]) + W[13])); 
    SHA256_ROUND(c,d,e,f,g,h,a,b, dev_k[46], (W[14] = SIG1(W[12]) + W[7] + SIG0(W[15]) + W[14])); 
    SHA256_ROUND(b,c,d,e,f,g,h,a, dev_k[47], (W[15] = SIG1(W[13]) + W[8] + SIG0(W[0]) + W[15])); 
    SHA256_ROUND(a,b,c,d,e,f,g,h, dev_k[48], (W[0] = SIG1(W[14]) + W[9] + SIG0(W[1]) + W[0])); 
    SHA256_ROUND(h,a,b,c,d,e,f,g, dev_k[49], (W[1] = SIG1(W[15]) + W[10] + SIG0(W[2]) + W[1])); 
    SHA256_ROUND(g,h,a,b,c,d,e,f, dev_k[50], (W[2] = SIG1(W[0]) + W[11] + SIG0(W[3]) + W[2])); 
    SHA256_ROUND(f,g,h,a,b,c,d,e, dev_k[51], (W[3] = SIG1(W[1]) + W[12] + SIG0(W[4]) + W[3])); 
    SHA256_ROUND(e,f,g,h,a,b,c,d, dev_k[52], (W[4] = SIG1(W[2]) + W[13] + SIG0(W[5]) + W[4])); 
    SHA256_ROUND(d,e,f,g,h,a,b,c, dev_k[53], (W[5] = SIG1(W[3]) + W[14] + SIG0(W[6]) + W[5])); 
    SHA256_ROUND(c,d,e,f,g,h,a,b, dev_k[54], (W[6] = SIG1(W[4]) + W[15] + SIG0(W[7]) + W[6])); 
    SHA256_ROUND(b,c,d,e,f,g,h,a, dev_k[55], (W[7] = SIG1(W[5]) + W[0] + SIG0(W[8]) + W[7])); 
    SHA256_ROUND(a,b,c,d,e,f,g,h, dev_k[56], (W[8] = SIG1(W[6]) + W[1] + SIG0(W[9]) + W[8])); 
    SHA256_ROUND(h,a,b,c,d,e,f,g, dev_k[57], (W[9] = SIG1(W[7]) + W[2] + SIG0(W[10]) + W[9])); 
    SHA256_ROUND(g,h,a,b,c,d,e,f, dev_k[58], (W[10] = SIG1(W[8]) + W[3] + SIG0(W[11]) + W[10])); 
    SHA256_ROUND(f,g,h,a,b,c,d,e, dev_k[59], (W[11] = SIG1(W[9]) + W[4] + SIG0(W[12]) + W[11])); 
    SHA256_ROUND(e,f,g,h,a,b,c,d, dev_k[60], (W[12] = SIG1(W[10]) + W[5] + SIG0(W[13]) + W[12])); 
    SHA256_ROUND(d,e,f,g,h,a,b,c, dev_k[61], (W[13] = SIG1(W[11]) + W[6] + SIG0(W[14]) + W[13])); 
    SHA256_ROUND(c,d,e,f,g,h,a,b, dev_k[62], (W[14] = SIG1(W[12]) + W[7] + SIG0(W[15]) + W[14])); 
    SHA256_ROUND(b,c,d,e,f,g,h,a, dev_k[63], (W[15] = SIG1(W[13]) + W[8] + SIG0(W[0]) + W[15])); 

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;

    #undef SHA256_ROUND
}

void sha256_setup_constants() {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k)));
}

__host__ __device__ __forceinline__ uint64_t byte_swap64(uint64_t x) {
    return ( (x & 0x00000000000000ffULL) << 56 ) |
           ( (x & 0x000000000000ff00ULL) << 40 ) |
           ( (x & 0x0000000000ff0000ULL) << 24 ) |
           ( (x & 0x00000000ff000000ULL) << 8  ) |
           ( (x & 0x000000ff00000000ULL) >> 8  ) |
           ( (x & 0x0000ff0000000000ULL) >> 24 ) |
           ( (x & 0x00ff000000000000ULL) >> 40 ) |
           ( (x & 0xff00000000000000ULL) >> 56 );
}

__host__ __device__ __forceinline__ uint32_t byte_swap32(uint32_t x) {
    return ((x >> 24) & 0xff) |
           ((x << 8) & 0xff0000) |
           ((x >> 8) & 0xff00) |
           ((x << 24) & 0xff000000);
}
__global__ void sha256_benchmark_kernel(
    uint64_t batch_offset,
    uint64_t num_hashes_in_batch,
    BYTE* __restrict__ d_hashes_out)
{
    const int hashes_per_thread = 64;
    uint64_t chunk_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t start_key_idx = chunk_id * hashes_per_thread;

    if (start_key_idx >= batch_offset + num_hashes_in_batch) {
        return;
    }

    WORD state[8];
    WORD block[16];
    uint32_t thread_xor_accumulator[8] = {0,0,0,0,0,0,0,0};
    uint64_t actual_hashes = min(static_cast<uint64_t>(hashes_per_thread),
                                 (batch_offset + num_hashes_in_batch - start_key_idx));

    for (int i = 0; i < actual_hashes; ++i) {
        uint64_t current_key = start_key_idx + i;
        uint64_t be_key = byte_swap64(current_key);
        reinterpret_cast<uint64_t*>(block)[0] = be_key;
        #pragma unroll
        for (int j = 2; j < 14; ++j) {
            block[j] = 0;
        }
        block[14] = 0x80000000;
        block[15] = 0x00000040; 

        sha256_init(state);
        sha256_transform_optimized(state, block);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            thread_xor_accumulator[k] ^= state[k];
        }
    }

    __shared__ uint32_t s_all_thread_results[1024][8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        s_all_thread_results[threadIdx.x][k] = thread_xor_accumulator[k];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {

            s_all_thread_results[threadIdx.x][0] ^= s_all_thread_results[threadIdx.x + s][0];
            s_all_thread_results[threadIdx.x][1] ^= s_all_thread_results[threadIdx.x + s][1];
            s_all_thread_results[threadIdx.x][2] ^= s_all_thread_results[threadIdx.x + s][2];
            s_all_thread_results[threadIdx.x][3] ^= s_all_thread_results[threadIdx.x + s][3];
            s_all_thread_results[threadIdx.x][4] ^= s_all_thread_results[threadIdx.x + s][4];
            s_all_thread_results[threadIdx.x][5] ^= s_all_thread_results[threadIdx.x + s][5];
            s_all_thread_results[threadIdx.x][6] ^= s_all_thread_results[threadIdx.x + s][6];
            s_all_thread_results[threadIdx.x][7] ^= s_all_thread_results[threadIdx.x + s][7];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint32_t* dest = reinterpret_cast<uint32_t*>(d_hashes_out + blockIdx.x * 32);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            dest[k] = s_all_thread_results[0][k];
        }
    }
}

__global__ void sha256_single_hash_kernel(const WORD* d_block, BYTE* d_hash_out) {

    WORD state[8];
    
    sha256_init(state);

    sha256_transform_optimized(state, d_block);

    hash_state_to_bytes_unrolled(state, d_hash_out);
}


void sha256_setup_constants();

#endif // SHA256_CUH
