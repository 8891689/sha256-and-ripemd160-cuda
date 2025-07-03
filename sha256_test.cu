// author： https://github.com/8891689
// nvcc sha256_test.cu -o sha256_test -O3 -arch=sm_61
#include "sha256_cuda.cuh"
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>


int main(int argc, char* argv[]) {
    try {
        int deviceId = 0;
        CHECK_CUDA_ERROR(cudaSetDevice(deviceId));
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        std::cout << "Using GPU: " << prop.name << std::endl;
        
        sha256_setup_constants();

        {
            std::cout << "\n--- Verifying SHA-256 for string \"abc\" ---" << std::endl;

            std::vector<BYTE> padded_msg(64, 0); 
            const char* input_str = "abc";
            size_t len = strlen(input_str); 
            memcpy(padded_msg.data(), input_str, len);
            padded_msg[len] = 0x80;
            uint64_t bit_len = len * 8; 
            uint64_t be_bit_len = byte_swap64(bit_len);
            memcpy(&padded_msg[56], &be_bit_len, 8);

            std::vector<WORD> host_block(16);
            for (int i = 0; i < 16; ++i) {
                WORD w;
                memcpy(&w, &padded_msg[i * 4], 4);
                host_block[i] = byte_swap32(w);
            }

            CudaUniquePtr<WORD> d_block;
            CudaUniquePtr<BYTE> d_hash_out;
            CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_block), 64));
            CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_hash_out), 32));

            CHECK_CUDA_ERROR(cudaMemcpy(d_block.get(), host_block.data(), 64, cudaMemcpyHostToDevice));

            sha256_single_hash_kernel<<<1, 1>>>(d_block.get(), d_hash_out.get());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize()); 

            std::vector<BYTE> h_hash(32);
            CHECK_CUDA_ERROR(cudaMemcpy(h_hash.data(), d_hash_out.get(), 32, cudaMemcpyDeviceToHost));

            std::cout << "Input string: \"abc\"" << std::endl;
            std::cout << "Expected hash:  ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" << std::endl;
            std::cout << "Computed hash:  ";
            for (int i = 0; i < 32; i++) {
                printf("%02x", h_hash[i]);
            }
            std::cout << std::endl;
        }

        std::cout << "\n--- SHA-256 CUDA Performance Benchmark (Original Reduction) ---" << std::endl;
        
        uint64_t total_hashes_to_test = 10ULL * 1000 * 1000 * 1000;
        int block_size = 256;
        const int hashes_per_thread = 64;
        
        if (argc > 1) total_hashes_to_test = std::stoull(argv[1]);
        if (argc > 2) block_size = std::stoi(argv[2]);

        std::cout << "Total Hashes to Test: " << total_hashes_to_test << std::endl;
        std::cout << "Block Size: " << block_size << std::endl;
        
        if ((block_size & (block_size - 1)) != 0 && block_size != 0) {
             std::cout << "Warning: Block size " << block_size 
                       << " is not a power of two. Tree reduction requires a power-of-two block size for correctness." << std::endl;
        }
        if (block_size > 1024) {
             std::cerr << "Error: Block size cannot exceed 1024 for this kernel's shared memory." << std::endl;
             return 1;
        }

        uint64_t total_hashes_in_block_processing_unit = 
            static_cast<uint64_t>(block_size) * hashes_per_thread;
        uint64_t total_blocks = (total_hashes_to_test + total_hashes_in_block_processing_unit - 1) 
                                / total_hashes_in_block_processing_unit;

        if (total_blocks > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            std::cerr << "Error: Too many blocks required. Reduce total hashes or increase block size." << std::endl;
            return 1;
        }

        CudaUniquePtr<BYTE> d_benchmark_hashes_out;
        size_t output_size = static_cast<size_t>(total_blocks) * 32;
        CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_benchmark_hashes_out), output_size));

        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        std::cout << "Starting benchmark (single kernel launch)..." << std::endl;
        
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        
        sha256_benchmark_kernel<<<static_cast<int>(total_blocks), block_size>>>(
            0,
            total_hashes_to_test, 
            d_benchmark_hashes_out.get()
        );
        
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        double duration_seconds = milliseconds / 1000.0;
        
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));

        std::cout << "\n\nBenchmark finished in " << duration_seconds << " seconds." << std::endl;
        
        double avg_hashes_per_second = (duration_seconds > 0) ? 
            static_cast<double>(total_hashes_to_test) / duration_seconds : 0.0;
        
        std::cout << "Average Performance: " << std::fixed << std::setprecision(2) 
                  << avg_hashes_per_second / 1e9 << " GHashes/s" << std::endl;
        std::cout << "Average Performance: " << std::fixed << std::setprecision(2) 
                  << avg_hashes_per_second / 1e6 << " MHashes/s" << std::endl;

        if (total_hashes_to_test > 0) {
            std::vector<BYTE> h_first_hash(32);
            CHECK_CUDA_ERROR(cudaMemcpy(h_first_hash.data(), d_benchmark_hashes_out.get(),
                                        32, cudaMemcpyDeviceToHost));
            std::cout << "First computed hash from benchmark: ";
            for (int i = 0; i < 32; i++) {
                printf("%02x", h_first_hash[i]);
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "\nAn error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
