/*MIT License
Copyright (c) 2025 8891689
//author： https://github.com/8891689
nvcc ripemd160_test.cu -o test -O3 -arch=sm_61
*/
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <iomanip>
#include "ripemd160.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ===================================================================
// 內核內循環進行最準確的測試
// ===================================================================
int main(int argc, char* argv[]) {
    try {
        int deviceId = 0;
        CUDA_CHECK(cudaSetDevice(deviceId));
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        std::cout << "Using GPU: " << prop.name << " (" 
                  << prop.totalGlobalMem / (1024 * 1024) << " MB VRAM)" << std::endl;
        
        // --- Correctness Verification ---
        {
            std::cout << "\n--- Verifying RIPEMD-160 for string \"abc\" (General Kernel) ---" << std::endl;
            const char* input_str = "abc";
            std::vector<uint8_t> h_hash(DIGEST_SIZE);
            Ripemd160BatchProcessor single_shot_proc(1, strlen(input_str));
            uint32_t h_offset_abc[] = {0};
            uint64_t h_length_abc[] = {(uint64_t)strlen(input_str)};
            single_shot_proc.process_batch_async(reinterpret_cast<const uint8_t*>(input_str), h_offset_abc, h_length_abc, 1, h_hash.data());
            single_shot_proc.synchronize();
            std::cout << "Expected hash:  8eb208f7e05d987a9b044a8e98c6b087f15a0bfc" << std::endl;
            std::cout << "Computed hash:  ";
            for(uint8_t c : h_hash) printf("%02x", c);
            std::cout << std::endl;
        }

        // --- Performance Benchmark ---
        std::cout << "\n--- RIPEMD-160 CUDA Performance Benchmark (Kernel-level loop for accuracy) ---" << std::endl;
        
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // 1. 精準探測階段 (Accurate Probing)
        // ===============================================
        std::cout << "Phase 1: Probing with kernel-level loops..." << std::endl;
        
        double hashes_per_sec = 0;
        uint32_t probe_lanes = 1024 * 1024;
        uint32_t probe_iterations = 1;

        uint64_t* d_probe_messages;
        uint8_t* d_probe_digests;
        CUDA_CHECK(cudaMalloc(&d_probe_messages, probe_lanes * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_probe_digests, probe_lanes * DIGEST_SIZE));

        int block_size = 256;
        int grid_size = (probe_lanes + block_size - 1) / block_size;
        ripemd160_fixed8_kernel<<<grid_size, block_size, 0, stream>>>(d_probe_messages, d_probe_digests, probe_lanes, 1);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cout << "  GPU warmed up." << std::endl;

        while (true) {
            cudaEvent_t probe_start, probe_stop;
            CUDA_CHECK(cudaEventCreate(&probe_start));
            CUDA_CHECK(cudaEventCreate(&probe_stop));
            CUDA_CHECK(cudaEventRecord(probe_start, stream));
            ripemd160_fixed8_kernel<<<grid_size, block_size, 0, stream>>>(d_probe_messages, d_probe_digests, probe_lanes, probe_iterations);
            CUDA_CHECK(cudaEventRecord(probe_stop, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float probe_milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&probe_milliseconds, probe_start, probe_stop));
            CUDA_CHECK(cudaEventDestroy(probe_start));
            CUDA_CHECK(cudaEventDestroy(probe_stop));
            if (probe_milliseconds > 100.0f) {
                uint64_t total_hashes = (uint64_t)probe_lanes * probe_iterations;
                hashes_per_sec = (probe_milliseconds > 0) ? (total_hashes / (probe_milliseconds / 1000.0)) : 0;
                std::cout << "  Probe stable at " << probe_iterations << " iterations. Estimated performance: " 
                          << std::fixed << std::setprecision(2) << hashes_per_sec / 1e6 << " MHashes/s." << std::endl;
                break;
            }
            if (probe_iterations > (1u << 28)) {
                 hashes_per_sec = 0;
                 std::cout << "  Probe iteration limit reached, could not get a stable timing." << std::endl;
                 break;
            }
            probe_iterations *= 2;
        }
        
        CUDA_CHECK(cudaFree(d_probe_messages));
        CUDA_CHECK(cudaFree(d_probe_digests));
        
        if (hashes_per_sec == 0) {
            throw std::runtime_error("Failed to get a valid performance estimate in Phase 1.");
        }

        // 2. 正式計時階段
        // ===============================================
        double target_duration_sec = 3.0;
        uint64_t total_hashes_to_compute = static_cast<uint64_t>(hashes_per_sec * target_duration_sec);
        
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        size_t bytes_per_lane = sizeof(uint64_t) + DIGEST_SIZE;
        size_t safety_margin = std::max((size_t)(free_mem * 0.1), (size_t)50 * 1024 * 1024);
        size_t usable_mem = (free_mem > safety_margin) ? (free_mem - safety_margin) : 0;
        uint32_t final_lanes = (bytes_per_lane > 0) ? (usable_mem / bytes_per_lane) : 0;

        if (final_lanes == 0) throw std::runtime_error("Not enough VRAM for final benchmark.");
        
        uint32_t final_iterations = (total_hashes_to_compute + final_lanes - 1) / final_lanes;
        if (final_iterations == 0) final_iterations = 1;
        
        uint64_t final_total_hashes = (uint64_t)final_lanes * final_iterations;

        std::cout << "Probe finished. Final configuration for benchmark:" << std::endl;
        std::cout << "  - Parallel Lanes:      " << final_lanes << std::endl;
        std::cout << "  - Iterations per Lane: " << final_iterations << std::endl;
        std::cout << "  - Total Hashes:        " << final_total_hashes << std::endl;
        
        uint64_t* d_messages;
        uint8_t* d_digests;

        if (final_total_hashes > 0) {
            std::cout << "\nPhase 2: Running benchmark..." << std::endl;
            
            CUDA_CHECK(cudaMalloc(&d_messages, final_lanes * sizeof(uint64_t)));
            CUDA_CHECK(cudaMalloc(&d_digests, final_lanes * DIGEST_SIZE));

            std::vector<uint64_t> h_messages(final_lanes);
            for(uint32_t i = 0; i < final_lanes; ++i) h_messages[i] = (uint64_t)i * final_iterations; 
            CUDA_CHECK(cudaMemcpy(d_messages, h_messages.data(), final_lanes * sizeof(uint64_t), cudaMemcpyHostToDevice));

            cudaEvent_t bench_start, bench_stop;
            CUDA_CHECK(cudaEventCreate(&bench_start));
            CUDA_CHECK(cudaEventCreate(&bench_stop));

            double estimated_time = (hashes_per_sec > 0) ? final_total_hashes / hashes_per_sec : 0;
            std::cout << "Starting benchmark. This will take about " << std::fixed << std::setprecision(1) << estimated_time << " seconds..." << std::endl;
            
            int final_grid_size = (final_lanes + block_size - 1) / block_size;
            
            CUDA_CHECK(cudaEventRecord(bench_start, stream));
            ripemd160_fixed8_kernel<<<final_grid_size, block_size, 0, stream>>>(d_messages, d_digests, final_lanes, final_iterations);
            CUDA_CHECK(cudaEventRecord(bench_stop, stream));
            
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            float final_milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&final_milliseconds, bench_start, bench_stop));
            
            CUDA_CHECK(cudaEventDestroy(bench_start));
            CUDA_CHECK(cudaEventDestroy(bench_stop));

            double duration_seconds = final_milliseconds / 1000.0;
            double avg_hashes_per_second = (duration_seconds > 0) ? (double)final_total_hashes / duration_seconds : 0.0;
            const double INPUT_MESSAGE_SIZE = 8.0; 
            double total_bytes_processed = (double)final_total_hashes * INPUT_MESSAGE_SIZE;
            double throughput_gb_per_sec = (duration_seconds > 0) ? (total_bytes_processed / (1024.0 * 1024.0 * 1024.0)) / duration_seconds : 0.0;

            std::cout << "\n\nBenchmark finished." << std::endl;
            std::cout << "Total Hashes Computed: " << final_total_hashes << std::endl;
            std::cout << "Execution Time:      " << std::fixed << std::setprecision(6) << duration_seconds << " seconds." << std::endl;
            std::cout << "Average Performance: " << std::fixed << std::setprecision(2) << avg_hashes_per_second / 1e6 << " MHashes/s" << std::endl;
            std::cout << "Data Throughput:     " << std::fixed << std::setprecision(2) << throughput_gb_per_sec << " GB/s" << std::endl;

            // --- Verification ---
            std::cout << "\n--- Verifying hash chain result (lane 0) ---" << std::endl;
    
            std::vector<uint8_t> gpu_result(DIGEST_SIZE);
            CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_digests, DIGEST_SIZE, cudaMemcpyDeviceToHost));
            
            std::cout << "GPU computed hash: ";
            for(uint8_t c : gpu_result) printf("%02x", c);
            std::cout << std::endl;

            std::cout << "CPU computing reference hash chain..." << std::endl;
            std::vector<uint8_t> cpu_hash_buf(DIGEST_SIZE);
            uint64_t initial_nonce = (uint64_t)0 * final_iterations;

            Ripemd160BatchProcessor cpu_proc(1, 20);
            uint32_t offset[] = {0};
            uint64_t length_u64[] = {sizeof(initial_nonce)};
            cpu_proc.process_batch_async(reinterpret_cast<const uint8_t*>(&initial_nonce), offset, length_u64, 1, cpu_hash_buf.data());
            cpu_proc.synchronize();

            uint64_t length_20b[] = {20};
            for (uint32_t iter = 1; iter < final_iterations; ++iter) {
                cpu_proc.process_batch_async(cpu_hash_buf.data(), offset, length_20b, 1, cpu_hash_buf.data());
                cpu_proc.synchronize();
            }

            std::cout << "CPU computed hash:   ";
            for(uint8_t c : cpu_hash_buf) printf("%02x", c);
            std::cout << std::endl;

            if (memcmp(gpu_result.data(), cpu_hash_buf.data(), DIGEST_SIZE) == 0) {
                std::cout << "Verification PASSED!" << std::endl;
            } else {
                std::cout << "Verification FAILED!" << std::endl;
            }
            
            CUDA_CHECK(cudaFree(d_messages));
            CUDA_CHECK(cudaFree(d_digests));
        }

        CUDA_CHECK(cudaStreamDestroy(stream));

    } catch (const std::exception& e) {
        std::cerr << "\nAn error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
