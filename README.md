# sha256-and-ripemd160-cuda

1. Highly Optimized CUDA Library: Focuses on its core technology (CUDA) and quality (optimized).

2. Leveraging Kernel-Level Loops: Focuses on the key techniques used to ensure the authenticity of performance data, which is the essence of our debugging.

3. Automatic Batch Adjustment Technology: Explains its ease of use and intelligence to adapt to GPUs of different performance.

4. Designed for: Indicates its dedicated purpose.

5.Hash Algorithm: Points out the specific algorithm.

6. Accurate and Trusted Extreme Performance Benchmarks: Summarizes the ultimate goal and value of the library - not only fast, but also accurate measurement.

# Compilation and testing

```
nvcc sha256_test.cu -o sha256_test -O3 -arch=sm_61

nvcc ripemd160_test.cu -o ripemd160_test -O3 -arch=sm_61
```

```
./sha256_test
Using GPU: NVIDIA GeForce GT 1030

--- Verifying SHA-256 for string "abc" ---
Input string: "abc"
Expected hash:  ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
Computed hash:  ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad

--- SHA-256 CUDA Performance Benchmark (Original Reduction) ---
Total Hashes to Test: 10000000000
Block Size: 256
Starting benchmark (single kernel launch)...


Benchmark finished in 28.979 seconds.
Average Performance: 0.35 GHashes/s
Average Performance: 345.08 MHashes/s
First computed hash from benchmark: 05a0f3f70da96984759732940161398ebd303a7a17afc1f78eca240d24f11fdf
```

```
./ripemd160_test
Using GPU: NVIDIA GeForce GT 1030 (1990 MB VRAM)

--- Verifying RIPEMD-160 for string "abc" (General Kernel) ---
Expected hash:  8eb208f7e05d987a9b044a8e98c6b087f15a0bfc
Computed hash:  8eb208f7e05d987a9b044a8e98c6b087f15a0bfc

--- RIPEMD-160 CUDA Performance Benchmark (Kernel-level loop for accuracy) ---
Phase 1: Probing with kernel-level loops...
  GPU warmed up.
  Probe stable at 64 iterations. Estimated performance: 560.40 MHashes/s.
Probe finished. Final configuration for benchmark:
  - Parallel Lanes:      59660697
  - Iterations per Lane: 29
  - Total Hashes:        1730160213

Phase 2: Running benchmark...
Starting benchmark. This will take about 3.1 seconds...


Benchmark finished.
Total Hashes Computed: 1730160213
Execution Time:      2.852578 seconds.
Average Performance: 606.53 MHashes/s
Data Throughput:     4.52 GB/s

--- Verifying hash chain result (lane 0) ---
GPU computed hash: fe8e79c54a597a07c24c622a7003ddd48e506004
CPU computing reference hash chain...
CPU computed hash:   fe8e79c54a597a07c24c622a7003ddd48e506004
Verification PASSED!
```
# Please fill in the table according to the compute capability of your graphics card.
-arch=sm_ ?

sm_61 1030 is 61, not 6.1. Please remove the decimal point in the following values.

```
NVIDIA A100	8	RTX A5000	8.6	GeForce RTX 3090 Ti	8.6	GeForce RTX 3080 Ti	8.6
NVIDIA A40	8.6	RTX A4000	8.6	GeForce RTX 3090	8.6	GeForce RTX 3080	8.6
NVIDIA A30	8	RTX A3000	8.6	GeForce RTX 3080 Ti	8.6	GeForce RTX 3070 Ti	8.6
NVIDIA A10	8.6	RTX A2000	8.6	GeForce RTX 3080	8.6	GeForce RTX 3070	8.6
NVIDIA A16	8.6	RTX 5000	7.5	GeForce RTX 3070 Ti	8.6	GeForce RTX 3060	8.6
NVIDIA A2	8.6	RTX 4000	7.5	GeForce RTX 3070	8.6	GeForce RTX 3050 Ti	8.6
NVIDIA T4	7.5	RTX 3000	7.5	Geforce RTX 3060 Ti	8.6	GeForce RTX 3050	8.6
NVIDIA V100	7	T2000	7.5	Geforce RTX 3060	8.6	Geforce RTX 2080	7.5
Tesla P100	6	T1200	7.5	GeForce GTX 1650 Ti	7.5	Geforce RTX 2070	7.5
Tesla P40	6.1	T1000	7.5	NVIDIA TITAN RTX	7.5	Geforce RTX 2060	7.5
Tesla P4	6.1	T600	7.5	Geforce RTX 2080 Ti	7.5	GeForce GTX 1080	6.1
Tesla M60	5.2	T500	7.5	Geforce RTX 2080	7.5	GeForce GTX 1070	6.1
Tesla M40	5.2	P620	6.1	Geforce RTX 2070	7.5	GeForce GTX 1060	6.1
Tesla K80	3.7	P520	6.1	Geforce RTX 2060	7.5	GeForce GTX 980	5.2
Tesla K40	3.5	Quadro P5200	6.1	NVIDIA TITAN V	7	GeForce GTX 980M	5.2
Tesla K20	3.5	Quadro P4200	6.1	NVIDIA TITAN Xp	6.1	GeForce GTX 970M	5.2
Tesla K10	3	Quadro P3200	6.1	NVIDIA TITAN X	6.1	GeForce GTX 965M	5.2
RTX A6000	8.6	Quadro P5000	6.1	GeForce GTX 1080 Ti	6.1	GeForce GTX 960M	5
RTX A5000	8.6	Quadro P4000	6.1	GeForce GTX 1080	6.1	GeForce GTX 950M	5
RTX A4000	8.6	Quadro P3000	6.1	GeForce GTX 1070 Ti	6.1	GeForce 940M	5
T1000	7.5	Quadro P2000	6.1	GeForce GTX 1070	6.1	GeForce 930M	5
T600	7.5	Quadro P1000	6.1	GeForce GTX 1060	6.1	GeForce 920M	3.5
T400	7.5	Quadro P600	6.1	GeForce GTX 1050	6.1	GeForce 910M	5.2
Quadro RTX 8000	7.5	Quadro P500	6.1	GeForce GTX TITAN X	5.2	GeForce GTX 880M	3
Quadro RTX 6000	7.5	Quadro M5500M	5.2	GeForce GTX TITAN Z	3.5	GeForce GTX 870M	3
Quadro RTX 5000	7.5	Quadro M2200	5.2	GeForce GTX TITAN Black	3.5	GeForce GTX 860M	3.0/5.0
Quadro RTX 4000	7.5	Quadro M1200	5	GeForce GTX TITAN	3.5	GeForce GTX 850M	5
Quadro GV100	7	Quadro M620	5.2	GeForce GTX 980 Ti	5.2	GeForce 840M	5
Quadro GP100	6	Quadro M520	5	GeForce GTX 980	5.2	GeForce 830M	5
Quadro P6000	6.1	Quadro K6000M	3	GeForce GTX 970	5.2	GeForce 820M	2.1
Quadro P5000	6.1	Quadro K5200M	3	GeForce GTX 960	5.2	GeForce 800M	2.1
Quadro P4000	6.1	Quadro K5100M	3	GeForce GTX 950	5.2	GeForce GTX 780M	3
Quadro P2200	6.1	Quadro M5000M	5	GeForce GTX 780 Ti	3.5	GeForce GTX 770M	3
Quadro P2000	6.1	Quadro K500M	3	GeForce GTX 780	3.5	GeForce GTX 765M	3
Quadro P1000	6.1	Quadro K4200M	3	GeForce GTX 770	3	GeForce GTX 760M	3
Quadro P620	6.1	Quadro K4100M	3	GeForce GTX 760	3	GeForce GTX 680MX	3
Quadro P600	6.1	Quadro M4000M	5	GeForce GTX 750 Ti	5	GeForce GTX 680M	3
Quadro P400	6.1	Quadro K3100M	3	GeForce GTX 750	5	GeForce GTX 675MX	3
Quadro M6000 24GB	5.2	GeForce GT 730 DDR3,128bit	2.1	GeForce GTX 690	3	GeForce GTX 675M	2.1
Quadro M6000	5.2	Quadro M3000M	5	GeForce GTX 680	3	GeForce GTX 670MX	3
Quadro 410	3	Quadro K2200M	3	GeForce GTX 670	3	GeForce GTX 670M	2.1
Quadro K6000	3.5	Quadro K2100M	3	GeForce GTX 660 Ti	3	GeForce GTX 660M	3
Quadro M5000	5.2	Quadro M2000M	5	GeForce GTX 660	3	GeForce GT 755M	3
Quadro K5200	3.5	Quadro K1100M	3	GeForce GTX 650 Ti BOOST	3	GeForce GT 750M	3
Quadro K5000	3	Quadro M1000M	5	GeForce GTX 650 Ti	3	GeForce GT 650M	3
Quadro M4000	5.2	Quadro K620M	5	GeForce GTX 650	3	GeForce GT 745M	3
Quadro K4200	3	Quadro K610M	3.5	GeForce GTX 560 Ti	2.1	GeForce GT 645M	3
Quadro K4000	3	Quadro M600M	5	GeForce GTX 550 Ti	2.1	GeForce GT 740M	3
Quadro M2000	5.2	Quadro K510M	3.5	GeForce GTX 460	2.1	GeForce GT 730M	3
Quadro K2200	5	Quadro M500M	5	GeForce GTS 450	2.1	GeForce GT 640M	3
Quadro K2000	3	GeForce 705M	2.1	GeForce GTS 450*	2.1	GeForce GT 640M LE	3
Quadro K2000D	3	NVIDIA NVS 810	5	GeForce GTX 590	2	GeForce GT 735M	3
Quadro K1200	5	NVIDIA NVS 510	3	GeForce GTX 580	2	GeForce GT 635M	2.1
Quadro K620	5	NVIDIA NVS 315	2.1	GeForce GTX 570	2	GeForce GT 730M	3
Quadro K600	3	NVIDIA NVS 310	2.1	GeForce GTX 480	2	GeForce GT 630M	2.1
Quadro K420	3	Quadro Plex 7000	2	GeForce GTX 470	2	GeForce GT 625M	2.1
GeForce GT 730	3.5	GeForce 710M	2.1	GeForce GTX 465	2	GeForce GT 720M	2.1
GeForce GT 720	3.5	GeForce 610M	2.1	GeForce GT 740	3	GeForce GT 620M	2.1
```
## Sponsorship

If this project has been helpful to you, please consider sponsoring. It is the greatest support for me, and I am deeply grateful. Thank you.
```
BTC:  bc1qt3nh2e6gjsfkfacnkglt5uqghzvlrr6jahyj2k
ETH:  0xD6503e5994bF46052338a9286Bc43bC1c3811Fa1
DOGE: DTszb9cPALbG9ESNJMFJt4ECqWGRCgucky
TRX:  TAHUmjyzg7B3Nndv264zWYUhQ9HUmX4Xu4
 ``` 
