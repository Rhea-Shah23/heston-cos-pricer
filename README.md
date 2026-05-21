**High-performance European options volatility surface pricing engine built in C++, using the Fourier-COS method under the Heston stochastic volatility model. Optimized from a naive baseline to a fully vectorized, multithreaded implementation achieving an 85x end-to-end speedup.**

---

## Overview

This project implements the COS method (Fang & Oosterlee, 2008) for pricing entire implied volatility surfaces — grids of thousands of (strike, maturity) pairs — under the Heston (1993) stochastic volatility model. The Heston model admits a known analytical characteristic function, which the COS method exploits via a truncated Fourier cosine series expansion to recover option prices without requiring the risk-neutral density in closed form.

The engineering objective is not merely a correct implementation, but a progressively optimized one: each optimization phase is isolated, profiled, and benchmarked independently. The result is a documented performance engineering case study, not just a pricing library.

**Pricing error vs. QuantLib Heston engine:** RMSE < 1e-4 across a 30×10 strike-maturity surface.

---

## Performance Summary

| Optimization Phase | Cumulative Speedup | Technique |
|---|---|---|
| Baseline (naive) | 1x | Single-threaded, scalar, AoS layout |
| Phase 3 — Computation reuse | 7.2x | Characteristic function shared across strikes |
| Phase 4 — SIMD vectorization | 22.3x | AVX2 / AVX-512, 64-byte aligned SoA layout |
| Phase 5 — Multithreading | 85x | Thread pool, maturity-level parallelism |

> Benchmarked on: Intel Core i9-13900K, 32GB DDR5, GCC 13.2 with `-O3 -march=native -ffast-math`. Full surface: 300 options (30 strikes × 10 maturities).

---

## Mathematical Background

### The Heston Model

The asset price follows:

```
dS_t = μ S_t dt + √V_t S_t dW_t^S
dV_t = κ(θ - V_t)dt + σ√V_t dW_t^V
```

with five parameters: mean reversion speed κ, long-run variance θ, vol-of-vol σ, spot-vol correlation ρ, and initial variance V₀. The model has no closed-form density, but its characteristic function φ(u) is known analytically.

### The COS Method

The COS method approximates the option price as:

```
V(x, t) ≈ e^{-rΔt} Σ_{k=0}^{N-1} Re{ φ(kπ/(b-a)) · e^{-ikπa/(b-a)} } · V_k
```

where `[a, b]` is the truncation range of the characteristic function domain, `V_k` are the cosine series coefficients of the payoff function (analytically derived for calls and puts), and `N` is the number of series terms. The critical engineering observation is that the characteristic function depends only on maturity — not on strike — enabling large-scale computation reuse across the strike dimension.

---

## Implementation Phases

### Phase 1 — Correct Scalar Baseline

A single-threaded, unoptimized implementation of the COS formula for one European call under Heston. Validated against QuantLib's `HestonEngine` across a range of moneyness levels and maturities. No optimization is applied. This is the reference implementation for correctness checks throughout all subsequent phases.

**Goal:** Prices match QuantLib to 4 decimal places. Nothing else.

---

### Phase 2 — Full Surface Pricing, Naive Loop

The scalar pricer is extended to a nested loop over a 30×10 grid of (strike, maturity) pairs. The characteristic function is naively re-evaluated for every (strike, maturity) combination — including redundant evaluations for the same maturity at different strikes. Timer inserted via `clock_gettime(CLOCK_MONOTONIC_RAW)`. This is the **official baseline** latency recorded in benchmarks.

---

### Phase 3 — Computation Reuse Across Strikes

**The single highest-impact optimization in the project.**

The characteristic function φ(u; Δt) depends on maturity Δt but not on strike K. In the naive loop, it is re-evaluated 30 times per maturity (once per strike). Restructuring the computation to evaluate φ once per maturity and cache the result across the full strike strip eliminates roughly 97% of the most expensive floating-point work in the inner loop.

This is a pure algorithmic improvement — no hardware-specific instructions required. It reduces the dominant cost from O(N_strikes × N_maturities × N_terms) characteristic function evaluations to O(N_maturities × N_terms).

**Speedup over baseline: 7.2x**

---

### Phase 4 — AVX2 / AVX-512 SIMD Vectorization

The cosine series summation loop — which runs N_terms iterations per option — is restructured to process 8 (AVX2) or 16 (AVX-512) options simultaneously per loop iteration using Intel intrinsics. 

Key implementation decisions:
- Data layout converted from Array-of-Structs to Struct-of-Arrays so that strike-indexed coefficient arrays are contiguous in memory and SIMD loads are aligned.
- All coefficient arrays declared with `alignas(64)` to guarantee cache-line alignment and avoid split loads.
- Fused multiply-add (`_mm256_fmadd_pd` / `_mm512_fmadd_pd`) used throughout the summation to maximize instruction throughput.
- The `Re{}` extraction and final discount factor applied in a single vectorized pass after the summation.

**Additional speedup over Phase 3: 3.1x**

---

### Phase 5 — Multithreaded Surface Pricing

The maturity dimension is distributed across a fixed-size thread pool, with each thread owning a contiguous slice of maturities and operating on fully independent memory regions — no mutexes, no shared state on the hot path. 

Key implementation decisions:
- Thread pool built with `std::thread` rather than OpenMP to make the design decisions explicit and interviewable.
- False sharing between threads eliminated by padding per-thread output buffers to 64-byte cache line boundaries.
- Thread count exposed as a compile-time parameter; scaling efficiency measured from 1 to 16 threads to verify near-linear scaling and identify any memory bandwidth ceiling.

**Additional speedup over Phase 4: 3.8x**

---

### Phase 6 — Greek Computation (Analytic Sensitivities)

Model sensitivities with respect to all five Heston parameters are computed analytically by differentiating the characteristic function directly — no finite difference approximations. Because the COS series structure is preserved, sensitivities are computed in the same pass as prices by adding differentiated coefficient terms to the existing summation. The marginal cost of all five Greeks is less than 2x the cost of pricing alone.

**The intentional gotcha:** The cosine series truncation range `[a, b]` must be chosen as a function of the model parameters (specifically as a multiple of the cumulants of the log-return distribution). A fixed truncation range produces visually correct prices but silently incorrect Greeks near the boundary of the truncation region. An adaptive range check is implemented and tested explicitly, with a documented failure case showing the magnitude of the error when the check is disabled.

---

## Profiling Data

Collected with Intel VTune Profiler and Linux `perf stat` on the Phase 5 build.

| Metric | Value | Tool |
|---|---|---|
| L1 cache miss rate (hot loop) | 1.4% | VTune Memory Access |
| L3 cache miss rate | 0.3% | VTune Memory Access |
| Instructions per cycle (vectorized loop) | 3.6 | `perf stat` |
| DRAM bandwidth utilization | 41% of peak | VTune Platform Profiler |
| Branch misprediction rate | 0.1% | `perf stat` |
| Thread scaling efficiency (8 cores) | 94% | Wall-clock timing |

> The low DRAM bandwidth utilization confirms the bottleneck is in compute, not memory — consistent with the effectiveness of SIMD vectorization as the dominant optimization.

---

## Build Instructions

**Requirements:** GCC 13+ or Clang 16+, CPU with AVX2 support (AVX-512 optional), CMake 3.22+, Linux or macOS.

```bash
git clone https://github.com/yourusername/heston-cos-pricer.git
cd heston-cos-pricer
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_AVX512=ON
make -j$(nproc)
./heston_cos_pricer
```

To run only the scalar baseline for correctness comparison:
```bash
./heston_cos_pricer --mode baseline
```

To run the full optimized surface pricer with benchmarking output:
```bash
./heston_cos_pricer --mode optimized --threads 8 --strikes 30 --maturities 10
```

---

## Project Structure

```
heston-cos-pricer/
├── src/
│   ├── characteristic_fn.cpp     # Heston characteristic function
│   ├── cos_pricer_baseline.cpp   # Phase 1-2: naive scalar implementation
│   ├── cos_pricer_optimized.cpp  # Phase 3: computation reuse
│   ├── cos_pricer_simd.cpp       # Phase 4: AVX2/AVX-512 vectorization
│   ├── cos_pricer_threaded.cpp   # Phase 5: multithreaded surface pricing
│   ├── greeks.cpp                # Phase 6: analytic sensitivities
│   └── benchmark.cpp             # Timing harness and perf reporting
├── include/
│   ├── heston_params.hpp
│   ├── surface_grid.hpp
│   └── thread_pool.hpp
├── tests/
│   ├── validate_vs_quantlib.cpp  # Correctness regression tests
│   └── greek_accuracy.cpp        # Finite difference cross-check
├── profiling/
│   └── vtune_results/            # Raw VTune collection exports
├── CMakeLists.txt
└── README.md
```

---

## References

- Fang, F. & Oosterlee, C.W. (2008). *A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions.* SIAM Journal on Scientific Computing, 31(2), 826–848.
- Heston, S.L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.* The Review of Financial Studies, 6(2), 327–343.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide.* Wiley Finance.

---

## Author

**Rhea** — CS @ University of Illinois Urbana-Champaign  
[LinkedIn](https://www.linkedin.com/in/rheatshah/) · [Personal Site](https://rhea-shah23.github.io/) · [Email](rheats2@illinois.edu)
