# Retrieval Modes Comparison Report (Phase 1 vs 2 vs 3)

Evaluation benchmark running **retrieval ONLY** (no Gemini calls) across `naive`, `parent_child`, `hybrid` (Dense + BM25 + RRF), and `hybrid_rerank` (Cross-Encoder).

## 1. Latency Benchmark Summary

| Question | Naive (ms) | Parent-Child (ms) | Hybrid (ms) | Hybrid+Rerank (ms) |
|:---|---:|---:|---:|---:|
| What methods reduce hallucination in LLMs? | 20690.6 | 4453.1 | 823.9 | 118662.4 |
| What is ChainPoll? | 9669.2 | 2142.4 | 1333.6 | 30580.3 |
| How is hallucination measured in benchmarks? | 10342.8 | 1435.4 | 1351.7 | 28434.7 |
| What is the HalluLens benchmark designed to evaluate? | 11377.7 | 1755.9 | 1118.4 | 28459.4 |
| Which papers discuss hallucination in multilingual settings? | 5384.1 | 948.1 | 827.9 | 27054.0 |
| **Average** | **11492.9** | **2147.0** | **1091.1** | **46638.2** |

## 2. Top-5 Retrieved Sources per Mode

### Question 1: What methods reduce hallucination in LLMs?

| Rank | Naive (Phase 1) | Parent-Child (Phase 2) | Hybrid (Phase 3) | Hybrid + Rerank (Phase 3) |
|:---:|:---|:---|:---|:---|
| 1 | [Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge, Page 1] `(0.6613)` | [Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge, Pages 1-2] (*1 Introduction*) `(0.6738)` | [Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge, Page 1] (*Abstract*) `(0.0310)` | [A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation, Page 2] (*2 Related Work*) `(0.9949)` |
| 2 | [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 1] `(0.6455)` | [Exploring the Relationship between LLM Hallucinations and Prompt Linguistic Nuances: Readability, Formality, and Concreteness, Page 1] (*Abstract*) `(0.6702)` | [Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge, Pages 1-2] (*1 Introduction*) `(0.0309)` | [Understanding LLM Hallucinations Caused by Soft Errors, Page 2] (*2 Related Works*) `(0.9913)` |
| 3 | [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 2] `(0.6450)` | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Pages 1-2] (*2 Related Work*) `(0.6605)` | [A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation, Page 2] (*2 Related Work*) `(0.0303)` | [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 1] (*ABSTRACT*) `(0.9870)` |
| 4 | [HalluLens: LLM Hallucination Benchmark, Page 4] `(0.6430)` | [Good Parenting is all you need -- Multi-agentic LLM Hallucination Mitigation, Page 3] (*Addressing Gaps in Existing Research*) `(0.6479)` | [Understanding LLM Hallucinations Caused by Soft Errors, Page 1] (*Abstract*) `(0.0283)` | [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 8] (*5 Mitigating Hallucination with FEWL*) `(0.9827)` |
| 5 | [Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge, Page 1] `(0.6425)` | [Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge, Page 1] (*Abstract*) `(0.6457)` | [Understanding LLM Hallucinations Caused by Soft Errors, Page 2] (*2 Related Works*) `(0.0257)` | [HalluLens: LLM Hallucination Benchmark, Pages 3-4] (*2.1 Distinction between LLM Hallucination and Factuality*) `(0.9819)` |

---

### Question 2: What is ChainPoll?

| Rank | Naive (Phase 1) | Parent-Child (Phase 2) | Hybrid (Phase 3) | Hybrid + Rerank (Phase 3) |
|:---:|:---|:---|:---|:---|
| 1 | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 3] `(0.5721)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 2-3] (*1.1 Summary of contributions*) `(0.6140)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 2-3] (*1.1 Summary of contributions*) `(0.0325)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 2-3] (*1.1 Summary of contributions*) `(0.9974)` |
| 2 | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 1] `(0.5525)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 7] (*4.2.1 ChainPoll-Correctness and ChainPoll-Adherence*) `(0.6120)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 8] (*5 Results*) `(0.0320)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 1] (*ABSTRACT*) `(0.9816)` |
| 3 | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 8] `(0.5501)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 6-7] (*4.2 Defining ChainPoll*) `(0.6096)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 6-7] (*4.2 Defining ChainPoll*) `(0.0313)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 8] (*5 Results*) `(0.9367)` |
| 4 | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 7] `(0.5487)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 8] (*5 Results*) `(0.6011)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 7] (*4.2.1 ChainPoll-Correctness and ChainPoll-Adherence*) `(0.0298)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 6-7] (*4.2 Defining ChainPoll*) `(0.8791)` |
| 5 | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 6] `(0.5387)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 2 [Table]] (*Table 2*) `(0.5823)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 6] (*4.1 Metrics evaluated*) `(0.0296)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 7] (*4.2.1 ChainPoll-Correctness and ChainPoll-Adherence*) `(0.8572)` |

---

### Question 3: How is hallucination measured in benchmarks?

| Rank | Naive (Phase 1) | Parent-Child (Phase 2) | Hybrid (Phase 3) | Hybrid + Rerank (Phase 3) |
|:---:|:---|:---|:---|:---|
| 1 | [HalluLens: LLM Hallucination Benchmark, Page 5] `(0.6841)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Page 4] (*2.1 Approach*) `(0.7056)` | [HalluLens: LLM Hallucination Benchmark, Pages 1-2] (*1 Introduction*) `(0.0320)` | [HalluLens: LLM Hallucination Benchmark, Pages 1-2] (*1 Introduction*) `(0.9515)` |
| 2 | [HalluLens: LLM Hallucination Benchmark, Page 19] `(0.6800)` | [HalluLens: LLM Hallucination Benchmark, Pages 1-2] (*1 Introduction*) `(0.7052)` | [HalluLens: LLM Hallucination Benchmark, Page 16] (*5 Revisiting Existing Benchmarks*) `(0.0307)` | [HalluLens: LLM Hallucination Benchmark, Page 16] (*5 Revisiting Existing Benchmarks*) `(0.9052)` |
| 3 | [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 2] `(0.6661)` | [HalluLens: LLM Hallucination Benchmark, Pages 18-19] (*5.3 Discussion on existing "factuality hallucination" benchmarks.*) `(0.6991)` | [HalluLens: LLM Hallucination Benchmark, Pages 2-3] (*2 Overview of LLM Hallucination*) `(0.0299)` | [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 1] (*ABSTRACT*) `(0.9030)` |
| 4 | [HalluLens: LLM Hallucination Benchmark, Page 2] `(0.6642)` | [HalluLens: LLM Hallucination Benchmark, Pages 2-3] (*2 Overview of LLM Hallucination*) `(0.6987)` | [HalluLens: LLM Hallucination Benchmark, Pages 18-19] (*5.3 Discussion on existing "factuality hallucination" benchmarks.*) `(0.0296)` | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Pages 1-2] (*2 Related Work*) `(0.8711)` |
| 5 | [HalluLens: LLM Hallucination Benchmark, Page 1] `(0.6617)` | [A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation, Page 6] (*4 Experiments*) `(0.6848)` | [A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation, Page 6] (*4 Experiments*) `(0.0279)` | [HalluLens: LLM Hallucination Benchmark, Pages 18-19] (*5.3 Discussion on existing "factuality hallucination" benchmarks.*) `(0.8691)` |

---

### Question 4: What is the HalluLens benchmark designed to evaluate?

| Rank | Naive (Phase 1) | Parent-Child (Phase 2) | Hybrid (Phase 3) | Hybrid + Rerank (Phase 3) |
|:---:|:---|:---|:---|:---|
| 1 | [HalluLens: LLM Hallucination Benchmark, Page 1] `(0.6169)` | [HalluLens: LLM Hallucination Benchmark, Pages 18-19] (*5.3 Discussion on existing "factuality hallucination" benchmarks.*) `(0.6200)` | [HalluLens: LLM Hallucination Benchmark, Page 20] (*7 Conclusion*) `(0.0320)` | [HalluLens: LLM Hallucination Benchmark, Page 20] (*7 Conclusion*) `(0.9377)` |
| 2 | [HalluLens: LLM Hallucination Benchmark, Page 20] `(0.6152)` | [HalluLens: LLM Hallucination Benchmark, Page 20] (*7 Conclusion*) `(0.6064)` | [HalluLens: LLM Hallucination Benchmark, Page 1] (*HalluLens: LLM Hallucination Benchmark*) `(0.0304)` | [HalluLens: LLM Hallucination Benchmark, Pages 1-2] (*1 Introduction*) `(0.8636)` |
| 3 | [HalluLens: LLM Hallucination Benchmark, Page 27] `(0.6113)` | [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 7] (*4 Experiments*) `(0.6032)` | [HalluLens: LLM Hallucination Benchmark, Pages 1-2] (*1 Introduction*) `(0.0297)` | [HalluLens: LLM Hallucination Benchmark, Pages 2-3] (*2 Overview of LLM Hallucination*) `(0.8066)` |
| 4 | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 2] `(0.6070)` | [HalluLens: LLM Hallucination Benchmark, Pages 1-2] (*1 Introduction*) `(0.5938)` | [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 7] (*4 Experiments*) `(0.0257)` | [HalluLens: LLM Hallucination Benchmark, Page 1] (*HalluLens: LLM Hallucination Benchmark*) `(0.5645)` |
| 5 | [HalluLens: LLM Hallucination Benchmark, Page 5] `(0.5948)` | [HalluLens: LLM Hallucination Benchmark, Pages 2-3] (*2 Overview of LLM Hallucination*) `(0.5922)` | [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 12-13] (*7 Conclusion*) `(0.0253)` | [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 7] (*4 Experiments*) `(0.5552)` |

---

### Question 5: Which papers discuss hallucination in multilingual settings?

| Rank | Naive (Phase 1) | Parent-Child (Phase 2) | Hybrid (Phase 3) | Hybrid + Rerank (Phase 3) |
|:---:|:---|:---|:---|:---|
| 1 | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 1] `(0.6309)` | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 1] (*1 Introduction*) `(0.6253)` | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 1] (*1 Introduction*) `(0.0318)` | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 1] (*1 Introduction*) `(0.8631)` |
| 2 | [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 1] `(0.6282)` | [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Pages 1-2] (*1 Introduction*) `(0.6242)` | [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 1] (*Abstract*) `(0.0310)` | [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 3] (*3 Hallucination Detection*) `(0.7817)` |
| 3 | [HalluLens: LLM Hallucination Benchmark, Page 24] `(0.6267)` | [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 1] (*Abstract*) `(0.5992)` | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 1] (*Abstract*) `(0.0307)` | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Pages 7-8] (*6 Conclusion*) `(0.7001)` |
| 4 | [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 11] `(0.6237)` | [Exploring the Relationship between LLM Hallucinations and Prompt Linguistic Nuances: Readability, Formality, and Concreteness, Page 1] (*Abstract*) `(0.5961)` | [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Pages 1-2] (*1 Introduction*) `(0.0306)` | [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Pages 1-2] (*1 Introduction*) `(0.6394)` |
| 5 | [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 11] `(0.6113)` | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 1] (*Abstract*) `(0.5959)` | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Pages 7-8] (*6 Conclusion*) `(0.0291)` | [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 1] (*Abstract*) `(0.6235)` |

---
