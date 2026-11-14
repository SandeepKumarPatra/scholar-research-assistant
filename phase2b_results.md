# Phase 2b results (parent_child)

## What methods reduce hallucination in LLMs?


============================================================
ANSWER:
============================================================
Based on the provided papers, the following methods are identified for reducing or mitigating hallucinations in Large Language Models (LLMs):

*   **Prompt Linguistic Nuance Optimization:** Research suggests that prompts characterized by greater formality and concreteness tend to result in reduced hallucination [Exploring the Relationship between LLM Hallucinations and Prompt Linguistic Nuances: Readability, Formality, and Concreteness, Page 1].
*   **Supervised Training/Fine-Tuning:** LLMs can be fine-tuned to reject questions that are likely to produce hallucinations, though this requires expensive supervised training on large labeled datasets [Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge, Page 1].
*   **Detection-Based Mitigation:** Various methodologies have been proposed to detect and mitigate hallucinations, such as using internal hidden states (e.g., the Eigenscore metric) [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 1].
*   **Agentic Frameworks:** Integrating hallucination detection and correction into multi-agent workflows, including the use of smaller, resource-efficient models as critics or reviewers and human-in-the-loop verification, is a research area aimed at enhancing reliability [Good Parenting is all you need -- Multi-agentic LLM Hallucination Mitigation, Page 3].

============================================================
SOURCES:
============================================================
- [Exploring the Relationship between LLM Hallucinations and Prompt Linguistic Nuances: Readability, Formality, and Concreteness, Page 1]
- [Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge, Pages 1-2]
- [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Pages 1-2]
- [Good Parenting is all you need -- Multi-agentic LLM Hallucination Mitigation, Page 3]
- [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Pages 1-2]
============================================================


---

## What is ChainPoll?


============================================================
ANSWER:
============================================================
ChainPoll is a novel approach to hallucination detection that is substantially more accurate than metrics previously cited in the academic literature, such as SelfCheckGPT, GPTScore, G-Eval, and TRUE [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 2-3]. It provides human-readable verbal justifications for its judgments via chain-of-thought text produced during inference [Chainpoll: A high efficacy method for LLM hallucination detection, Page 8].

The methodology for calculating ChainPoll metrics involves the following steps:
1. Asking `gpt-3.5-turbo` whether a completion contains hallucination(s) using a "detailed CoT" (chain-of-thought) prompt designed to elicit a specific, systematic explanation [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 6-7].
2. Running this process multiple times (typically five) [Chainpoll: A high efficacy method for LLM hallucination detection, Page 6].
3. Dividing the number of "yes" answers by the total number of answers to produce a score between 0 and 1 [Chainpoll: A high efficacy method for LLM hallucination detection, Page 6].

ChainPoll is implemented in two specialized versions: ChainPoll-Correctness for detecting open-domain hallucinations and ChainPoll-Adherence for detecting closed-domain hallucinations [Chainpoll: A high efficacy method for LLM hallucination detection, Page 2]. The two versions differ only in the prompt format used when querying `gpt-3.5-turbo` [Chainpoll: A high efficacy method for LLM hallucination detection, Page 7].

============================================================
SOURCES:
============================================================
- [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 2-3]
- [Chainpoll: A high efficacy method for LLM hallucination detection, Page 7]
- [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 6-7]
- [Chainpoll: A high efficacy method for LLM hallucination detection, Page 8]
- [Chainpoll: A high efficacy method for LLM hallucination detection, Page 2]
============================================================


---

## How is hallucination measured in benchmarks?


============================================================
ANSWER:
============================================================
Hallucination measurement in benchmarks varies depending on the specific framework and objectives:

*   **Binary Classification:** Some approaches treat hallucination detection as a binary classification task, where a metric acts as a classifier outputting a scalar score to indicate whether a completion contains a hallucination [Chainpoll: A high efficacy method for LLM hallucination detection, Page 4].
*   **Factuality and Hallucination Rates:** Certain benchmarks, such as HaluEval 2.0, primarily measure factuality through metrics like the Micro Hallucination Rate (MiHR), which tracks the proportion of hallucinatory statements in a response, and the Macro Hallucination Rate (MaHR), which tracks the proportion of responses containing at least one hallucinatory statement [HalluLens: LLM Hallucination Benchmark, Page 19].
*   **Answer and Rationale Accuracy:** Benchmarks like ERBench evaluate hallucinations by focusing on answer accuracy, rationale accuracy, and answer-relation accuracy [HalluLens: LLM Hallucination Benchmark, Page 19].
*   **Refusal/Abstention Metrics:** Some benchmarks analyze refusal or abstention rates. For instance, while some benchmarks deduct refusals as missing rates, othersùsuch as the PreciseWikiQA proposed in HalluLensùutilize a "hallucination when not refused" metric [HalluLens: LLM Hallucination Benchmark, Page 19].
*   **Dataset Construction:** Benchmarks are often composed of lists of prompts, completions, and ground-truth boolean labels that indicate whether a completion contains hallucinations [Chainpoll: A high efficacy method for LLM hallucination detection, Page 4].

============================================================
SOURCES:
============================================================
- [Chainpoll: A high efficacy method for LLM hallucination detection, Page 4]
- [HalluLens: LLM Hallucination Benchmark, Pages 2-3]
- [A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation, Page 6]
- [Chainpoll: A high efficacy method for LLM hallucination detection, Pages 3-4]
- [HalluLens: LLM Hallucination Benchmark, Pages 18-19]
============================================================


---

## What is the HalluLens benchmark designed to evaluate?


============================================================
ANSWER:
============================================================
The HalluLens benchmark is designed to evaluate both intrinsic and extrinsic hallucinations in large language models [HalluLens: LLM Hallucination Benchmark, Pages 1-2]. Specifically, it incorporates three newly proposed extrinsic hallucination evaluation tasks alongside three existing intrinsic hallucination tasks [HalluLens: LLM Hallucination Benchmark, Page 20]. 

The benchmark aims to assess extrinsic hallucinations by evaluating model generations in reference to the model's training data [HalluLens: LLM Hallucination Benchmark, Page 20]. To ensure robustness against data leakage, it utilizes dynamically generated test sets [HalluLens: LLM Hallucination Benchmark, Page 1]. Additionally, the benchmark design is informed by a clear taxonomy that distinguishes hallucination from the broader problem of LLM factuality [HalluLens: LLM Hallucination Benchmark, Pages 1-2, 20].

============================================================
SOURCES:
============================================================
- [HalluLens: LLM Hallucination Benchmark, Page 20]
- [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 7]
- [HalluLens: LLM Hallucination Benchmark, Pages 2-3]
- [HalluLens: LLM Hallucination Benchmark, Page 1]
- [HalluLens: LLM Hallucination Benchmark, Pages 1-2]
============================================================


---

## Which papers discuss hallucination in multilingual settings?


============================================================
ANSWER:
============================================================
The following papers discuss hallucination in multilingual settings:

*   **How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination**: This paper addresses the gap in multilingual hallucination research by conducting a large-scale study across 30 languages. The authors propose a protocol for estimating hallucination rates in open-domain, knowledge-intensive long-form question answering (LFQA) settings and introduce a 30-language evaluation benchmark called MFAVA [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Pages 1-2].
*   **HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations**: This paper introduces "HalluVerse25," a fine-grained multilingual hallucination dataset covering English, Arabic, and Turkish. It aims to address the literature's lack of self-contained, fine-grained multilingual datasets for understanding the nuances of hallucinations [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Pages 1-2].

============================================================
SOURCES:
============================================================
- [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Pages 1-2]
- [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Pages 1-2]
- [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 1]
- [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 6]
- [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 1]
============================================================


---

