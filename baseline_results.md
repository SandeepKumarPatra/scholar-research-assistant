# Baseline results (naive RAG)

## What methods reduce hallucination in LLMs?


============================================================
ANSWER:
============================================================
Based on the provided papers, several methods and approaches are used to mitigate LLM hallucinations:

*   **Abstention or refusal:** Some mitigation strategies involve the model abstaining or refusing to answer when uncertain, which prevents the generation of hallucinated content [HalluLens: LLM Hallucination Benchmark, Page 4].
*   **Knowledge enhancement:** Factuality can be improved by providing additional knowledge to the model, such as through Retrieval-Augmented Generation (RAG) [HalluLens: LLM Hallucination Benchmark, Page 4].
*   **Detection and Training-based approaches:** Other methods include fine-tuning LLMs to reject questions likely to produce hallucinations or training classifiers to detect hallucinations [Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge, Page 1].
*   **General categorization of mitigation:** The broader body of work on tackling LLM hallucinations falls into three main areas: detection (identifying hallucinated content), evaluation (quantifying the extent and severity of hallucinations), and mitigation (focusing on reducing hallucinative tendencies) [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 1].

============================================================
SOURCES:
============================================================
- [Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge, Page 1]
- [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 1]
- [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 2]
- [HalluLens: LLM Hallucination Benchmark, Page 4]
============================================================


---

## What is ChainPoll?


============================================================
ANSWER:
============================================================
ChainPoll is a novel methodology designed to detect hallucinations in large language model (LLM) outputs [Chainpoll: A high efficacy method for LLM hallucination detection, Page 1]. It operates by taking a dataset of text inputs and their corresponding LLM completions to determine if any hallucinations are present [Chainpoll: A high efficacy method for LLM hallucination detection, Page 3].

The ChainPoll process involves three main steps:
1. Asking `gpt-3.5-turbo` whether a completion contains hallucinations using a detailed, carefully engineered prompt known as "detailed CoT" (chain-of-thought) [Chainpoll: A high efficacy method for LLM hallucination detection, Page 7].
2. Running this step multiple times, typically five, using batch inference [Chainpoll: A high efficacy method for LLM hallucination detection, Page 7].
3. Dividing the number of "yes" answers by the total number of answers to produce a score between 0 and 1 [Chainpoll: A high efficacy method for LLM hallucination detection, Page 7].

ChainPoll is differentiated by its use of "detailed CoT," which elicits specific, systematic explanations from the model, and its request for boolean judgments rather than numeric scores [Chainpoll: A high efficacy method for LLM hallucination detection, Page 7]. It also provides human-readable verbal justifications for its judgments based on the CoT text generated during inference [Chainpoll: A high efficacy method for LLM hallucination detection, Page 8].

There are two specialized versions of the metric:
* **ChainPoll-Correctness**: Used to detect open-domain hallucinations (false claims about the world) [Chainpoll: A high efficacy method for LLM hallucination detection, Page 3, Page 7].
* **ChainPoll-Adherence**: Used to detect closed-domain hallucinations (measuring an LLM's reasoning within provided documents and context, often in Retrieval Augmented Generation workflows) [Chainpoll: A high efficacy method for LLM hallucination detection, Page 1, Page 7].

============================================================
SOURCES:
============================================================
- [Chainpoll: A high efficacy method for LLM hallucination detection, Page 3]
- [Chainpoll: A high efficacy method for LLM hallucination detection, Page 1]
- [Chainpoll: A high efficacy method for LLM hallucination detection, Page 8]
- [Chainpoll: A high efficacy method for LLM hallucination detection, Page 7]
- [Chainpoll: A high efficacy method for LLM hallucination detection, Page 6]
============================================================


---

## How is hallucination measured in benchmarks?


============================================================
ANSWER:
============================================================
In the context of evaluating Large Language Models (LLMs), hallucination measurement is approached through various metrics and methodologies:

*   **Statistical and Model-based Metrics:** Standard hallucination metrics often utilize statistical methods such as ROUGE or BLEU, as well as model-based metrics that rely on information extraction or question-answering for answer matching [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 2].
*   **Hallucination Rates:** Some benchmarks, such as HaluEval 2.0, quantify hallucinations using two specific rates:
    *   **Micro Hallucination Rate (MiHR):** The proportion of hallucinatory statements within each response [HalluLens: LLM Hallucination Benchmark, Page 19].
    *   **Macro Hallucination Rate (MaHR):** The proportion of responses that contain at least one hallucinatory statement [HalluLens: LLM Hallucination Benchmark, Page 19].
*   **Accuracy and Relation Metrics:** Benchmarks like ERBench evaluate extrinsic hallucinations by focusing on answer accuracy, rationale accuracy, and answer-relation accuracy [HalluLens: LLM Hallucination Benchmark, Page 19].
*   **Refusal/Abstention:** Some frameworks measure refusal rates, noting that failure to refuse when appropriate can be categorized as a hallucination [HalluLens: LLM Hallucination Benchmark, Page 19]. For example, ERBench treats refusal or abstention cases as "missing rates" rather than hallucinations [HalluLens: LLM Hallucination Benchmark, Page 19].
*   **General Benchmark Design:** An effective benchmark for measuring hallucination should maintain strong stability (low intra-model variance) and high sensitivity (inter-model variance exceeding intra-model variance) [HalluLens: LLM Hallucination Benchmark, Page 5]. Additionally, modern approaches are incorporating dynamic test set generation to mitigate data leakage and ensure that benchmarks measure a model's underlying capabilities rather than its ability to game evaluation metrics [HalluLens: LLM Hallucination Benchmark, Page 1, Page 5].

============================================================
SOURCES:
============================================================
- [HalluLens: LLM Hallucination Benchmark, Page 5]
- [HalluLens: LLM Hallucination Benchmark, Page 19]
- [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 2]
- [HalluLens: LLM Hallucination Benchmark, Page 2]
- [HalluLens: LLM Hallucination Benchmark, Page 1]
============================================================


---

## What is the HalluLens benchmark designed to evaluate?


============================================================
ANSWER:
============================================================
The HalluLens benchmark is designed to evaluate LLM hallucinations by distinguishing between extrinsic and intrinsic types and incorporating a clear taxonomy [HalluLens: LLM Hallucination Benchmark, Page 1]. Specifically, it includes three newly proposed extrinsic hallucination evaluation tasks alongside three existing intrinsic hallucination tasks [HalluLens: LLM Hallucination Benchmark, Page 20]. 

The benchmark aims to:
*   Evaluate extrinsic hallucinations by assessing model generations in reference to the modelÆs training data [HalluLens: LLM Hallucination Benchmark, Page 20].
*   Ensure robustness against data leakage by dynamically generating test sets [HalluLens: LLM Hallucination Benchmark, Page 1, Page 20].
*   Provide a comprehensive, reliable, and ungameable evaluation that reflects real-world applications across diverse domains, tasks, and scenarios [HalluLens: LLM Hallucination Benchmark, Page 1, Page 5, Page 27].

============================================================
SOURCES:
============================================================
- [HalluLens: LLM Hallucination Benchmark, Page 1]
- [HalluLens: LLM Hallucination Benchmark, Page 20]
- [HalluLens: LLM Hallucination Benchmark, Page 27]
- [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 2]
- [HalluLens: LLM Hallucination Benchmark, Page 5]
============================================================


---

## Which papers discuss hallucination in multilingual settings?


============================================================
ANSWER:
============================================================
The following papers discuss hallucination in multilingual settings:

*   **HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations**: This paper introduces a fine-grained multilingual dataset spanning English, Arabic, and Turkish [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 1].
*   **How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination**: This paper addresses gaps in multilingual hallucination research, noting that existing efforts have been sparse and primarily targeted at high-resource languages [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 1].

============================================================
SOURCES:
============================================================
- [HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations, Page 1]
- [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 1]
- [HalluLens: LLM Hallucination Benchmark, Page 24]
- [Measuring and Reducing LLM Hallucination without Gold-Standard Answers, Page 11]
- [How Much Do LLMs Hallucinate across Languages? On Realistic Multilingual Estimation of LLM Hallucination, Page 11]
============================================================


---

## My verdicts
| Question | Verdict | Notes |
|---|---|---|
| Methods to reduce hallucination | Weak | Few papers, generic bullet, missing mitigation papers |
| ChainPoll | Good | Single-paper, well cited |
| Measuring in benchmarks | Okay | Mostly HalluLens p19 (related-work table) |
| HalluLens purpose | Mostly good | Encoding glitch "modelÆs"; irrelevant source |
| Multilingual papers | Weak | Only 2 papers; appendix pages retrieved |

## Observed weaknesses
- Long papers dominate top-5
- Appendix/reference chunks retrieved
- pypdf encoding glitch
- Broad questions underperform