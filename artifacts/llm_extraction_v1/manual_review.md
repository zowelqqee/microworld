# Manual review — Gemini extraction pilot

Mark each triple: literal_support (yes/no), clean_entities (yes/no), new_predicate (yes/no), verdict.

## pilot-00

Source: Lastly, while GAs optimizes models (based on evolutionary processes) for better outcomes, probabilistic reasoning lowers their statistical uncertainties.

- `{'subject': 'GAs', 'predicate': 'optimizes', 'object': 'models', 'evidence_span': 'GAs optimizes models'}` — verdict:
- `{'subject': 'probabilistic reasoning', 'predicate': 'lowers', 'object': 'their statistical uncertainties', 'evidence_span': 'probabilistic reasoning lowers their statistical uncertainties'}` — verdict:

## pilot-01

Source: Based on the needs identified from a formative study, this paper presents NB2Slides, an AI system that facilitates users to compose presentations of their data science work.

- `{'subject': 'this paper', 'predicate': 'presents', 'object': 'NB2Slides', 'evidence_span': 'this paper presents NB2Slides'}` — verdict:
- `{'subject': 'NB2Slides', 'predicate': 'facilitates', 'object': 'users to compose presentations of their data science work', 'evidence_span': 'NB2Slides, an AI system that facilitates users to compose presentations of their data science work'}` — verdict:

## pilot-02

Source: Yet adding a focus on the universality of the physicist approach as a generic path to investigate a topic.

- No triples returned.

## pilot-03

Source: SciServer builds upon and extends the SkyServer system of server-side tools that introduced the astronomical community to SQL (Structured Query Language) and has been serving the Sloan Digital Sky Survey catalog data to the public.

- `{'subject': 'SciServer', 'predicate': 'builds upon', 'object': 'the SkyServer system', 'evidence_span': 'SciServer builds upon and extends the SkyServer system of server-side tools'}` — verdict:
- `{'subject': 'SciServer', 'predicate': 'extends', 'object': 'the SkyServer system', 'evidence_span': 'SciServer builds upon and extends the SkyServer system of server-side tools'}` — verdict:
- `{'subject': 'the SkyServer system of server-side tools', 'predicate': 'introduced', 'object': 'the astronomical community to SQL (Structured Query Language)', 'evidence_span': 'the SkyServer system of server-side tools that introduced the astronomical community to SQL (Structured Query Language)'}` — verdict:
- `{'subject': 'the SkyServer system of server-side tools', 'predicate': 'has been serving', 'object': 'the Sloan Digital Sky Survey catalog data to the public', 'evidence_span': 'the SkyServer system of server-side tools that introduced the astronomical community to SQL (Structured Query Language) and has been serving the Sloan Digital Sky Survey catalog data to the public'}` — verdict:

## pilot-04

Source: REGAI improves on the performance of both classical LLMs and RAG-based LLM techniques.

- `{'subject': 'REGAI', 'predicate': 'improves on', 'object': 'the performance of both classical LLMs and RAG-based LLM techniques', 'evidence_span': 'REGAI improves on the performance of both classical LLMs and RAG-based LLM techniques.'}` — verdict:

## pilot-05

Source: Based on this simulation, we propose a novel tutor model for adaptive training, which considers students' proficiency before and during an ongoing training session.

- `{'subject': 'we', 'predicate': 'propose', 'object': 'a novel tutor model for adaptive training', 'evidence_span': 'we propose a novel tutor model for adaptive training'}` — verdict:
- `{'subject': 'a novel tutor model', 'predicate': 'considers', 'object': "students' proficiency before and during an ongoing training session", 'evidence_span': "which considers students' proficiency before and during an ongoing training session"}` — verdict:

## pilot-06

Source: Contemporary fuzz testing techniques focus on identifying memory corruption vulnerabilities that allow adversaries to achieve either remote code execution or information disclosure.

- `{'subject': 'Contemporary fuzz testing techniques', 'predicate': 'focus on', 'object': 'identifying memory corruption vulnerabilities', 'evidence_span': 'Contemporary fuzz testing techniques focus on identifying memory corruption vulnerabilities'}` — verdict:
- `{'subject': 'memory corruption vulnerabilities', 'predicate': 'allow', 'object': 'adversaries to achieve remote code execution', 'evidence_span': 'memory corruption vulnerabilities that allow adversaries to achieve either remote code execution or information disclosure'}` — verdict:
- `{'subject': 'memory corruption vulnerabilities', 'predicate': 'allow', 'object': 'adversaries to achieve information disclosure', 'evidence_span': 'memory corruption vulnerabilities that allow adversaries to achieve either remote code execution or information disclosure'}` — verdict:

## pilot-07

Source: Existing AML threat evaluation approaches focus primarily on technical attack robustness, overlooking crucial real-world factors like deployment environments, system dependencies, and attack feasibility.

- `{'subject': 'Existing AML threat evaluation approaches', 'predicate': 'focus primarily on', 'object': 'technical attack robustness', 'evidence_span': 'Existing AML threat evaluation approaches focus primarily on technical attack robustness'}` — verdict:
- `{'subject': 'Existing AML threat evaluation approaches', 'predicate': 'overlooking', 'object': 'crucial real-world factors like deployment environments, system dependencies, and attack feasibility', 'evidence_span': 'Existing AML threat evaluation approaches focus primarily on technical attack robustness, overlooking crucial real-world factors like deployment environments, system dependencies, and attack feasibility.'}` — verdict:

## pilot-08

Source: This study addresses the crucial need for fairness and explainability in AI applications within healthcare to ensure equitable outcomes across diverse patient demographics.

- `{'subject': 'This study', 'predicate': 'addresses', 'object': 'the crucial need for fairness and explainability in AI applications within healthcare', 'evidence_span': 'This study addresses the crucial need for fairness and explainability in AI applications within healthcare'}` — verdict:

## pilot-09

Source: AutoSlim uses Random Forest classification to prune low-impact transitions based on edge scores and structural features, significantly reducing automata graph density while preserving semantic correctness.

- `{'subject': 'AutoSlim', 'predicate': 'uses', 'object': 'Random Forest classification', 'evidence_span': 'AutoSlim uses Random Forest classification'}` — verdict:
- `{'subject': 'AutoSlim', 'predicate': 'prunes', 'object': 'low-impact transitions', 'evidence_span': 'AutoSlim uses Random Forest classification to prune low-impact transitions'}` — verdict:
- `{'subject': 'Random Forest classification', 'predicate': 'prunes', 'object': 'low-impact transitions', 'evidence_span': 'Random Forest classification to prune low-impact transitions'}` — verdict:

## pilot-10

Source: We design a learning-based model (ProbCT) to achieve CT of such clouds, based on noisy multi-view spaceborne images.

- `{'subject': 'We', 'predicate': 'design', 'object': 'a learning-based model (ProbCT)', 'evidence_span': 'We design a learning-based model (ProbCT)'}` — verdict:
- `{'subject': 'a learning-based model (ProbCT)', 'predicate': 'achieve', 'object': 'CT of such clouds', 'evidence_span': 'a learning-based model (ProbCT) to achieve CT of such clouds'}` — verdict:

## pilot-11

Source: With a focus on trapped-ion simulators, we numerically benchmark both methods and show that high-fidelity wave packets can be achieved in near-term experiments.

- `{'subject': 'we', 'predicate': 'numerically benchmark', 'object': 'both methods', 'evidence_span': 'we numerically benchmark both methods'}` — verdict:
- `{'subject': 'we', 'predicate': 'show', 'object': 'that high-fidelity wave packets can be achieved in near-term experiments', 'evidence_span': 'we numerically benchmark both methods and show that high-fidelity wave packets can be achieved in near-term experiments'}` — verdict:

## pilot-12

Source: Based on these results, we construct debiased estimators that are asymptotically Gaussian, allowing hypothesis testing for linear combinations of precision matrix entries across populations.

- `{'subject': 'we', 'predicate': 'construct', 'object': 'debiased estimators', 'evidence_span': 'we construct debiased estimators'}` — verdict:
- `{'subject': 'debiased estimators', 'predicate': 'are', 'object': 'asymptotically Gaussian', 'evidence_span': 'debiased estimators that are asymptotically Gaussian'}` — verdict:
- `{'subject': 'debiased estimators', 'predicate': 'allowing', 'object': 'hypothesis testing for linear combinations of precision matrix entries across populations', 'evidence_span': 'allowing hypothesis testing for linear combinations of precision matrix entries across populations'}` — verdict:

## pilot-13

Source: In this paper, we propose a novel architecture based on Quantum Artificial Intelligence by enabling quantum and AI at various levels of navigation decision making and communication process in Autonomous vehicles : Quantum Neural Networks for multimodal sensor fusion, Nav-Q for Quantum reinforcement learning for navigation policy optimization and finally post-quantum cryptographic protocols for secure communication.

- `{'subject': 'we', 'predicate': 'propose', 'object': 'a novel architecture', 'evidence_span': 'we propose a novel architecture'}` — verdict:
- `{'subject': 'a novel architecture', 'predicate': 'is based on', 'object': 'Quantum Artificial Intelligence', 'evidence_span': 'a novel architecture based on Quantum Artificial Intelligence'}` — verdict:
- `{'subject': 'Quantum Neural Networks', 'predicate': 'are for', 'object': 'multimodal sensor fusion', 'evidence_span': 'Quantum Neural Networks for multimodal sensor fusion'}` — verdict:
- `{'subject': 'Nav-Q', 'predicate': 'is for', 'object': 'Quantum reinforcement learning', 'evidence_span': 'Nav-Q for Quantum reinforcement learning'}` — verdict:
- `{'subject': 'Quantum reinforcement learning', 'predicate': 'is for', 'object': 'navigation policy optimization', 'evidence_span': 'Quantum reinforcement learning for navigation policy optimization'}` — verdict:
- `{'subject': 'post-quantum cryptographic protocols', 'predicate': 'are for', 'object': 'secure communication', 'evidence_span': 'post-quantum cryptographic protocols for secure communication'}` — verdict:

## pilot-14

Source: Thus, the proposed framework addresses fundamental challenges in autonomous vehicles navigation by providing quantum performance and future proof security.

- `{'subject': 'the proposed framework', 'predicate': 'addresses', 'object': 'fundamental challenges in autonomous vehicles navigation', 'evidence_span': 'the proposed framework addresses fundamental challenges in autonomous vehicles navigation'}` — verdict:
- `{'subject': 'the proposed framework', 'predicate': 'providing', 'object': 'quantum performance and future proof security', 'evidence_span': 'the proposed framework addresses fundamental challenges in autonomous vehicles navigation by providing quantum performance and future proof security'}` — verdict:
