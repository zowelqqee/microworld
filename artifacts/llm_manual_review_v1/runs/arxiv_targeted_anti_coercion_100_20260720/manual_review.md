# Manual review — Gemini 3.1 Flash Lite curated batch

Status: **PENDING HUMAN REVIEW.** These records passed unchanged node-quality triage using a proposal-local literal index. They are not admitted, precision-gated, or promoted.

- Batch: `arxiv_targeted_anti_coercion_100_20260720`
- Model: `gemini-3.1-flash-lite`
- Source sentences: 100
- Review candidates: 14
- Rate limit: one request every 5s (maximum 12.0 RPM)

Rows are grouped by explicit, non-decision heuristics derived from the prior learning log. The grouping only suggests a batch-review order: every candidate still needs a human verdict, and any group may contain exceptions.

For every row, fill `literal_support`, `clean_entities`, `new_predicate` with `yes` or `no`, then set `verdict` to `ACCEPT` or `REJECT`. Use `notes` for the concrete reason. An `ACCEPT` is still only a manually verified proposal; do not add it to an overlay in this review file.

## Temporal-referent-likely (1)

| ID | Labels / reasons | Source sentence | Subject | Predicate | Object | Evidence span | literal_support | clean_entities | new_predicate | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arxiv-034:0 | temporal_referent_likely; generic_property_likely | Our axiomatization is given in the style of a quantitative extension of equational logic recently proposed by Mardare, Panangaden, and Plotkin (LICS 2016) that uses equality relations $t \equiv_\varepsilon s$ indexed by rationals, expressing that `$t$ is approximately equal to $s$ up to an error $\varepsilon$'. | quantitative extension of equational logic | is a type of | equational logic | Our axiomatization is given in the style of a quantitative extension of equational logic recently proposed by Mardare, Panangaden, and Plotkin |  |  |  |  |  |

## Generic-property-likely (13)

| ID | Labels / reasons | Source sentence | Subject | Predicate | Object | Evidence span | literal_support | clean_entities | new_predicate | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arxiv-004:0 | generic_property_likely | Then, based on the novel use of channel busyness ratio, a more accurate metric to characterize the network utilization and congestion status, I propose a new wireless congestion control protocol (WCCP) to efficiently and fairly support the transport service in multihop ad hoc networks. | WCCP | is a type of | wireless congestion control protocol | a new wireless congestion control protocol (WCCP) |  |  |  |  |  |
| arxiv-020:0 | generic_property_likely | Major breakthrough in quantum computation has recently been achieved using quantum annealing to develop analog quantum computers instead of gate based computers. | quantum annealing | is a type of | quantum computation | Major breakthrough in quantum computation has recently been achieved using quantum annealing |  |  |  |  |  |
| arxiv-031:0 | generic_property_likely | The principle of the designs is based on utilizing the potential computational efficiency which the Fourier Slice Theorem and the Non-uniform Fast Fourier Transform (NUFFT) will bring forth. | Non-uniform Fast Fourier Transform (NUFFT) | is a type of | Fast Fourier Transform | Non-uniform Fast Fourier Transform (NUFFT) |  |  |  |  |  |
| arxiv-050:0 | generic_property_likely | Fortunately, as a novel computing paradigm, edge computing (MEC) can provide abundant computing resources to execute all or parts of the tasks of MDs and thereby can greatly reduce the energy of MD and improve the QoS of applications. | edge computing | is a type of | computing paradigm | edge computing (MEC) can provide abundant computing resources to execute all or parts of the tasks of MDs and thereby can greatly reduce the energy of MD and improve the QoS of applications. |  |  |  |  |  |
| arxiv-059:0 | generic_property_likely | In this whitepaper we advocate that the Planetary Science (PS) community build a discipline-specific digital library, in collaboration with the existing astronomy digital library, ADS. | ADS | is a type of | astronomy digital library | the existing astronomy digital library, ADS |  |  |  |  |  |
| arxiv-067:0 | generic_property_likely | This chapter will also introduce The QAC Toolkit Max package, analyze its performance, and explore some examples of what it can offer to realtime creative practice. | The QAC Toolkit | is a | Max package | The QAC Toolkit Max package |  |  |  |  |  |
| arxiv-068:0 | generic_property_likely | This naturally includes $\sqrt{\texttt{iSWAP}}$, which provides an advantage over $\texttt{CNOT}$ as a basis gate. | iSWAP | is a type of | basis gate | iSWAP, which provides an advantage over CNOT as a basis gate |  |  |  |  |  |
| arxiv-068:1 | generic_property_likely | This naturally includes $\sqrt{\texttt{iSWAP}}$, which provides an advantage over $\texttt{CNOT}$ as a basis gate. | CNOT | is a type of | basis gate | CNOT as a basis gate |  |  |  |  |  |
| arxiv-072:0 | generic_property_likely | However, due to the lack of an evaluation dataset that directly supports execution-based model evaluation, existing work relies on code surface form similarity metrics (e.g., BLEU, CodeBLEU) for model selection, which can be inaccurate. | BLEU | is a type of | code surface form similarity metrics | code surface form similarity metrics (e.g., BLEU, CodeBLEU) |  |  |  |  |  |
| arxiv-072:1 | generic_property_likely | However, due to the lack of an evaluation dataset that directly supports execution-based model evaluation, existing work relies on code surface form similarity metrics (e.g., BLEU, CodeBLEU) for model selection, which can be inaccurate. | CodeBLEU | is a type of | code surface form similarity metrics | code surface form similarity metrics (e.g., BLEU, CodeBLEU) |  |  |  |  |  |
| arxiv-087:0 | generic_property_likely | The paper introduces MATILDA, a creativity-based data science design platform, showing how it can support the design process of data science pipelines guided by human and computational creativity. | MATILDA | is a type of | creativity-based data science design platform | MATILDA, a creativity-based data science design platform |  |  |  |  |  |
| arxiv-090:0 | generic_property_likely | This paper builds and extends on the authors' previous work related to the algorithmic tool, Cylindrical Algebraic Decomposition (CAD), and one of its core applications, Real Quantifier Elimination (QE). | Cylindrical Algebraic Decomposition | is a type of | algorithmic tool | the algorithmic tool, Cylindrical Algebraic Decomposition (CAD) |  |  |  |  |  |
| arxiv-090:1 | generic_property_likely | This paper builds and extends on the authors' previous work related to the algorithmic tool, Cylindrical Algebraic Decomposition (CAD), and one of its core applications, Real Quantifier Elimination (QE). | Real Quantifier Elimination | is a type of | application | one of its core applications, Real Quantifier Elimination (QE) |  |  |  |  |  |
