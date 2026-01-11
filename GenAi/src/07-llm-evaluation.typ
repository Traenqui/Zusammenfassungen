#import "../../template_zusammenf.typ": *

= LLM Evaluation

== BLEU (N-gram score)
BLEU compares a candidate output to one or more reference texts using overlapping *n-grams* (common in translation, sometimes in summarization).

- Count matching n-grams between candidate and reference(s)
- Intuition:
  - small $n$ → meaning / word choice
  - large $n$ → fluency / well-formedness
- Common practice: geometric mean of $n=1..4$ (Mean BLEU)

=== BLEU with brevity penalty (BP)
Penalizes candidates that are shorter than the reference.

_Brevity penalty_: $"BP" = cases(1 "if" "candidate" >= "reference", e^(1 - r / c) "if" c < r) $

Full BLEU (illustrative form):
$ "BLEU" = min(1, exp(1 - "reference length" / "candidate length")) dot (product_(i=1)^4 p_i)^(1/4) $

Practical (numerically stable) form: $"BLEU" = "BP" times e^((1/4) times sum_(n=1)^4 log p_n)$

=== BLEU limitations
- Surface overlap only → does not measure meaning / faithfulness
- Weak signal for syntax/grammar quality
- Penalizes valid paraphrases
- Often correlates poorly with human judgements (esp. for summarization and morphologically rich languages)

== ROUGE
*Recall-Oriented Understudy for Gisting Evaluation* (common for summarization).

*BLEU vs ROUGE:*
- BLEU → precision: “how much of what I generated appears in the reference?”
- ROUGE → recall: “how much reference content did I cover?”

=== ROUGE variants
- *ROUGE-N:* n-gram overlap (recall)  
  Example: ROUGE-2 → bigram overlap
- *ROUGE-L:* longest common subsequence (LCS)  
  Captures word order / sentence-level structure without requiring consecutive matches
- *ROUGE-S:* skip-bigram overlap  
  Counts word pairs in order, not necessarily adjacent → more flexible than ROUGE-2

=== ROUGE-1 example
*Reference tokens*: the, cat, sat, on, the, mat 
*Prediction tokens*: the, cat, is, on, the, mat \
Overlapping: 5 (the×2, cat, on, mat), reference total: 6

$ "ROUGE-1" = "Total overlapping" / "Total reference" = 5/6 $

== Perplexity
Measures how well a language model predicts a token sequence (average uncertainty over next-token probabilities).

*Range:* 1 (best) to $|V|$ (worst-case, uniform over vocab)

$ "PPL"(X) = exp(-1 / t sum_(i=1)^t log p_theta(x_i | x_(<i))) = (product_(i=1)^t p(x_i | x_(<i)))^(-1 / t) $

*Example*
Probabilities: 0.5, 0.25, 0.125, 0.25 for $t=4$:
$ "PPL" = exp(-1/4 times (log 0.5 + log 0.25 + log 0.125 + log 0.25)) approx 4 $

Interpretation: on average, the model is as uncertain as choosing between 4 tokens.

=== Fine-tuning and perplexity
- Fine-tuning typically increases probability on in-domain sequences
- Expected effect: perplexity decreases (model is “less surprised”)

#hinweis[
Use BLEU/ROUGE for reference-based tasks (translation/summarization), and PPL for language modeling fit; for assistants, prefer human eval + task-specific metrics (factuality, faithfulness, safety, latency/cost).
]
