
= LLM Evaluation

== BLEU (N-gram Score)
BLEU compares produced output with reference using overlapping n-grams (often for translation/summarization).

- Count matching n-grams in candidate & reference
- BLEU_n intuition:
  - small n → meaning/word choice
  - large n → fluency/well-formedness
- Common practice: geometric mean of BLEU_n for n=1..4 (Mean BLEU)

=== BLEU with Penalty
Adds penalty for candidate translations whose length is less than references → *Brevity Penalty (BP)*

$ "BLEU" = min(1, exp(1 - "reference-length" / "output-length")) dot (product_(i=1)^4 "precision"_i)^(1/4) $

Practical BLEU (for floating point arithmetic):
$ "BLEU" = "BP" times e^(1/4 times sum_(n=1)^4 log p_n) $

=== BLEU Limitations
- Doesn't consider meaning
- Doesn't directly consider sentence structure
- Doesn't handle morphologically rich languages well
- Doesn't map well to human judgements

== ROUGE
*Recall-Oriented Understudy for Gisting Evaluation*

Measures overlap between generated summary and reference summaries, focusing on how much reference content is recovered.

*Comparison to BLEU:*
- BLEU → precision ("How much of what I generated is correct?")
- ROUGE → recall ("How much of important reference content did I cover?")

=== ROUGE Variants
*ROUGE-N:* n-gram overlap (like BLEU but recall)
- Example: ROUGE-2 → Bigram overlap

*ROUGE-L:* LCS (Longest common subsequence)
- Measures longest common subsequence between candidate and reference
- Captures sentence-level structure and word order
- Does not require consecutive words

*ROUGE-S:* skip bigram overlap
- Counts pairs of words appearing in same order, not necessarily adjacent
- More flexible than ROUGE-2
- Sensitive to overall flow

=== ROUGE-1 Example
Reference tokens: the, cat, sat, on, the, mat\
Prediction tokens: the, cat, is, on, the, mat\
Total overlapping: 5 (the×2, cat, on, mat)\
Total reference: 6
$ "ROUGE-1" = "Total overlapping" / "Total reference" = 5/6 $

== Perplexity
Measures how well a language model predicts a sequence of tokens. Evaluates average uncertainty when assigning probabilities to next token.

*Range:* 1 (best) to $|V|$ (vocab size, worst case)

$ "PPL"(X) = exp{-1/t sum^t_i log p_theta (x_i | x_(<i))} $

Simplified: $"PPL"(X) = (product^t_i p(y_i | y_(<i), x))^(-1/t)$

=== Example
Probabilities: 0.5, 0.25, 0.125, 0.25 for N=4 tokens

$ "PPL"(X) = exp{-1/4 times (log(0.5) + log(0.25) + log(0.125) + log(0.25))} approx 4 $

*Interpretation:* Model is, on average, as uncertain as choosing between 4 tokens

=== Fine-Tuning and Perplexity
- Fine-tuned model hesitates over fewer tokens during next word predictions
- Perplexity score expected to improve (decrease) after fine-tuning
- Fine-tuning allows model to adapt to specific dataset/task → better predictions, lower perplexity
