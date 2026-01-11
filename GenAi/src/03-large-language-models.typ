
= LLMs (Large Language Models)

== Tokenizer
*Purpose:* Convert raw text into discrete token IDs for Transformer models.

*Pipeline:* Text → Tokens → Token IDs → Embeddings

*Tokenization Levels:*
- Character-level: small vocabulary, long sequences
- Word-level: large vocabulary, unknown-word problem
- Subword-level (BPE, WordPiece, Unigram): splits words into frequent subwords, avoids OOV issues

*Special Tokens:* `<BOS>`/`<EOS>` mark boundaries, `<PAD>` enables batching, `<UNK>` for unknown tokens.

*Why it matters:* Determines sequence length, vocabulary size, memory usage, and model behavior.

== Decoder Architecture
*Decoder:* Uses masked self-attention so each token can only attend to previous tokens, enforcing causality during generation.

*Autoregressive Generation:* Text generated one token at a time, each predicted token fed back as input.

*Next-Word Prediction:* Model predicts probability distribution over vocabulary for next token, trained using cross-entropy (negative log-likelihood).

*LM Head:* Final linear layer mapping decoder hidden states to vocabulary logits, converted to probabilities via softmax.

== KV Cache
Stores past Keys and Values to avoid recomputing attention during autoregressive decoding.

Cache per layer:
$ K_(1:t) = [K_1; dots; K_t], quad V_(1:t) = [V_1; dots; V_t] $

At step $t+1$, compute only $K_(t+1), V_(t+1)$ and reuse cache:
$ y_t = "softmax"((Q_t K_(1:t)^T) / sqrt(d_k)) V_(1:t) $

*Benefit:* faster inference; *Cost:* extra memory

== Training vs Inference

*Training:* Uses teacher forcing - model sees full input sequence and predicts next token at every position in parallel. LM head outputs logits for all tokens, loss (cross-entropy) computed over entire sequence.

*Inference (Generation):* Text generated step by step - only last hidden state passed through LM head to predict next token. Predicted token appended to input and process repeated autoregressively.

== Encoder-Only (BERT, DistilBERT)

Uses bidirectional self-attention to produce contextual embeddings for each token (good for understanding, not generation).

*BERT:* Pretrained with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
- MLM: mask tokens and predict them using left+right context

*DistilBERT:* Smaller BERT trained via knowledge distillation (student mimics teacher). Reduces depth (~half layers) and uses distillation losses (incl. cosine loss aligning hidden states).

== Special Tokens
- `<pad>` / [PAD]: padding for batching (ignored via attention_mask)
- `<unk>` / [UNK]: unknown token
- `<bos>` / `<s>`: begin of sequence
- `<eos>` / `</s>`: end of sequence (stops generation)
- [CLS]: sequence-level representation for classification (BERT)
- [SEP]: separates segments/sentences (BERT)
- [MASK]: masked LM pre-training target (BERT)

== Transfer Learning
Splits model into body and head. Head is task-specific final network layer. Only change LM Head to customize for your needs (body gets frozen).

== Sampling Strategies

*Greedy Sampling:* Token with highest probability
$ y_t = arg max_y P(y_t | y_(<t), x) $

*Random Sampling:* Sample from probability distribution over full vocab
$ P(y_t = w_i | y_(<t), x) = "softmax"(z_(i,t)) = (exp(z_(i,t)))/(sum_(j=1)^(|V|) exp(z_(i,j))) $

*Temperature (T):* Controls output diversity by rescaling logits before softmax
$ P(y_t = w_i | y_(<t), x) = (exp(z_(t,i) / T))/(sum_j exp(z_(t,j) / T)) $
- $T < 1$: sharper distribution, low diversity
- $T = 1$: normal softmax
- $T > 1$: flatter distribution, higher diversity

*Beam Search:* Keeps top-k partial sequences at each step and prunes rest

*Top-k Sampling:* Keep k most probable tokens, renormalize, and sample. Controls diversity by limiting candidate set size.

*Nucleus Sampling (Top-p):* Keep smallest set of tokens whose cumulative probability ≥ p, then sample. Adapts candidate set to model's confidence.

*Difference:* Top-k uses fixed number of tokens; top-p uses fixed probability mass.