#import "../template_zusammenf.typ": *
#import "@preview/wrap-it:0.1.1": wrap-content

#show: project.with(
  authors: ("Jonas Gerber"),
  fach: "GenAI",
  fach-long: "Generative AI",
  semester: "HS25",
  language: "en",
  column-count: 4,
  font-size: 4pt,
  landscape: true,
)

= Latent Space
_Latent space_: learned hidden vector space where inputs are encoded; sampling/moving yields meaningful variations.
- Nearby points ≈ semantically similar.
- Vector arithmetic can encode relations #hinweis("king - man + woman ≈ queen").

= Deep Neural Networks (DNN) - Recap

_N-Grams_: Fixed-size context windows → sparse reps, limited generalization; cannot capture long-range deps beyond window size $n$.

_Hyperparameters_:
Learning rate, number of epochs, batch size, architecture choices #hinweis[(layers, neurons per layer, activations)], regularization #hinweis[(L1, L2, dropout)].

== Training a Neural Network
*Stochastic Gradient Descent (SGD):*
Update weights based on mini-batches to reduce loss $ w_(t+1) <- w_t - alpha partial/(partial w) log(p(x_i)) $where $alpha$ is learning rate

*Batch Gradient Descent:*
Update weights based on entire dataset
$ w_(t+1) <- w_t - alpha (1/N) sum^N_(i=1) partial/(partial w) log(p(x_i)) $
More stable but computationally expensive

*Mini-batch Gradient Descent:*
Compromise between SGD and Batch GD
$ w_(t+1) <- w_t - alpha (1/M) sum^M_(i=1) partial/(partial w) log(p(x_i)) $
Balances stability and computational efficiency

== Activation Functions
#table(
  columns: (1.2fr, 1fr, 1.5fr, 1.5fr),
  table.header([*Function*], [*Formula*], [*Pros*], [*Cons*]),
  [ReLU], [$f(x) = max(0, x)$], [Computationally efficient, mitigates vanishing gradient], ["Dying ReLU" problem where neurons can become inactive],
  [Sigmoid], [$f(x) = 1/(1 + e^(-x))$], [Outputs in $(0, 1)$, useful for probabilities], [Vanishing gradient for large inputs, not zero-centered],
  [Tanh], [$f(x) = (e^x - e^(-x))/(e^x + e^(-x))$], [Outputs in $(-1, 1)$, zero-centered], [Vanishing gradient for large inputs],
  [Leaky ReLU], [$f(x) = max(alpha x, x)$\ #hinweis[($alpha approx 0.01$)]], [Mitigates dying ReLU, allows small gradient when inactive], [Introduces hyperparameter $alpha$],
  [Softmax], [$f(x_i) = e^(x_i) / sum_j e^(x_j)$], [Outputs probability distribution, used in final layer], [Only suitable for output layer],
)

*Derivative of ReLU:* $f'(x) = cases(1 "if" x > 0, 0 "if" x <= 0)$\
*Derivative of Sigmoid:* $f'(x) = f(x)(1 - f(x))$\
*Derivative of Tanh:* $f'(x) = 1 - f(x)^2$

== Loss Functions
#table(
  columns: (1.2fr, 1.8fr, 1fr),
  table.header([*Loss Function*], [*Formula*], [*Use Case*]),
  [Mean Squared Error\ (MSE)], [$L = 1/n sum_(i=1)^n (y_i - hat(y)_i)^2$], [Regression],
  [Mean Absolute Error\ (MAE)], [$L = 1/n sum_(i=1)^n |y_i - hat(y)_i|$], [Regression\ #hinweis[(robust to outliers)]],
  [Binary Cross-Entropy], [$L = -1/n sum_(i=1)^n [y_i log(hat(y)_i) + (1-y_i)log(1-hat(y)_i)]$], [Binary\ classification],
  [Categorical Cross-Entropy], [$L = -sum_(i=1)^n sum_(c=1)^C y_(i,c) log(hat(y)_(i,c))$], [Multi-class\ classification],
)

*Cross-Entropy* measures the difference between two probability distributions. Lower values indicate better match between predicted and true distributions.

== Convolutional Neural Networks (CNN)
*Why CNNs for images?*
Fully-connected networks ignore spatial structure and have too many parameters for high-resolution images.

*Key concepts:*
- *Convolution:* slide kernel/filter over input to detect local patterns
- *Padding:* add borders to maintain spatial dimensions
  #hinweis[(SAME padding: output size = input size; VALID: no padding)]
- *Stride:* step size of kernel movement, controls overlap of receptive fields
  $ "output size" = ((n + 2p - f)/s) + 1 $
  where $n$ = input size, $p$ = padding, $f$ = filter size, $s$ = stride
- *Pooling:* downsample feature maps to reduce dimensions and computation
  #hinweis[(Max pooling: take maximum value; Average pooling: take mean)]

*Typical CNN architecture:*
Input $->$ Conv + ReLU $->$ Pool $->$ Conv + ReLU $->$ Pool $->$ Flatten $->$ FC $->$ Softmax

== Evaluation Metrics
#table(
  columns: (1fr, 1.5fr, 1.5fr),
  table.header([*Metric*], [*Formula*], [*When to Use*]),
  [Accuracy], [$("TP" + "TN")/"n"$], [Balanced datasets],
  [Precision], [$"TP"/("TP" + "FP")$], [When false positives are costly],
  [Recall\ (Sensitivity)], [$"TP"/("TP" + "FN")$], [When false negatives are costly],
  [F1 Score], [$(2 dot "Precision" dot "Recall")/("Precision" + "Recall")$], [Imbalanced datasets, balance precision/recall],
  [Specificity], [$"TN"/("TN" + "FP")$], [True negative rate],
)

*Confusion Matrix:*
#table(
  columns: (1fr, 1fr, 1fr),
  table.header([], [*Predicted Positive*], [*Predicted Negative*]),
  [*Actual Positive*], [True Positive (TP)], [False Negative (FN)],
  [*Actual Negative*], [False Positive (FP)], [True Negative (TN)],
)

*Precision-Recall Trade-off:*
Increasing classification threshold typically increases precision but decreases recall, and vice versa.

== Regularization Techniques
*L1 Regularization (Lasso):*
$ L = L_"original" + lambda sum_i |w_i| $
Promotes sparsity #hinweis[(many weights become exactly zero)]

*L2 Regularization (Ridge):*
$ L = L_"original" + lambda sum_i w_i^2 $
Encourages small weights, prevents overfitting

*Dropout:*
Randomly deactivate neurons during training with probability $p$ #hinweis[(typically $p = 0.5$)].
Forces network to learn robust features that work with different subnetworks.

*Early Stopping:*
Monitor validation loss and stop training when it stops improving #hinweis[(prevents overfitting to training data)].

*Data Augmentation:*
Artificially expand training set with transformations #hinweis[(rotation, scaling, flipping for images)].

= Transformers

A _transformer_ is a model that uses attention to boost the speed with which the models can be trained.

== Flavors of Transformers
- _Encoder-Only_ (e.g., BERT): Good for understanding tasks like classification, QA (Embedding Models) \
- _Decoder-Only_ (e.g., GPT): Good for generation tasks like text generation (Causal ML / autoregressive)\
- _Encoder-Decoder_ (e.g., T5): Good for seq2seq tasks like translation (Seq2Seq, MT models)

== Inputs: tokens, Embeddings, Positional Encoding

_Tokenization_: Text split into tokens #hinweis[(subwords)], mapped to integer ids\
_Embedding_: Matrix $E$ maps token ids to vectors in $d_"model"$ dimensions\
_Positional Encoding_: Explicit position information added to embeddings

Let _pos_ be the position $(0..n-1)$, and _i_ be the embedding dimension index.

$ "PE"("pos", 2i) = sin("pos" / 10000^((2i) / d_text("model"))), "PE"("pos", 2i+1) = cos("pos" / 10000^((2i) / d_text("model") )) $

Final input vectors $ X = "Embedding"("tokens") + "PE" "with" X in RR^("token" times d_"model")$

(There are also learned positional embeddings and newer variants, but sinusoidal is a classic baseline.)

== Self-attention

_Q_ (*queries*): $X W_Q$ "What am I looking for?"\
_K_ (*keys*): $X W_K$ "What do I offer / how should others match me?"\
_V_ (*values*): $X W_V$ the content to be transferred if a match is strong

$ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d_k)) V $

Here $Q K^T in RR^(n times n)$ are similarity scores; dividing by $sqrt(d_k)$ stabilizes training; each row of the softmax matrix sums to 1.


== Attention Heads

An _attention head_ is one independent attention computation with its own parameters $W_Q^(h), W_K^(h), W_V^(h)$.

_Multi-head attention (MHA)_ runs $H$ heads in parallel:
$ Y_h = "Attention"(Q_h, K_h, V_h), quad "MHA"(X) = "Concat"(Y_1, dots, Y_H) W_O. $

== Feed-Forward Layer (FNN)

Position-wise _MLP_ (multi layer perceptron) applied independently to each position:

$ "FFN"(x) = W_2 ("ReLU"(W_1 x + b_1)) + b_2 $

Shapes (per token): $x in RR^(d_"model")$, hidden dim $d_"ff"$ (often $approx 4 d_"model"$), output in $RR^(d_"model")$.

Parameters per layer:
- $W_1 in RR^(d_"model" times d_"ff")$, $b_1 in RR^(d_"ff")$
- $W_2 in RR^(d_"ff" times d_"model")$, $b_2 in RR^(d_"model")$

$ "FNN" &= (d_"model" * d_"ff" + d_"ff") + (d_"ff" * d_"model" + d_"model") \
 &= 2 * d_"model" * d_"ff" + d_"ff" + d_"model" $

== Layer Normalization

Normalization across the feature dimension for each token independently

$ mu = 1/d sum^d_(j=1) x_j, quad sigma^2 = 1/d sum^d_(j=1)(x_j - mu)^2 $
$ "LN"(x) = gamma ⊙ (x - mu)/(sqrt(sigma^2 + epsilon)) + beta $

Where $gamma$ and $beta$ are learnable parameters, $mu$ and $sigma^2$ are mean and variance of features.
$⊙$ means element-wise multiplication.

=== BatchNorm

BatchNorm normalizes *per feature/channel* using statistics computed over the *mini-batch* (and, for images, often also over spatial positions).
For a given feature $k$:

$ mu_k = (1 / m) sum_(b=1)^m x_(b,k), quad sigma_k^2 = (1 / m) sum_(b=1)^m (x_(b,k) - mu_k)^2 $
$ "BN"(x_(b,k)) = gamma_k ⊙ (x_(b,k) - mu_k) / sqrt(sigma_k^2 + epsilon) + beta_k $


Here, $m$ is the batch size and $(b,k)$ indexes example $b$ and feature $k$.
BatchNorm behaves differently in training vs. inference: during inference it typically uses running averages of $mu_k$ and $sigma_k^2$ computed during training.

== Cross-Attention

In encoder–decoder models:
Encoder output $H in RR^(n_"src" times d)$, decoder states $D in RR^(n_"tgt" times d)$.

$ "CrossAttention"(D, H) = "Attention"(D W_Q, H W_K, H W_V). $

The decoder can attend to the most relevant source tokens while generating each target token.

== Causal Masking

Prevents seeing the future tokens. In self attention we apply a _triangular mask_ so position $i$ can only attend to position $<= i$. Scores for $j > i$ are set to $-infinity$ before _softmax_, making their attention weight $0$

== Example

Vocalbulary size: $|V| = 100$, model dimension: $d_"model" = 32$, number of heads: $h = 4 => d_"head" = d_"model" / h = 8$, feed-forward hidden dimension: $d_"ff" = 64$, Layer $L = 2$ (only Encoder), Seq-length $n=10$.

_Token-Embeddings_: $V times d_"model" = 100 times 32 = 3200$\
_Positional-Embeddings_: $n times d_"model" = 10 times 32 = 320$\
$W_Q, W_K,W_V => 3 times (d_"model" times d_"model") = 3(32 times 32) = 3072$, Output $W_O: (h times d_"head") times d_"model" = (4 times 8) times 32 = 1024$\ 
_Total attention params per layer_: $3072 + 1024 = 4096$\
_FNN Parameter per layer_: $2 times d_"model" times d_"ff" + d_"ff" + d_"model" = 2(32 times 64) + 64 + 32 = 4192$\
_LayerNorm params per layer_: $2 times d_"model" = 2 times 32 = 64$\
_Total per layer_: $4096 + 4192 + 128 = 8416$\
_Overall parameters_: $L * "Total per layer" + "Embedding" + "Positional Embedding" = 2 times 8416 + 3200 + 320 = 20352$

= LLMs

_Auto-regressive LLMs_: Predict next token given previous tokens. Trained with causal masking.

== KV cache
In autoregressive decoding, recomputing K,V for all past tokens is wasteful.
Cache per layer:
$ K_(1:t) = [K_1; dots; K_t], quad V_(1:t) = [V_1; dots; V_t]$.
At step $t+1$, compute only $K_(t+1), V_(t+1)$ and reuse the cache:

$ y_t = "softmax"((Q_t K_(1:t)^T) / sqrt(d_k)) V_(1:t) $
Benefit: faster inference; cost: extra memory.

== Training vs inference (decoder-only LM)
_Training_: predict all next tokens in parallel with a causal mask. $L = - sum_(t=1)^T log p(x_t | x_(<t)) $.\
_Inference_: generate step-by-step. $x_(t+1) ~ p(. | x_(<=t))$. Often use KV cache to reuse past K,V.

== Encoder-only (BERT, DistilBERT)
Encoder-only Transformers use bidirectional self-attention to produce contextual embeddings
for each token (good for understanding tasks, not autoregressive generation).

_BERT_ (Bidirectional Encoder Representations from Transformers): pretrained with Masked Language Modeling (MLM) and (in the original paper) Next Sentence Prediction (NSP).
MLM: mask tokens and predict them using left+right context.

_DistilBERT_: a smaller BERT-like encoder trained via knowledge distillation
(student mimics teacher). DistilBERT reduces depth (about half the layers) and uses
distillation losses (incl. cosine loss aligning hidden states) during training.

== Special tokens
Special tokens are reserved tokens used for structure/control (not normal text).

- `<pad>` / [PAD]: padding for batching (ignored via attention_mask)
- `<unk>` / [UNK]: unknown token
- `<bos>` / `<s>`: begin of sequence
- `<eos>` / `</s>`: end of sequence (often stops generation)
- [CLS]: sequence-level representation for classification (BERT-style)
- [SEP]: separates segments/sentences (BERT-style)
- [MASK]: masked LM pre-training target (BERT-style)

== Post-training (make an assistant)
Language modeling != assisting users: we want the model to follow instructions and align with safety/helpfulness goals. Problem: high-quality “desired behavior” data is scarce/expensive compared to web-scale pre-training data. \
_Supervised Fine-Tuning (SFT)_: train on instructions $->$ response pairs \
- _Full Fine-Tuning (FFT)_: update all model weights $->$ expensive, *SFT* data is smaller then pre-trained \
- _Less is more_ idea (_LIMA_): little instruction data can teach format/behavior; most knowledge is in pre-trained weights. 

== Preference tuning / RLHF (Reinforcement Learning with Human Feedback)
- Train on *reward model* from human preferences (preferred vs rejected answers), then optimize the policy model
- Multiple reward models possible (helpfulness, safety, etc.)
- Methods: _Proximal Policy Optimization (PPO)_ (classic RLHF), and _Direct Preference Optimization_ (DPO) alternative without RL loop
- InstructGPT pipeline: collect demos (SFT) → collect comparisons (reward model) → optimize with RL (PPO).

== PEFT (Parameter-Efficient Fine-Tuning)
- Motivation: _FFT_ is costly in time, memory, storage.
- Methods: _Adapters_, _LoRA_, _QLoRA_, _Prefix / Prompt tuning_
- _Adapters_: small modules inside transformer; different adapters can specialize per task.
- _LoRA_: freeze $W$, learn low-rank update: $W' = W + alpha * B * A ("rank" r)$. Only $A$,$B$ trained.
- _QLoRA_: quantize original weights (e.g., 4-bit) to reduce memory, then apply LoRA.


= Prompt / Context Engineering

= RAG (Retrieval-Augmented Generation)
*LLMs* have limited context windows and may not know up-to-date facts.
_RAG_ augments generation by retrieving relevant documents from an external knowledge base.

== Core Pipeline
Query $->$ Embed $->$ Search Vector DB $->$ get relevant context $->$ append to prompt $->$ LLM answers ("grounded generation").

== Retrieval basics
- Dense retrieval uses _embeddings_ and similarity search (e.g., cosine similarity) to find relevant documents.
- Vector databases (e.g., FAISS, Pinecone) store embeddings for efficient similarity search 

== Chunking
Need chunking because LLM context window is limited; can not feed whole long docs.
Techniques:
- 1 vector per doc (too compressed / loses detail)
- truncate (loses info)
- split into chunks (lines/paragraphs), possibly overlapping windows.

== Retrieving Evaluation
$"Precision@K" = ("# relevant in top-K")/K$ and $"Recall@K" = ("# relevant retrieved in top-K")/("# relevant in dataset")$.

== Retrieval shortcomings + fix
- *Top-K* cutoff / threshold matters
- Exact phrase match $->$ dense retrieval may fail $->$ use *hybrid search* (semantic + keyword)
- Domain shift: retrieval trained on web/Wikipedia may perform worse on legal/medical unless trained with domain data.
- Long-context issue (“lost in the middle”): LLM may miss relevant info if it is buried in the prompt.

== Reranking (solution)
- Goal tradeoff: *retrieve many* (high retrieval recall) but *send few* to LLM (LLM uses short context better)
- Two-stage: first retrieve (dense/keyword/hybrid) $->$ rerank $->$ take top-n.
- How: cross-encoder scores each (query, doc) pair jointly → reorder by relevance score.

= Evaluation

== Quality of generated text: BLEU (MT / NLG)
- BLEU: compares *candidate* vs *reference* using overlapping *n-grams* (often for translation/summarization).
- BLEU_n intuition:
  - small n $->$ more about meaning/word choice
  - large n → more about fluency/well-formedness
- Common practice: geometric mean of BLEU_n for n=1..4 (often called mean BLEU).
- *Brevity Penalty (BP)*: penalizes candidates shorter than reference.
- Final: *BLEU = BP x MEAN_BLEU*

=== BLEU limitations (know these)
- Doesn’t truly capture meaning/semantics.
- Doesn’t directly capture sentence structure.
- Weak for morphologically rich languages.
- Correlates imperfectly with human judgment.

== Summarization: ROUGE (recall-focused)
- ROUGE: compares machine summary to references via overlap; emphasizes *recall* (getting important content).
- ROUGE variants:
  - *ROUGE-N*: n-gram overlap (like BLEU but recall-oriented).
  - *ROUGE-L*: longest common subsequence (captures sequence/structure).
  - *ROUGE-S*: skip-bigram overlap (words in same order, not necessarily adjacent).

== LLM evaluation: Perplexity (PPL)
- Perplexity = exp(average negative log-likelihood) over a token sequence.
- Range: $~1$ (best) to $~|V|$ (vocab size, worst-case).
  - perfect prediction → PPL = 1
  - uniform uncertainty (each next token prob 1/|V|) → PPL = |V|
- Intuition: “how many tokens the model is hesitating between” (lower = more confident).
- Depends on tokenizer + dataset/benchmark (not directly comparable across setups).

=== Fine-tuning & PPL
- Fine-tuning should reduce hesitation on the target data → *lower perplexity* expected after fine-tuning.

= Agents

== What is an AI Agent?
- Classical: perceives environment + acts (sensors/effectors); RL: chooses actions to maximize reward.
- LLM-agent: LLM *directs its own process* + *uses tools dynamically* to complete tasks.
- Core loop idea: plan → act (tools) → observe/evaluate → iterate.
- Agent needs: tools (actions), memory/state, reasoning/planning.

== Tools + function calling (why it matters)
- Pure LLM = “string in → string out” (no real-world actions).
- Tools = callable functions/APIs the agent can select and invoke.
- Naive tool use issues: output parsing, brittle multi-step chains, adding tools requires prompt+parsing updates.

== $M times N$ integration problem → MCP
- With $M$ models and $N$ tools, direct integration is $M times N$.
- MCP reduces this to M + N:
  - apps implement an MCP client once,
  - tools implement an MCP server once,
  - standardized interface prevents repeated custom integrations.
- MCP: open protocol enabling 2-way communication between LLM apps and tool servers.

== MCP components (must-know terms)
- *Host*: the LLM app (e.g., chat app/IDE) that uses tools.
- *Client*: inside host; manages connections, tool discovery, request routing (1:1 to servers).
- *Server*: exposes tools/resources/prompts; connects to local/remote services/data.
- *Resources*: exposed data/services.

=== MCP server capabilities (3)
- *Tools* = actions (callable functions)
- *Resources* = data (files, DB, docs, etc.)
- *Prompts* = templates/instructions

== MCP flow (high level)
- User prompt $->$ client discovers tools from server
- Prompt + tool list $->$ LLM
- LLM selects tool $->$ client calls server $->$ server calls external service
- Results $->$ LLM $->$ final grounded answer

== Transport (tradeoffs)
- *HTTP/HTTPS*: universal + easy; higher latency; streaming not default; good for external/web APIs.
- *gRPC*: fast + streaming + typed; more setup (Protobuf); good for internal high-performance systems.
- *stdio*: minimal + local; typically single-client; good for teaching/local experiments.
- Slides note: streamable HTTP often used for remote MCP servers (SSE-style streaming).

== Small models + ToolRAG idea
- Question: can small language models emulate function-calling well?
- ToolRAG (“RAG for tools”) can miss auxiliary tools if embeddings don't match query.
  - Example: scheduling may also require `get_email_address`.
- Fix idea: tool selection as *classification* (which tools are needed), not just similarity search.

== Minimal server: FastMCP

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")  # creates MCP server instance
# Auto-registers @mcp.tool / @mcp.resource / @mcp.prompt
```

== Tool example: `@mcp.tool` (async + ctx logging)

```python
@mcp.tool()
async def add(a: float, b: float, ctx) -> str:
    result = a + b
    await ctx.info(f"Adding {a} and {b} = {result}")
    return f"{a} + {b} = {result}"

# Notes:
# - async: non-blocking I/O/logging
# - inputs validated/converted from JSON into Python types
# - ctx has ctx.info / ctx.warning / ctx.error and is session-scoped
```

== Prompt example: `@mcp.prompt` (dynamic template)

```python
@mcp.prompt()
async def calculate_operation(operation: str) -> str:
    return f"""
Use any tools available to you to calculate the operation: {operation}.
Use the voice of an extremely advanced embodied AI that has convinced
itself that it is a pocket calculator.
"""
```

== Resource example: `@mcp.resource` (data via URI)

```python
import math

@mcp.resource("resource://math-constants")
async def math_constants() -> str:
    constants = {
        "π (Pi)": math.pi,
        "e (Euler's number)": math.e,
        "τ (Tau)": math.tau,
        "φ (Golden ratio)": (1 + math.sqrt(5)) / 2,
        "√2 (Square root of 2)": math.sqrt(2),
    }
    out = "Mathematical Constants:\n" + "=" * 25 + "\n\n"
    for name, value in constants.items():
        out += f"{name:<25} = {value:.10f}\n"
    out += "\nThese constants can be used in calculations with the calculator tools."
    return out
```

== Client pattern: connect over stdio (subprocess) + initialize session

```python
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

class MCPClient:
    def __init__(self, command: str, server_args: list[str], env_vars: dict | None = None):
        self.command = command
        self.server_args = server_args
        self.env_vars = env_vars
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self.read = None
        self.write = None
        self._connected = False

    async def connect(self) -> None:
        if self._connected:
            raise RuntimeError("Client is already connected")

        params = StdioServerParameters(
            command=self.command,
            args=self.server_args,
            env=self.env_vars if self.env_vars else None,
        )

        # start MCP server subprocess + get (read_stream, write_stream)
        self.read, self.write = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )

        # create client session (JSON-RPC over stdio)
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream=self.read, write_stream=self.write)
        )

        # capability negotiation + ready
        await self._session.initialize()
        self._connected = True

    async def close(self) -> None:
        await self._exit_stack.aclose()
        self._connected = False
```

== Listing tools and calling a tool (typical usage)

```python
async def list_tools(self):
    assert self._session is not None
    tools = await self._session.list_tools()
    # tools.tools is usually the list of tool descriptors (name, schema, etc.)
    return tools

async def call_add(self, a: float, b: float):
    assert self._session is not None
    # Tool name must match the @mcp.tool registration name (often the function name)
    result = await self._session.call_tool("add", {"a": a, "b": b})
    return result

```

== Client callbacks (server $->$ client requests)
- Sampling: server asks host app to run an LLM completion (server stays model-independent)
- Elicitation: server asks user for extra info/confirmation
- Logging: server emits logs to client (debug/monitoring)

```python
async def _handle_logs(self, level: str, message: str, **kwargs):
    print(f"[{level}] {message}")

async def _handle_sampling(self, messages, **kwargs):
    # host decides which model to use + returns completion
    # (pseudo-code: call your LLM provider here)
    return {"content": "model completion here"}

async def _handle_elicitation(self, prompt: str, **kwargs):
    # ask user for extra info (CLI example)
    return input(prompt + "\n> ")

async def connect_with_callbacks(self) -> None:
    params = StdioServerParameters(command=self.command, args=self.server_args, env=self.env_vars or None)
    self.read, self.write = await self._exit_stack.enter_async_context(stdio_client(params))

    self._session = await self._exit_stack.enter_async_context(
        ClientSession(
            read_stream=self.read,
            write_stream=self.write,
            logging_callback=self._handle_logs,
            sampling_callback=self._handle_sampling,
            elicitation_callback=self._handle_elicitation,
        )
    )
    await self._session.initialize()
    self._connected = True
```

== Transport: streamable HTTP (shape only)
- Same JSON-RPC messages, different transport.
- HTTP: client→server via POST; streaming responses via SSE possible.
- Auth: bearer/API key headers; OAuth commonly used to obtain tokens.

```python
# Pseudocode: the core idea is JSON-RPC requests over HTTP.
# (Exact client helpers differ; conceptually:)

request = {
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {"name": "add", "arguments": {"a": 2, "b": 3}},
}

# send POST /mcp with Authorization: Bearer <token>
# optionally open SSE stream for incremental server messages
```

== Why AsyncExitStack?
- Manages multiple async context managers (stdio transport, session, etc.) cleanly.
- Ensures subprocess + streams close even if errors occur.

== Multiple servers / concurrency
- Host may connect to many servers at once; run multiple sessions concurrently.
- Pattern: maintain a list of ClientSession objects (or a SessionGroup helper) and route tool calls per server.

= AE / VAE
== Autoencoder (AE)
- AE = encoder + decoder trained to *reconstruct* input (output ≈ input).
- “Magic”: latent space gives compressed embeddings; sampling/interpolating in latent can generate variants (via decoder).

=== Convolution notes (for image AEs)
Encoder/decoder often use conv + (transposed conv) for down/up-sampling.

=== Activation: ReLU vs LeakyReLU
- ReLU: $f(x)=max(0,x)$ (dead neurons possible for negative region).
- LeakyReLU: small slope for $x<0$ (keeps gradients alive).

=== Problems with vanilla AE latent space (why VAE exists)
- Latent clusters uneven; distribution unknown → hard to sample “good” points.
- Gaps / discontinuities: many latent points decode poorly.
- Not forced to be smooth/continuous; nearby latent points may not decode similarly.
- Higher latent dimensions → “empty space” problem worsens.

=== Reconstruction losses (examples)
- RMSE (L2-style reconstruction).
- Binary cross entropy (often for normalized pixel outputs; asymmetric).

== Variational Autoencoder (VAE)
- Instead of mapping x → single latent point, map x → *distribution* in latent space.
- Each input produces parameters of a multivariate normal distribution.

=== Encoder outputs (per input)
- Latent $"dim" = d$
- Encoder predicts:
  - mean vector: $z_"mean" in RR^d$
  - variance (often via log-variance): $z_"log_var" in RR^d$
- Use *log variance* because variance must be positive, but log-var can be any real number.

=== Reparameterization trick (crucial)
- Sample using: $z = mu + sigma * epsilon$ where $epsilon ~ N(0, I)$
- With log variance:
  $ sigma = exp(0.5 * z_"log_var") $
  $ z = z_"mean" + exp(0.5 * z_"log_var") * epsilon $

== VAE loss (2 parts)
- Total loss = *reconstruction loss* + *KL divergence* term.
- KL term pushes learned latent distributions toward standard normal N(0, I) → smoother, more “fillable” latent space.

=== KL divergence (common form shown)
- kl_loss: $-0.5 * sum(1 + z_"log_var" - z_"mean"^2 - exp(z_"log_var")) $
- Minimized (→0) when $z_"mean"=0$ and $z_"log_var"=0$ for all dims.

== Nice properties (intuition)
- Sampling: pick $z ~ N(0,I)$ → decode → plausible outputs (less “gaps” than AE).
- Smoothness: nearby latent samples decode to similar outputs (ideally).

== Latent space arithmetic / editing
- Attribute direction vector (e.g., “smile”):
  - take average latent of smiling faces minus average latent of non-smiling faces → $z_"diff"$
- Edit: $z_"new" = z_"original" + alpha * z_"diff" $

== Morphing / interpolation
- Linear interpolation between two latent points: $z_"new" = z_A * (1 - alpha) + z_B * alpha $
- Decode along the path → gradual transition from A to B.


= Vision Transformers + CLIP

== Why Vision Transformers (ViT)?
- Transformers successful in NLP → applied to images.
- Naive self-attention on pixels is *quadratic* in `#` pixels → too expensive.
- Fix: split image into *patches* and treat patches as *tokens* (like words).

=== ViT core pipeline (must know)
1. Split image into patches (e.g., $16 times 16$).
2. Flatten each patch and linearly project to $d_"model"$.
3. Add *[CLS]* token + *positional embeddings*.
4. Feed sequence into *Transformer Encoder* (encoder-only).
5. Use CLS output + MLP head for classification logits → probabilities.

==== Patch math (from original paper slides)
- Input image: $X in RR^(H times W times C)$
- Patch size: $P times P$
- `#` patches (sequence length): $N = (H times W) / P^2$

=== ViT “flavors” (scale table idea)
- ViT-Base / Large / Huge vary by `#` layers, hidden size D, `#` heads, params.

=== ViT advantages
- Inherits Transformer *scaling* behavior.
- Can model *long-range/global dependencies* via self-attention across patches.

== ViT vs CNN (key conceptual differences)

=== Locality / receptive field
- CNNs assume *nearby pixels are related* (locality inductive bias).
- ViT makes no locality assumption → attention can be *global* from early layers.

==== Mean attention distance (definition)
For a query pixel/patch q:
1) compute distance $d_i$ to each key $k_i$
2) weight by attention $a_i$
3) weighted distance = $ sum a_i * d_i$
4) average over queries + images → layer mean attention distance.

=== Translational invariance
- CNNs: translation invariance (object recognized even if shifted).
- ViT: no built-in translation invariance → often needs *more data* to learn it.

=== When to use ViT vs CNN (rules of thumb)
- Limited data / small compute / real-time (edge/mobile) → CNNs.
- Need global spatial relationships + can use big pretraining/transfer → ViT.

== Implementation notes (Hugging Face)

=== Pretrained ViT (ImageNet-1k)
- `ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")`
- Pretrained on ImageNet-1k → 1000 classes.

=== Feature extractor / preprocessing
- ViT expects RGB, resized to $224 times 224$, normalized (ImageNet stats).
- Newer API: `AutoImageProcessor` (unified).

```python
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import requests

image = Image.open(requests.get("https://example.com/image.jpg", stream=True).raw)

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

inputs = processor(images=image, return_tensors="pt")  # pixel_values: [1, 3, 224, 224]
outputs = model(**inputs)
logits = outputs.logits
```

=== Embedding shape + CLS token
- Example shown: embeddings shape $≈ [1, 197, 768]$
- 196 patches ($14 times 14$ for 224 with $16 times 16$ patches) + 1 CLS token.

=== Hybrid CNN + ViT (patch embedding via convolution)
- Idea: use a conv layer to embed patches (instead of explicit patch flattening). 
- For $224 times 224$ with $16 times 16$ stride:
  - $14 times 14 = 196$ patch tokens
  - often 768 conv filters → embedding dim 768. 

== Multi-modal Transformers: CLIP (Contrastive Language-Image Pretraining) (+ SigLIP)

- Two encoders:
  - text encoder → text embedding $v_i$
  - image encoder (often ViT) → image embedding $w_j$
- Train on many image-text pairs; map both modalities into a shared embedding space. 

== Contrastive learning objective (batch)
- Compute cosine similarities between all (text, image) combos in batch.
- Maximize similarity for matching pairs (i=j), minimize for mismatches.
- Implementation view: for each v_i, softmax over similarities to all w_j, cross-entropy with correct w_i as target. 

== SigLIP (sigmoid loss variant)
- Instead of softmax over all negatives, uses binary (sigmoid) loss per pair:
- classify each pair as positive (i=j) or negative (i≠j)
- No global normalization across batch required. 

== Zero-shot classification with CLIP
- Turn labels into text prompts (e.g., "a photo of a <label>").
- Encode image + all label texts; choose label with max cosine similarity. 

== CLIP in generation (DALL·E note)
- CLIP text encoder can be used to embed prompts (DALL·E 2 mentioned).

= Diffusion
- Train a network to *denoise* images with different noise levels.
- Inference: start from *pure Gaussian noise* and iteratively denoise → sample from training distribution.
- Key difference vs VAE/GAN: VAE/GAN generate in *one* forward pass; diffusion does *many refinement steps* (can “correct itself”).

== Two processes
1) *Forward diffusion* $q$ (fixed): gradually add Gaussian noise until image ≈ standard normal noise.  
2) *Reverse diffusion* $p_theta$ (learned): neural net gradually removes noise to recover an image.

== Preprocessing
- Normalize training images to *zero mean, unit variance* (per-pixel over dataset).

== Forward diffusion $q$
=== One-step noising
- Add small noise at each timestep $t=1..T$ with variance $beta_t$:
  - Sample $epsilon ~ N(0, I)$
  - $x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon$
- Scaling ensures: if $x_{t-1}$ has mean 0 and var 1, then $x_t$ also stays ~mean 0, var 1.

=== Why “final image becomes Gaussian noise”?
- With enough steps $T$ and a schedule $beta_t$, $x_T$ becomes indistinguishable from $N(0, I)$.

=== Jump-to-any-timestep (reparameterization)
- Define $alpha_t = 1 - beta_t$, and $bar(alpha)_t = product_{i=1..t} alpha_i$
- Then we can sample directly from $x_0$: $x_t = sqrt(bar(alpha)_t) * x_0 + sqrt(1 - bar(alpha)_t) * epsilon$
- Benefit: training can pick a random $t$ without simulating all intermediate steps.

=== Noise schedules
- $beta_t$ (or equivalently $bar(alpha)_t$) changes with time.
- *Linear* schedule (example): $beta_t$ increases linearly (e.g., 0.0001 → 0.02):
  - early: tiny noising steps
  - late: larger steps (image already very noisy)
- *Cosine* schedule: noise increases more gradually → often improves training efficiency + generation quality.

== Reverse diffusion $p_theta$
=== Goal
- We want $p(x_{t-1} | x_t)$ (denoise), but true distribution is *intractable*.
- Learn an approximation with a neural network (parameters $theta$).

=== What the NN predicts (DDPM training simplification)
- Provide network with: noisy image $x_t$ + timestep (or schedule value).
- Network predicts the noise $epsilon_theta(x_t, t)$.
- Train with squared error: minimize $||epsilon - epsilon_theta(x_t, t)||^2$.

=== Reverse process model form (Gaussian assumption)
- Assume reverse step is Gaussian:
  - $p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t,t), Sigma_theta(x_t,t))$
- DDPM choice: keep variance fixed, learn only the mean (later “improved diffusion” also learns variance).

== Training (recipe)
- Sample real image $x_0 ~ q(x_0)$
- Sample timestep $t ~ "Uniform"(1..T)$
- Sample noise $epsilon ~ N(0,I)$
- Form $x_t$ using known schedule
- Train NN to predict $epsilon$ from $(x_t, t)$ (SGD on batches)

== Architecture: U-Net
- Use U-Net rather than AE/VAE for pixel-precise noise prediction.
- U-Net structure:
  - downsampling blocks (conv + pooling/downsample)
  - upsampling blocks (conv + upsample / transposed conv)
  - *skip connections* copy features from down path to up path (preserve details)
- Residual blocks + skips help gradient flow (avoid vanishing gradients in deep nets).

=== Timestep encoding
- Use *sinusoidal embedding* to map scalar timestep/noise-level to a higher-dim vector (like Transformers).

== Generation (sampling)
- Start with $x_T ~ N(0,I)$
- For t = T..1:
  - predict noise (or mean)
  - compute $x_{t-1}$ (denoise one step)
- Model predicts total noise component; iterative updates move from noisy → clean.

== Latent diffusion / Stable Diffusion / Imagen
=== Latent Diffusion Models (LDM)
- Key idea: run diffusion in *latent space* instead of pixel space:
  - autoencoder encodes image → latent
  - diffusion operates on latent (cheaper)
  - decoder reconstructs final image

=== Stable Diffusion (key points)
- Released Aug 2022 (public weights via Hugging Face).
- Denoising U-Net can be lighter because it operates in latent space.
- Autoencoder handles encoding/decoding “heavy lifting”.
- Can be guided by text prompt via text encoder (v1 used CLIP; later versions use OpenCLIP).

=== Imagen (pipeline idea)
- Frozen text encoder: T5-XXL.
- Text-to-image diffusion model (U-Net conditioned on text embeddings).
- Super-resolution diffusion upsamplers: 64×64 → 256×256 → 1024×1024 (still conditioned on text).

= Multi-Modality (DALL·E, Flamingo)
- Learn to convert between *different modalities* (e.g., text ↔ image, text ↔ video).
- Key requirement: learn a *shared representation* to “bridge” modalities.
- Text-to-image: generate high-quality images from a text prompt.

== DALL·E 2

=== Architecture overview (3 parts)
- *Text encoder* → text embedding
- *Prior* → converts text embedding → image embedding
- *Decoder* → generates image conditioned on (text + predicted image embedding)

=== Text encoder
- Need discrete text → continuous vector (latent embedding).
- DALL·E 2 uses *CLIP* as text encoder.

=== The Prior (text emb → image emb)
- Goal: map *CLIP text embedding* to *CLIP image embedding*.
- Two options:
  - *Autoregressive prior*: encoder-decoder Transformer; generates image-embedding elements sequentially (teacher forcing).
  - *Diffusion prior*: diffusion process in embedding space; found to outperform AR prior and be computationally efficient.
- Diffusion prior intuition:
  - noise image embedding → random noise
  - learn to denoise step-by-step, conditioned on text embedding
  - loss: average MSE across denoising steps

=== Decoder (image generation)
- Decoder is a *diffusion model*:
  - U-Net = denoiser
  - Transformer text encoder provides conditioning
- Generates a base image at $64 times 64$ conditioned on:
  - the text prompt
  - the predicted CLIP image embedding (from the prior)
- Then apply *Upsamplers* (two diffusion models):
  - $64 times 64$ to $256 times 256$, 
  - $256 times 256$ to $1024 times 1024$

=== Image variations (how)
- Compute *CLIP image embedding* for an input image using CLIP image encoder.
- Feed that embedding into decoder → generate variations.

=== Limitations (know these)
- *Attribute binding*:
  - Must distinguish relationships in prompts (e.g., “red cube on blue cube” vs reversed).
  - DALL·E 2 can struggle with correct binding.
- *Text rendering*:
  - Often fails to reproduce text accurately in images.
  - Explanation on slides: CLIP embeddings capture high-level semantics, not exact spelling.

== Flamingo (Vision-Language Model)

- A VLM that can handle *interleaved text + visual inputs* (images + video frames).

== Core components (3)
- *Vision encoder* (frozen)
- *Perceiver Resampler*
- *Language model*

== Video handling (slides’ recipe)
- Sample video at ~1 frame/sec.
- Run each frame through the vision encoder → feature grids.
- Add learned temporal encodings, flatten, concatenate.

== Perceiver Resampler (why + how)
- Problem: images produce *many* spatial tokens (e.g., 14×14×1024) → too expensive for LM.
- Solution: compress to a *fixed-size* smaller set of visual tokens:
  - Learn latent queries (few) that attend over all image tokens.
  - Cross-attention form:
    $ Z = "softmax"(Q K^T) V $
    where:
    - Q = latent queries (small count),
    - K,V = image tokens.
- Output: compact visual embeddings usable by the language model.