#import "../../template_zusammenf.typ": *

= Post-Training & Fine-Tuning

== Post-Training (Make an Assistant)

Language modeling != assisting users:
- Pretraining: predicts next token well, but may not follow instructions reliably
- Post-training aligns behavior toward helpfulness, instruction-following, and safety goals

Problem:
- High-quality “desired behavior” data is scarce / expensive compared to web-scale pretraining data

_Supervised Fine-Tuning (SFT):_ Train on instructions → response pairs
- _Full Fine-Tuning (FFT):_ update all model weights → expensive, SFT data smaller than pretrained
- _Less is more idea (LIMA):_ little instruction data can convert format/behavior; most knowledge in pre-trained weights


== Preference Tuning / RLHF

*Reinforcement Learning with Human Feedback (RLHF):*
1. Collect preference data: human chooses preferred vs rejected answers
2. Train a reward model $r_phi(x, y)$ to score outputs
3. Optimize the policy model to maximize reward (and stay close to a reference model)

Notes:
- Multiple reward models can exist (helpfulness, harmlessness, style, etc.)
- Preference tuning aims to match human preferences beyond supervised demos

*Methods:*
- _PPO (Proximal Policy Optimization)_: classic RLHF optimizer (policy-gradient with constraints)
- _DPO (Direct Preference Optimization)_: alternative that fits preferences without an explicit RL loop

*InstructGPT-style pipeline (high level):*
SFT demos -> reward model from comparisons -> optimize with RL (often PPO)

== PEFT (Parameter-Efficient Fine-Tuning)
- Motivation: _FFT_ is costly in time, memory, storage.
- Methods: _Adapters_, _LoRA_, _QLoRA_, _Prefix / Prompt tuning_
- _Adapters_: small modules inside transformer; different adapters can specialize per task.
- _LoRA_: freeze $W$, learn low-rank update: $W' = W + alpha * B * A ("rank" r)$. Only $A$,$B$ trained.
- _QLoRA_: quantize original weights (e.g., 4-bit) to reduce memory, then apply LoRA.

*Key difference:*
- LoRA: reduces trainable parameters
- QLoRA: reduces trainable parameters *and* base-model memory usage

=== Prefix / Prompt Tuning
- Learn additional trainable “prompt” vectors (or prefix key/value vectors)
- Keep backbone frozen
- Useful when you want minimal updates and fast task switching

#hinweis[
Rule of thumb: Use PEFT (LoRA/QLoRA/adapters) when you need task adaptation with tight memory/storage budgets; use FFT when you can afford full updates and want maximal capacity to shift behavior.
]