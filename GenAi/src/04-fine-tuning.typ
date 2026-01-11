
= Post-Training & Fine-Tuning

== Post-Training (Make an Assistant)
Language modeling ≠ assisting users: want model to follow instructions and align with safety/helpfulness goals.

*Problem:* High-quality "desired behavior" data is scarce/expensive compared to web-scale pre-training data.

*Supervised Fine-Tuning (SFT):* Train on instructions → response pairs
- *Full Fine-Tuning (FFT):* update all model weights → expensive, SFT data smaller than pretrained
- *Less is more idea (LIMA):* little instruction data can convert format/behavior; most knowledge in pre-trained weights

== Preference Tuning / RLHF
*Reinforcement Learning with Human Feedback:*
- Train reward model from human preferences (preferred vs rejected answers)
- Optimize policy model
- Multiple reward models possible (helpfulness, safety, etc.)
- *Methods:* Proximal Policy Optimization (PPO) (classic RLHF), Direct Preference Optimization (DPO) alternative without RL loop
- *InstructGPT pipeline:* collect demos (SFT) → collect comparisons (reward model) → optimize with RL (PPO)

== PEFT (Parameter-Efficient Fine-Tuning)
*Motivation:* FFT is costly in time, memory, storage.

*Methods:* Adapters, LoRA, QLoRA, Prefix/Prompt tuning

*Adapters:* Small modules inside transformer; different adapters can specialize per task.

*LoRA (Low-Rank Adaptation):*
- Freeze $W$, learn low-rank update: $W' = W + alpha * B * A$ (rank $r$)
- Only $A$, $B$ trained
- Strong performance with few trainable parameters

*QLoRA (Quantized LoRA):*
- Extension of LoRA with weight quantization
- Frozen base model stored in low precision (e.g., 4-bit)
- LoRA adapters trained in higher precision
- Enables fine-tuning very large LLMs on limited hardware

*Key Difference:*
- LoRA: reduces number of trainable parameters
- QLoRA: reduces both trainable parameters and memory usage