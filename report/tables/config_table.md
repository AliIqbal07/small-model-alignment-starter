| Setting | SFT | DPO-HH | DPO-UF | Persona LoRA |
|---|---|---|---|---|
| Init model | Qwen/Qwen2.5-0.5B | /kaggle/input/notebooks/aliiqbal1/pa2-02-sft/pa2/outputs/sft/best | /kaggle/input/notebooks/aliiqbal1/pa2-02-sft/pa2/outputs/sft/best | HuggingFaceTB/SmolLM2-360M-Instruct |
| Trainable params | full fine-tuning | full fine-tuning | full fine-tuning | LoRA adapter only |
| Train / val examples | 30000 / 1000 | 15000 / 1000 | 15000 / 1000 | 200 / 20 |
| Max length | 512 | 512 | 512 | 512 |
| Max prompt length | 256 | 256 | 256 | 256 |
| Batch size | 2 | 1 | 1 | 4 |
| Grad accum | 16 | 32 | 32 | 4 |
| Effective batch | 32 | 32 | 32 | 16 |
| Optimizer | adamw | rmsprop | rmsprop | adamw (TRL default) |
| Learning rate | 2e-05 | 1e-06 | 1e-06 | 5e-05 |
| Weight decay | 0.1 | — | — | — |
| Warmup | 3% of steps | 50 steps | 50 steps | — |
| LR schedule | linear warmup + cosine to 0 | constant after linear warmup | constant after linear warmup | linear decay (TRL default) |
| Grad clipping | max_grad_norm 1 | max_grad_norm 1 | max_grad_norm 1 | max_grad_norm 1 (TRL default) |
| Beta | — | 0.1 | 0.1 | 0.1 |
| Epochs | 1 | 1 | 1 | 3 |
| Max steps | — | — | — | — |
| Precision | fp16 autocast, fp32 weights | fp16 autocast, fp32 weights | fp16 autocast, fp32 weights | fp16 autocast, fp32 weights |
| Reference model | — | /kaggle/input/notebooks/aliiqbal1/pa2-02-sft/pa2/outputs/sft/best (frozen copy) | /kaggle/input/notebooks/aliiqbal1/pa2-02-sft/pa2/outputs/sft/best (frozen copy) | base model with LoRA adapter disabled |
| Reference dtype | — | float32 | float32 | float32 |
| LoRA r / alpha / dropout | — | — | — | 16 / 32 / 0.05 |
| LoRA targets | — | — | — | q_proj, k_proj, v_proj, o_proj |
| Seed | 42 | 42 | 42 | 42 |
