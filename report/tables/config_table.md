| Setting | SFT | DPO-HH | DPO-UF | Persona LoRA |
|---|---|---|---|---|
| Init model | Qwen/Qwen2.5-0.5B | /kaggle/input/notebooks/aliiqbal1/pa2-02-sft/pa2/outputs/sft/best | — | — |
| Trainable params | full fine-tuning | full fine-tuning | — | — |
| Train / val examples | 30000 / 1000 | 15000 / 1000 | — | — |
| Max length | 512 | 512 | — | — |
| Max prompt length | 256 | 256 | — | — |
| Batch size | 2 | 1 | — | — |
| Grad accum | 16 | 32 | — | — |
| Effective batch | 32 | 32 | — | — |
| Optimizer | adamw | rmsprop | — | — |
| Learning rate | 2e-05 | 1e-06 | — | — |
| Weight decay | 0.1 | — | — | — |
| Warmup | 3% of steps | 50 steps | — | — |
| LR schedule | linear warmup + cosine to 0 | constant after linear warmup | — | — |
| Grad clipping | max_grad_norm 1 | max_grad_norm 1 | — | — |
| Beta | — | 0.1 | — | — |
| Epochs | 1 | 1 | — | — |
| Max steps | — | — | — | — |
| Precision | fp16 autocast, fp32 weights | fp16 autocast, fp32 weights | — | — |
| Reference model | — | /kaggle/input/notebooks/aliiqbal1/pa2-02-sft/pa2/outputs/sft/best (frozen copy) | — | — |
| Reference dtype | — | float32 | — | — |
| LoRA r / alpha / dropout | — | — | — | — |
| LoRA targets | — | — | — | — |
| Seed | 42 | 42 | — | — |
