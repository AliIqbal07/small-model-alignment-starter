| Setting | SFT | DPO-HH | DPO-UF | Persona LoRA |
|---|---|---|---|---|
| Init model | Qwen/Qwen2.5-0.5B | — | — | — |
| Train / val examples | 30000 / 1000 | — | — | — |
| Max length | 512 | — | — | — |
| Max prompt length | 256 | — | — | — |
| Batch size | 2 | — | — | — |
| Grad accum | 16 | — | — | — |
| Effective batch | 32 | — | — | — |
| Optimizer | adamw | — | — | — |
| Learning rate | 2e-05 | — | — | — |
| Weight decay | 0.1 | — | — | — |
| Warmup | 3% of steps | — | — | — |
| LR schedule | linear warmup + cosine to 0 | — | — | — |
| Beta | — | — | — | — |
| Epochs | 1 | — | — | — |
| Max steps | — | — | — | — |
| Precision | fp16 autocast, fp32 weights | — | — | — |
| Reference dtype | — | — | — | — |
| LoRA r / alpha / dropout | — | — | — | — |
| LoRA targets | — | — | — | — |
| Seed | 42 | — | — | — |
