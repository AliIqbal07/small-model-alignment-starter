| Setting | Value |
|---|---|
| Decoding | greedy (do_sample=False, so temperature / top-p are not used) |
| max_new_tokens | 512 |
| repetition_penalty | 1.1 |
| no_repeat_ngram_size | 0 (off) |
| Batch size | 16 |
| Prompt format | each model's own tokenizer chat template, one user turn, add_generation_prompt=True; no system message added by us; left padding |
| Chat template (Qwen2.5 runs) | SFT, DPO-HH, DPO-UF: Qwen chat template, which inserts the default system prompt "You are a helpful assistant." |
| Persona comparison | SmolLM2-360M-Instruct base vs. base + merged LoRA: SmolLM2's own chat template and the default system prompt it inserts, with the same greedy settings |
| Stop tokens | tokenizer eos_token_id + `<\|im_end\|>` (deduplicated) |
| Output text | new tokens only, special tokens skipped, whitespace stripped |

The same settings are used for every model in a comparison.
