| Comparison | Eval set | n | Wins | Losses | Ties | Unknown | Failures | Win rate (excl. ties) | Win rate (ties = 0.5) | 95% CI (ties = 0.5) |
|---|---|---|---|---|---|---|---|---|---|---|
| dpo_hh vs sft (hh) | hh_rlhf_eval_300 | 300 | 163 | 114 | 15 | 8 | 0 | 58.8% | 58.4% | [52.9%, 63.9%] |
| persona_lora vs base (persona_safety) | persona_style_40 | 35 | 15 | 14 | 4 | 2 | 5 | 51.7% | 51.5% | [34.8%, 68.2%] |
| persona_lora vs base (persona_style) | persona_style_40 | 35 | 11 | 9 | 13 | 2 | 5 | 55.0% | 53.0% | [39.4%, 66.7%] |
| dpo_uf vs sft (uf) | alpaca_eval_300 | 300 | 125 | 134 | 28 | 13 | 0 | 48.3% | 48.4% | [43.0%, 53.5%] |

Wins / losses / win rates are for the first model named in each comparison.
