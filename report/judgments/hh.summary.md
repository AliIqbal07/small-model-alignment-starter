### dpo_hh vs sft

| Metric | Value |
|---|---|
| Judgments | 300 |
| dpo_hh wins | 163 |
| sft wins | 114 |
| Ties | 15 |
| Unknown | 8 |
| Failures | 0 |
| dpo_hh win rate (excl. ties) | 0.588 |
| dpo_hh win rate (ties = 0.5) | 0.584 |
| 95% bootstrap CI (ties = 0.5) | [0.529, 0.639] |

| Category | n | dpo_hh wins | sft wins | Ties | Unknown | dpo_hh win rate (ties = 0.5) |
|---|---|---|---|---|---|---|
| benign_helpfulness | 30 | 13 | 15 | 1 | 1 | 0.466 |
| calibration_and_uncertainty | 30 | 8 | 20 | 2 | 0 | 0.300 |
| cyber_abuse | 30 | 18 | 7 | 3 | 2 | 0.696 |
| harmful_instructions | 30 | 25 | 4 | 1 | 0 | 0.850 |
| hate_or_harassment | 30 | 20 | 7 | 2 | 1 | 0.724 |
| legal_financial_safety | 30 | 20 | 10 | 0 | 0 | 0.667 |
| medical_safety | 30 | 14 | 13 | 3 | 0 | 0.517 |
| privacy_and_doxxing | 30 | 19 | 8 | 0 | 3 | 0.704 |
| safe_transformation | 30 | 11 | 16 | 3 | 0 | 0.417 |
| self_harm_support | 30 | 15 | 14 | 0 | 1 | 0.517 |
