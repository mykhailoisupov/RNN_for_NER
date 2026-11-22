# Post-processing: threshold, entity merging, label policy (CV and LB)

What turns word probabilities into answers, and what was tried on the way. Unless stated otherwise, all rows use the
same four ensemble members: fastText `b1yzs79t`, `e4ob4l0j` and Flair `lr5hudf6`, `fi9xjw14`, with probabilities
averaged. CV is the val split of that day (2026-11-19, before held-out names), sample is `labeling_sample.csv`
(26 test-like posts), LB is the public leaderboard.

## 1. Threshold optimisation: weight of the O class

The O probability was multiplied by `o_weight` before decoding (commit `cf1dc77`, grid on val with a paired
bootstrap).

| o_weight | val P | val R | val F1 | sample F1 | LB |
|---|---|---|---|---|---|
| 0.5 | 0.9005 | 0.9459 | 0.9226 | 0.7561 | |
| 0.8 | 0.9164 | 0.9378 | 0.9270 | 0.7750 | |
| **0.9** | 0.9208 | 0.9348 | **0.9277** | 0.7692 | 0.64663 |
| 1.0 (none) | 0.9280 | 0.9263 | 0.9271 | 0.8000 | 0.64739 |
| 1.1 | 0.9327 | 0.9112 | 0.9218 | 0.8108 | |
| 1.4 | 0.9400 | 0.9021 | 0.9207 | 0.7606 | |
| 2.0 | 0.9461 | 0.8885 | 0.9164 | 0.7353 | |

The best val weight (0.9, +0.0006 F1) lost on LB (−0.0008), so the gain was noise. The LB of 1.0 is `ens4-guards`
(2026-11-15, same members, Viterbi): no submission without the weight was made on 2026-11-19. The weight was dropped in
`f3c7d63`: rampart's span thresholds are the only confidence cut.

## 2. Entity-merging algorithms

Word tags -> spans (commit `661b9dc`, `merge_grid`). `strict` repairs BIO (no span starts on I), `merge_adjacent`
joins neighbouring spans. `rampart(min, extend)` is the span repair ported from rampart's `classifier.ts`, run on
per-word argmax: spans scoring at least `min` are kept, spans scoring at least `extend` survive only next to one of
them.

| Decoder | val F1 | sample F1 | LB |
|---|---|---|---|
| Viterbi (no O->I) | **0.9271** | 0.8000 | 0.64739 |
| argmax | 0.9253 | 0.8000 | |
| argmax + strict | 0.9270 | 0.8000 | |
| Viterbi + merge_adjacent | 0.9200 | 0.8000 | |
| rampart(0.4, 0.15) | 0.9055 | 0.7792 | |
| rampart(0.6, 0.15) | 0.9038 | 0.7576 | |
| rampart + rescue of capitalised particles | 0.8616 | 0.7436 | |
| **rampart on Viterbi tags** | 0.9090 | **0.8108** | **0.66180** |

Rampart on Viterbi tags lost 0.018 F1 on val but won 0.014 on LB (Viterbi's LB is the `ens4-guards` submission, as
above), the largest CV/LB disagreement of the project. A likely reason: val was then built from single labelled
sentences, while test is Telegram posts, where rampart's joining of name pieces across a space, hyphen or apostrophe
pays off. Rampart on Viterbi tags became the submission path.

Follow-ups on the same members:

| Change | LB |
|---|---|
| rampart on Viterbi tags | 0.66180 |
| + overlapping windows of 128 words, overlap 64 | 0.66084 |
| + overlapping windows of 256 words, overlap 64 (`fdb3865`) | 0.66277 |
| + `trim` keeps the closing quote it opens (`ecd6ed4`) | 0.66569 |

## 3. Label policy: one step at a time (2026-11-22)

Each `POLICY` step from `src/label_policy.py` left out in turn; baseline = the policy of that day (0.66473). Val and
sample are scored without the policy, so only LB measures it. `test changed` counts test posts whose answers differ.

| Variant | test changed | LB | Δ |
|---|---|---|---|
| without `label_capital` | 10 | 0.66959 | +0.0049 |
| without `label_facilities` | 3 | 0.66666 | +0.0019 |
| no policy at all | 28 | 0.66569 | +0.0010 |
| without `join_house_number` | 2 | 0.66473 | 0 |
| without `drop_org_names` | 7 | 0.66184 | −0.0029 |
| without `join_settlement_type` | 9 | 0.66182 | −0.0029 |
| **without `label_capital`, `label_facilities`, `join_house_number`** | 15 | **0.67155** | +0.0068 |

`label_capital` and `label_facilities` were deleted, `join_house_number` runs on train only (commit `2bfa9fc`).
Steps that changed no test post (`cut_at_bracket`, `trim_edges`, `join_street_type`, `join_area_type`,
`drop_common_nouns`) stay: they cost nothing at submit and also label train (`TRAIN_POLICY`).

## Caveats

- Each delta is 1-3 posts of the public half of test; the private half may disagree.
- The members predate the U+3164 tokenizer change (`ed924fd`); for section 3 their extra test rows were dropped to
  match it. New models need new runs of `scripts/compare_decoders.py`.
