# BPE tokenization (BPEmb): tried, lost to word-level fastText, removed

The assignment asks to try different tokenization algorithms (word-level, BPE). On 2026-11-10 the same BiLSTM was
trained twice; only the input representation changed:

- **BPE**: Ukrainian BPEmb subwords (vocabulary 50 000, 300-d vectors, frozen); a word is split into pieces, the first
  piece carries the word's tag and the rest continue it as `I`.
- **Word-level**: fastText `cc.uk.300` word vectors (frozen), one vector per word.

Both runs used 200 000 training posts, 5 epochs, a 2-layer BiLSTM with hidden size 256, dropout 0.3, batch 128, lr 1e-3.

| Run (W&B) | Tokenization | val F1 | sample F1 | Public LB |
|---|---|---|---|---|
| [`jv9r39lu`](https://forge.coreweave.com/wandb/whoresandbaddies/rnn-for-ner/runs/jv9r39lu) `bpemb-bilstm` | BPE (BPEmb) | 0.8564 | 0.6133 | 0.56186 |
| [`xbb5ejtq`](https://forge.coreweave.com/wandb/whoresandbaddies/rnn-for-ner/runs/xbb5ejtq) `fasttext-bilstm` | word (fastText) | 0.8751 | 0.7532 | 0.57061 |

Word-level fastText won on all three scores, most clearly on `labeling_sample.csv` (test-like posts, the only gold
labels drawn from test).

## Why BPE was removed from the code

- It lost on every split, and the models built afterwards (Flair, CRF, char CNN, word shape) were all word-level.
- Several ids per word needed extra code everywhere: piece labels, a piece -> word map for every prediction, and
  windowed inference that could not cut posts between pieces.

## Caveats

- These runs predate `scripts/build_dataset.py`: their val is an older random split, so the val F1 above is not
  comparable to later runs. Compare the two runs with each other only.
- The architecture was the early one: no CRF, no char CNN, no word shape, no windowed inference.

## Code

- W&B records commit `d1bdd9b` for both runs; the BPEmb code was committed in `03e3d61`.
- The last version with BPEmb support (`BpeVocab` in `src/vocab.py`, `--vocab.embedding bpemb`) is tagged
  `bpemb-last`: `git show bpemb-last:src/vocab.py`, or `git checkout bpemb-last` to run it.
