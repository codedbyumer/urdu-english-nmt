# Urdu–English Neural Machine Translation (Low-Resource NLP)

This project fine-tunes `mT5-small` for Urdu↔English translation, then tries to improve
it with LLM-based data augmentation (back-translation with NLLB-200, paraphrasing with a
small open instruct model), and compares BLEU / chrF before and after augmentation.

## Why this project

Urdu is spoken by over 230 million people, but it's still a low-resource language in
NLP — there just isn't as much parallel training data available for it compared to
languages like French or German. I wanted to see how much a low-resource pair like
Urdu-English could be improved just by adding LLM-generated synthetic data on top of a
small fine-tuned baseline, without needing a huge new human-labeled dataset.

## Files
- `urdu_english_nmt.ipynb` — the full pipeline, built to run on Google Colab.

## How to run
1. Upload `urdu_english_nmt.ipynb` to [Google Colab](https://colab.research.google.com/).
2. `Runtime → Change runtime type → T4 GPU` (the free tier is enough).
3. Run the cells top to bottom.
4. Total runtime is roughly 2–2.5 hours on a free T4 (most of that is the two fine-tuning
   runs and the back-translation/paraphrase generation step).

## Pipeline
1. **Data** — loads the `en-ur` pair from `Helsinki-NLP/opus-100` (20k sentence pairs by
   default, controlled by `N_TRAIN`).
2. **Preprocessing** — cleans, dedupes, filters misaligned pairs by length ratio, and
   tokenizes with the mT5 tokenizer using a `"translate English to Urdu: "` prefix.
3. **Baseline model** — fine-tunes `mT5-small` for 3 epochs.
4. **Data augmentation**:
   - *Back-translation*: `facebook/nllb-200-distilled-600M` translates extra English
     monolingual sentences into Urdu, creating new synthetic training pairs.
   - *Paraphrasing*: `Qwen/Qwen2.5-1.5B-Instruct` paraphrases English sentences from the
     training set, keeping the original Urdu translation paired with the new phrasing.
5. **Augmented model** — fine-tunes a *fresh* `mT5-small` checkpoint on baseline +
   synthetic data, so the comparison to the baseline is fair (same architecture, same
   hyperparameters — only the training data is different).
6. **Evaluation** — BLEU and chrF on a shared held-out test set for both models.
7. **Error analysis** — pulls out test sentences where the two models disagree and
   buckets them by source sentence length, to see where augmentation actually helps.
8. **Optional** — a commented-out stub to compare against the Google Translate API if I
   want an external reference point later (not required for the core results).

## Results

| Model | BLEU | chrF |
|---|---|---|
| Baseline | 7.37 | 25.02 |
| Augmented | 8.21 | 26.10 |

Adding synthetic data (back-translation + paraphrasing) improved both BLEU and chrF over
the baseline. Looking at the error analysis, the two models disagree far more often on
longer sentences (59 disagreements for 0–5 word sentences, rising to 157 for 20+ word
sentences) — so augmentation seems to help most on harder, longer translations rather
than short/simple ones.

**Training data used:** 19,793 baseline pairs + 5,000 back-translated pairs + ~1,995
paraphrased pairs (~26,788 total pairs for the augmented model).

For context: BLEU scores in the single digits are typical for a small model (mT5-small) fine-tuned on a modest 20k-pair dataset for a genuinely low-resource pair like Urdu-English — published low-resource NMT baselines in this data/compute range commonly fall in a similar range. The relative improvement from augmentation (+0.84 BLEU, +1.08 chrF) is the meaningful signal here, not the absolute score.

## Notes / things I had to work around

## Notes / things I had to work around
- I ran into NaN losses fine-tuning mT5 with the default AdamW optimizer + fp16 — mT5
  turns out to be numerically unstable that way. Switching to the `Adafactor` optimizer
  (what Google originally pretrained mT5 with) and disabling fp16 fixed it.
- I load NLLB-200 and Qwen2.5 directly through 🤗 `transformers` instead of something
  like Ollama, since that keeps everything in one self-contained notebook without a
  separate background server process.
- Colab kept disconnecting mid-run on the free tier, so I added checkpointing to Google
  Drive after each expensive step (data cleaning, back-translation, paraphrasing) — if
  the runtime restarts, I can resume instead of redoing everything from scratch.

## Adjusting scope
- `N_TRAIN`, `N_MONOLINGUAL`, `N_PARAPHRASE` at the top of their respective cells control
  how much data/compute each stage uses.
- Swap `MODEL_NAME = "google/mt5-small"` for `"facebook/mbart-large-50-many-to-many-mmt"`
  to try the mBART variant instead.
