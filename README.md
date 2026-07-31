# ExpandNet: Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection

[![Paper](https://img.shields.io/badge/Paper-PMLR-red?style=flat-square)](https://proceedings.mlr.press/v318/basil26a.html)
[![arXiv](https://img.shields.io/badge/arXiv-2604.14397-b31b1b?style=flat-square)](https://arxiv.org/abs/2604.14397)
[![Poster](https://img.shields.io/badge/Poster-PDF-blue?style=flat-square)](assets/poster.pdf)
[![Slides](https://img.shields.io/badge/Slides-PDF-green?style=flat-square)](assets/slides.pdf)
[![Venue](https://img.shields.io/badge/Canadian%20AI-2026-orange?style=flat-square)](https://proceedings.mlr.press/v318/)

This repository is for the paper Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection. David Basil, Chirooth Girigowda, Bradley Hauer, Grzegorz Kondrak, Sahir Momin, Ning Shi. Proceedings of the The 39th Canadian Conference on Artificial Intelligence, PMLR 318:1036-1043, 2026.

We study the task of automatically expanding WordNet-style lexical resources to new languages through **sense generation**: given a sense-tagged English corpus and its translation, we associate target-language lemmas with existing lexical concepts via **semantic projection**, projecting annotated synsets onto aligned target-language tokens. To generate alignments and ensure their quality, we augment a pretrained base aligner with a **bilingual dictionary**, which is also used to filter incorrect sense projections. The resulting project-and-filter strategy improves precision while remaining interpretable and resource-efficient.

---

## Overview

ExpandNet converts source-language lexical/sense annotations into target-language equivalents in three steps:

1. **Translate** — translate the source sentences (GPT or Helsinki MT), producing tokens, lemmas, and optional POS tags.
2. **Align** — align source and target tokens, using either SimAlign or **DBAlign**, a dictionary-augmented aligner.
3. **Project** — transfer sense annotations across the alignment, applying POS, named-entity, dictionary, and OOV filters.

Each step runs independently and can be customized with your own dictionaries. If your language pair is unsupported by Step 1, you may supply translations manually and skip directly to Step 2.

---

## Repository Structure

```
.
├── expandnet_step1_translate.py   # Step 1: sentence translation (GPT / Helsinki)
├── expandnet_step2_align.py       # Step 2: token alignment (SimAlign / DBAlign)
├── expandnet_step3_project.py     # Step 3: sense projection + filtering
├── expand_synsets.py              # Build a BabelNet gold file for evaluation
├── eval.py                        # Score projected senses against a gold file
├── align_utils.py                 # Aligner implementations and helpers
├── gpt_translate.py               # OpenAI translation backend
├── xml_utils.py                   # Corpus XML parsing utilities
├── pos_mapping_u.tsv              # Universal POS → simplified tagset mapping
├── inputs/                        # Source corpus (SemCor) and gold key
├── res/dicts/                     # Bilingual dictionaries (en–es, en–fr, en–zh, en–ur)
├── res/                           # Gold sense inventories and word lists
├── outputs/                       # Generated sense inventories (es, fr, zh, ur)
└── assets/                        # Paper, poster, and slides
```

---

## Dependencies

Install the Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Steps 1 and 2 additionally require spaCy language models, which must be downloaded separately:

```bash
python3 -m spacy download <MODELNAME>
```

The models used by default are `en_core_web_lg`, `es_core_news_lg`, `fr_core_news_lg`, `zh_core_web_lg`, and `xx_ent_wiki_sm`.

BabelNet (`babelnet>=5.0.0`) is optional, and needed only if you pass `bn` as the dictionary in Step 2 or run `expand_synsets.py`.

---

## 🚀 Quick Start & Replication

To replicate our system end to end (English → Spanish shown):

```bash
# 1. Translate
python3 expandnet_step1_translate.py \
  --src_data inputs/semcor_en.data.dev.xml \
  --lang_src en --lang_tgt es \
  --translator gpt \
  --output_file expandnet_step1_translate.out.tsv \
  --target_join_char _

# 2. Align
python3 expandnet_step2_align.py \
  --translation_df_file expandnet_step1_translate.out.tsv \
  --src_data inputs/semcor_en.data.dev.xml \
  --lang_src en --lang_tgt es \
  --aligner dbalign \
  --dictionary res/dicts/wikpan-en-es.tsv \
  --output_file expandnet_step2_align.out.tsv \
  --source_join_char _ --target_join_char _

# 3. Project
python3 expandnet_step3_project.py \
  --src_data inputs/semcor_en.data.dev.xml \
  --src_gold inputs/semcor_en.gold.key.dev.txt \
  --dictionary res/dicts/wikpan-en-es.tsv \
  --alignment_file expandnet_step2_align.out.tsv \
  --output_file expandnet_step3_project.out.tsv \
  --pos_mapping_file pos_mapping_u.tsv \
  --source_join_char _ --target_join_char _ \
  --token_info_file expandnet_step3_project.token_info.out.tsv \
  --no_ne_screen

# 4. Evaluate
python3 eval.py res/spanishgold.tsv expandnet_step3_project.out.tsv
```

Pre-generated sense inventories for Spanish, French, Chinese, and Urdu are available in `outputs/`.

---

## Step 1: Translate

Takes seven arguments:

1. `src_data`: An XML file containing the sentences to be translated.
2. `lang_src`: The language key for the source language.
3. `lang_tgt`: The language key for the target language.
4. `output_file`: The address of the file where the result of the translation will be saved.
5. `translator`: `gpt` or `helsinki` to denote which of those translators to use (`gpt` requires an OpenAI API key).
6. `target_join_char`: The character used to connect multi-word expressions in the target language. Should not be a space.
7. `no_pos`: The system adds part-of-speech tags by default. Set this flag to skip that step.

### Translation Output

The output is a TSV file with columns `sentence_id`, `text`, `translation`, `lemma`, `translation_token`, `translation_lemma`, and optionally `translation_pos`:

- **sentence_id**: unique identifier of the source sentence.
- **text**: raw source-side text.
- **translation**: raw translation.
- **lemma**: space-separated source-language lemmas.
- **translation_token**: space-separated target-language tokens. Tokens containing spaces must use the `join_char` (e.g. underscores), applied consistently in later steps.
- **translation_lemma**: space-separated target-side lemmas.
- **translation_pos** (optional): space-separated target-side POS tags, using either the Universal POS tagset (17 tags), the simplified tagset (`n`, `a`, `j`, `r`, `x`), or another tagset mappable to the simplified one by modifying the `pos_mapping_file` in Step 3.

Example:

```tsv
sentence_id	text	translation	lemma	translation_token	translation_lemma	translation_pos
d000.s001	I ran	Yo corrí	I run	Yo corrí	yo correr	PRON VERB
```

**If Step 1 is unsupported for your language pair, you may create a file of this format on your own and continue to Step 2.**

---

## Step 2: Align

For the alignment step, we recommend **DBAlign**, which requires a **dictionary**. Dictionaries are TSV files where each row contains a source-side word, a tab, then a space-separated list of possible target-side translations. The `join_char` replaces spaces in multi-word expressions and in any token containing spaces; you pass this character to Steps 2 and 3 (default: underscore). See `res/dicts/wikpan-en-es.tsv` for the expected format.

Takes ten arguments:

1. `translation_df_file`: The TSV created by Step 1 (or created independently for an unsupported language pair).
2. `src_data`: An XML file containing the sentences to be translated.
3. `lang_src`: The language key for the source language (default `en`).
4. `lang_tgt`: The language key for the target language (default `fr`).
5. `aligner`: One of `simalign` or `dbalign`.
6. `dictionary`: If using DBAlign, the path to the multilingual dictionary, or `bn` to use BabelNet as the dictionary (if available).
7. `output_file`: The address of the file where the alignment result will be saved.
8. `source_join_char`: The character used to connect multi-word expressions in the source language. Should not be a space.
9. `target_join_char`: The character used to connect multi-word expressions in the target language. Should not be a space.
10. `num_workers`: The number of parallel processes to use. The default, 1, is strongly recommended for most cases.

---

## Step 3: Projection

The projection step takes the output of Step 2 and uses it to transfer sense annotations or lexical information from the source language to the target language.

### Required Arguments

1. **src_data** — The original XML file containing the source-language sentences (the same file used in Step 1).
2. **src_gold** — The gold key file containing the source-language sense annotations.
3. **dictionary** — The bilingual dictionary used for lexical projection (typically the same `.tsv` dictionary used in Step 2).
4. **pos_mapping_file** — Maps POS tags to one of the basic four ExpandNet expects (default: `pos_mapping_u.tsv`).
5. **alignment_file** — The alignment output file produced in Step 2.
6. **output_file** — Path to the file where projected annotations will be saved.
7. **source_join_char** — Character used to join multi-word lexical items on the source side (default: `_`).
8. **target_join_char** — Character used to join multi-word lexical items on the target side (default: `_`).
9. **token_info_file** — Path to the file where detailed token-level logs will be written.

### Optional Flags

These flags toggle the filtering screens. By default **all filters are ON**; passing a flag turns the corresponding filter **OFF**.

- **`--no_pos_screen`** — Turn off part-of-speech filtering. Normally, projections are rejected when the source and target POS differ. Requires POS information from previous steps.
- **`--no_ne_screen`** — Turn off named-entity filtering, which normally filters out capitalized named entities.
- **`--no_dict_screen`** — Turn off dictionary-based filtering. Normally, only dictionary-supported translations are projected.
- **`--no_oov_screen`** — Allow projection of English lexical items not found in the dictionary. By default, OOV English terms are not projected.

---

## Evaluation

### Generating a gold file: `expand_synsets.py`

`expand_synsets.py` generates a gold-standard information file using BabelNet, which can be used to evaluate the outputs of ExpandNet. An example is provided in `res/spanishgold.tsv`.

It takes three arguments: the ISO code for the target language, the address of the input annotation file containing the sense tags, and the desired output address.

```bash
python3 expand_synsets.py es inputs/semcor_en.gold.key.dev.txt res/spanishgold.tsv
```

### Evaluating using a gold file: `eval.py`

Takes two arguments:

1. A gold-standard file (as generated above), listing the acceptable target-language senses for each synset. Format: `[synset ID] [TAB] [lemmas, space separated]`.
2. An output file (as generated in Step 3) listing exactly one sense per line. Format: `[synset ID] [TAB] [lemma]`.

Output is an evaluation for each sense, plus overall statistics.

```bash
python3 eval.py res/spanishgold.tsv expandnet_step3_project.out.tsv
```

---

## BibTeX

```bibtex
@InProceedings{pmlr-v318-basil26a,
  title = 	 {Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection},
  author =       {Basil, David and Girigowda, Chirooth and Hauer, Bradley and Kondrak, Grzegorz and Momin, Sahir and Shi, Ning},
  booktitle = 	 {Proceedings of the The 39th Canadian Conference on Artificial Intelligence},
  pages = 	 {1036--1043},
  year = 	 {2026},
  editor = 	 {Bouzar-Benlabiod, Lydia and Leung, Carson},
  volume = 	 {318},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--29 May},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v318/main/assets/basil26a/basil26a.pdf},
  url = 	 {https://proceedings.mlr.press/v318/basil26a.html},
  abstract = 	 {We study the task of automatically expanding WordNet-style lexical resources to new languages through sense generation. We generate senses by associating target-language lemmas with existing lexical concepts via semantic projection. Given a sense-tagged English corpus and its translation, our method projects the annotated synsets onto aligned target-language tokens and assigns the corresponding lemmas to those synsets. To generate alignments and ensure their quality, we augment a pretrained base aligner with a bilingual dictionary, which is also used to filter incorrect sense projections. We evaluate the method on multiple languages, comparing it to prior methods, as well as dictionary-based and large language model baselines. Results show that the proposed project-and-filter strategy improves precision while remaining interpretable and resource-efficient. We release our code, documentation, and generated sense inventories at https://github.com/UAlberta-NLP/ExpandNet.}
}
```
