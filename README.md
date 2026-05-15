# ExpandNet
This repository is for the paper Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection. In Proceedings of the 39th Canadian Conference on Artificial Intelligence (Canadian AI 2026), Proceedings of Machine Learning Research (PMLR).

ExpandNet converts source-language lexical/sense annotations into target-language equivalents.
It does this in three steps: translating sentences, aligning source/target tokens, and projecting annotations across the alignment.
You can run each step independently and customize the dictionaries employed.
You may also manually supply translations if your language pair isn’t supported, thus skipping step 1.

## Step 1 Translate

Takes seven arguments:
1. src_data: An XML file containing the sentences to be translated
2. lang_src: The language key for the source language.
3. lang_tgt: The language key for the target language.
4. output_file: The address of the file where the result of the translation will be saved.
5. translator: 'gpt' or 'helsinki' to denote which of those translators to use. ('gpt' will require an OpenAI API key)
6. target_join_char: The character to use to connect multi-word expressions in the target language. Should not be a space.
7. no_pos: The system adds part-of-speech tags by default. Set this flag to skip that step.

Altogether, it can be run as such:

```bash 
python3 expandnet_step1_translate.py \
--src_data inputs/semcor_en.data.dev.xml \
--lang_src en \
--lang_tgt es \
--translator gpt \
--output_file expandnet_step1_translate.out.tsv \
--target_join_char _ 
```

## Translation Output

The output of the translation step is a tsv file with columns named: 'sentence_id', 'text', 'translation', 'lemma', 'translation_token', 'translation_lemma' and, optionally, 'translation_pos'. These columns should be tab-separated. The sentence id should be a unique identifier. 

- **sentence_id**: unique identifier of the source sentence.
- **text**: raw source-side text.
- **translation**: raw translation.
- **lemma**: space-separated source-language lemmas.
- **translation_token**: space-separated target-language tokens.  
  - If a token contains spaces, replace them using the `join_char` (e.g., underscores).  
  - This same character must be used consistently in later steps.
- **translation_lemma**: space-separated target-side lemmas.
- **translation_pos** (optional): space-separated target-side POS tags, using either  
  - the Universal POS tagset (17 tags),  
  - the simplified tagset: `n`, `a`, `j`, `r`, `x`, or
  - another tagset which can be mapped to the simplified tagset by modifying the pos_mapping_file in Step 3.


**If Step 1 is unsupported for your language pair, you may create a file of this format on your own, and continue to Step 2.**

Here is an example:


```tsv
sentence_id	text	translation	lemma	translation_token	translation_lemma	translation_pos
d000.s001	I ran	Yo corrí	I run	Yo corrí	yo correr	PRON VERB
```

### Note
Please refer to `requirements.txt` for dependencies. For steps 1 and 2, you may need to download additional spaCy language models.
You can do this with:

```bash
python3 -m spacy download <MODELNAME>
```

The models employed in the code as needed by default are: en_core_web_lg, es_core_news_lg, fr_core_news_lg, zh_core_web_lg, xx_ent_wiki_sm.


## Step 2 Align

For the alignment step, it is recommended to use DBAlign, for which a **dictionary** is required.
Dictionaries must be .tsv files, where each row contains a source-side word, then a tab character, then a space-separated list of possible target-side words that it may be translated as. A character which we call the `join_char` should be used in place of spaces for multi-word expressions, or for any tokens with spaces within them. You will provide this `join_char` as a command line argument to steps 2 and 3 (default: underscore)
An example dictionary, `res/dicts/wikpan-en-es.tsv` is included to demonstrate the format these dictionaries should take.

Step 2 takes ten arguments:
1. translation_df_file: The address of the .tsv created by Step 1 (or created independently if working with an unsupported language pair)
2. src_data: An XML file containing the sentences to be translated
3. lang_src: The language key for the source language (default 'en').
4. lang_tgt: The language key for the target language (default 'fr').
5. aligner: The aligner to be used, one of 'simalign' or 'dbalign'.
6. dictionary: If using dbalign, the path to the multilingual dictionary which it will use, or 'bn' to use BabelNet as this dictionary (if available). 
7. output_file: The address of the file where the result of the alignment step will be saved.
8. source_join_char:  The character to use to connect multi-word expressions in the source language. Should not be a space.
9. target_join_char:  The character to use to connect multi-word expressions in the target language. Should not be a space.
10. num_workers: The number of parallel processes to use. The default, 1, is strongly recommended for most cases.

Altogether, it can be run as such:

```bash 
python3 expandnet_step2_align.py \
--translation_df_file expandnet_step1_translate.out.tsv \
--src_data inputs/semcor_en.data.dev.xml \
--lang_src en \
--lang_tgt es \
--aligner dbalign \
--dictionary res/dicts/wikpan-en-es.tsv \
--output_file expandnet_step2_align.out.tsv \
--source_join_char _ \
--target_join_char _
```

## Step 3: Projection

The projection step takes the output of **Step 2 (alignment)** and uses it to transfer sense annotations or lexical information from the source language to the target language.

This script has **nine required arguments** plus four **optional flags** that toggle different filtering behaviors.

---

### **Required Arguments**

1. **src_data**  
   The original XML file containing the source-language sentences (the same file used in Step 1).

2. **src_gold**  
   The gold key file containing the source-language sense annotations.

3. **dictionary**  
   The bilingual dictionary used for lexical projection (typically the same `.tsv` dictionary used in Step 2).

4. **pos_mapping_file**  
   Points to a dictionary that corresponds POS tags to one of the basic four ExpandNet expects. (default: `pos_mapping_u.tsv`)

5. **alignment_file**  
   The alignment output file produced in Step 2.

6. **output_file**  
   Path to the file where projected annotations will be saved.

7. **source_join_char**  
   Character used to join multi-word lexical items from the source side during projection (default: `_`).

8. **target_join_char**  
   Character used to join multi-word lexical items from the target side during projection (default: `_`).

9. **token_info_file**  
   Path to the file where detailed token-level logs will be written.

---

### **Optional Flags**

These flags toggle different filtering screens.  
By default, **all filters are ON**.  
Passing a flag turns **OFF** the corresponding filter.

- **`--no_pos_screen`**  
  Turn off part-of-speech filtering. Normally, projections are rejected when the source and target POS differ. Requires POS information from previous steps.

- **`--no_ne_screen`**  
  Turn off named-entity filtering (which normally filters out capitalized named entities).

- **`--no_dict_screen`**  
  Turn off dictionary-based filtering. Normally, only dictionary-supported translations are projected.

- **`--no_oov_screen`**  
  Allow projection of English lexical items not found in the dictionary (OOV items).  
  By default, OOV English terms are not projected.



```bash 
python3 expandnet_step3_project.py \
--src_data inputs/semcor_en.data.dev.xml \
--src_gold inputs/semcor_en.gold.key.dev.txt \
--dictionary res/dicts/wikpan-en-es.tsv \
--alignment_file expandnet_step2_align.out.tsv \
--output_file expandnet_step3_project.out.tsv \
--source_join_char _ \
--target_join_char _ \
--token_info_file expandnet_step3_project.token_info.out.tsv \
--pos_mapping_file pos_mapping_u.tsv \
--no_ne_screen
```

## Evaluation

### Generating a gold file: expand_synsets.py

The file `expand_synsets.py` generates a gold-standard information file using BabelNet. 
This file can be used to evaluate the outputs of ExpandNet.
An example is provided in `res/spanishgold.tsv`

It takes three arguments:
1. The ISO code for the target langauge to be used
2. The address of the input (annotation) file containing the sense tags to be used
3. The desired address of the output file.

It can be run as such:

```bash 
python3 expand_synsets.py es inputs/semcor_en.gold.key.dev.txt res/spanishgold.tsv
```

### Evaluating using a gold file: eval.py

Takes two arguments:
1. A gold-standard file (as generated above), listing the acceptable target-language senses for each synset. Format: [synset ID] [TAB] [lemmas, space separated]
2. An output file (as generated in Step 3) listing exactly one sense per line. Format: [synset ID] [TAB] [lemma]

Output is an evaluation for each sense, and overall statistics.

```bash 
python eval.py res/spanishgold.tsv expandnet_step3_project.out.tsv
```