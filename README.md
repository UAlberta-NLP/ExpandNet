# ExpandNet
Official codebase for the project *Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection*.

The pipeline consists of three steps: translation, alignment, and projection. 
The built-in translation module only support a limited number of language pairs, so if you are working with an unsupported pair, you may directly supply translations to the second step of the pipeline in the format it expects.

## Step 1 Translate

Takes four arguments:
1. src_data: An xml file containing the sentences to be translated
2. lang_src: The language key for the source language (default 'en').
3. lang_tgt: The language key for the target language (default 'fr').
4. output_file: The address of the file where the result of the translation will be saved.

Altogether, it can be run as such:

python3 expandnet_step1_translate.py --src_data xlwsd_se13.xml --lang_src en --lang_tgt es --output_file expandnet_step1_translate.out.tsv

## Translation Output

The output of the translation step is a tsv file with columns named: 'sentence_id', 'text', 'translation', 'lemma', 'translation_token', 'translation_lemma'.
These columns should naturally be tab-separated. The sentence id should be a unique identifier, the field 'text' should contain the raw source-side text, 'translation' should contain the raw translation, 'lemma' should contain a string of space-separated source-language lemmatization, translation_token should contain space-separated target language tokens (with underscores used to replace tokens which contain spaces for any reason) and translation_lemma contains a space-separated string of target-side lemmas in much the same way. 
If Step 1 is unsupported for your language pair, you may create a file of this format on your own, and continue to step 2.
Note that lemmatization is recommended, but not strictly necessary. The lemmatized columns may contain the same text as the tokenized columns if needed.

## Step 2 Align

For the alignment step, it is recommended to use DBAlign, for which a dictionary is required. 
Dictionaries must be .tsv files, where each row contains a source side word, then a tab character, then a space-separated list of possible target-side words that it may be translated as. Underscores should be used in place of spaces for multi-word expressions, or any tokens with spaces within them.
An example, wikpan-en-es.tsv is included to demonstrate the format these dictioaries should take.

Takes six arguments:
1. translation_df_file: The address of the .tsv created by Step 1 (or created independently if working with an unsupported language pair)
2. lang_src: The language key for the source language (default 'en').
3. lang_tgt: The language key for the target language (default 'fr').
4. aligner: The aligner to be used, one of 'simalign' or 'dbalign'.
5. dict: If using dbalign, the multilingual dictionary which it will use, or 'bn' to use BabelNet as this dictionary (if available). 
6. output_file: The address of the file where the result of the alignment step will be saved.

Altogether, it can be run as such:

python3 expandnet_step2_align.py --translation_df_file expandnet_step1_translate.out.tsv --lang_src en --lang_tgt es --aligner dbalign --dict wikpan-en-es.tsv --output_file expandnet_step2_align.out.tsv


## eval_release.py

Takes two arguments:
1. A gold-standard file, listing the acceptable target-language senses for each synset. Format: [synset ID] [TAB] [lemmas, space separated]
2. An output file, listing exactly one sense per line. Format: [synset ID] [TAB] [lemma]

Output is an evaluation for each sense, and overall statistics.