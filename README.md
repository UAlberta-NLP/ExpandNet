# ExpandNet
Official codebase for the project *Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection*.

## eval_release.py

Takes two arguments:
1. A gold-standard file, listing the acceptable target-language senses for each synset. Format: [synset ID] [TAB] [lemmas, space separated]
2. An output file, listing exactly one sense per line. Format: [synset ID] [TAB] [lemma]

Output is an evaluation for each sense, and overall statistics.