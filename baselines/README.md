# ExpandNet Baselines

This subfolder contains scripts to run various baselines for comparison against ExpandNet.

## UWN Baseline

The script `uwn-to-bn.py` extracts senses from Universal WordNet (UWN) and converts their synset IDs from the WordNet format to BabelNet format. 

*Note: This script requires the [BabelNet package](https://babelnet.org/guide).*

**Prerequisites:**
1. Unzip the UWN resource located in the local `res/` subfolder (inside this `baselines` directory).
2. Identify the 3-letter ISO 639-3 code for your target language (see [SIL data](https://iso639-3.sil.org/code_tables/639/data) or [Wikipedia](https://en.wikipedia.org/wiki/List_of_ISO_639-3_codes)).

**Arguments:**
* `--input_file`: The `.tsv` file containing the contents of UWN.
* `--language`: The 639-3 language code to extract.
* `--output_file`: Where to save the output senses.
* `--data_file`: The file containing the IDs of the synsets to process.

**Example:**
```bash 
python3 uwn-to-bn.py \
--input_file res/uwn.tsv \
--language spa \
--output_file uwn-bn-out.tsv \
--data_file res/scdev_gold_senses.txt

```

## OC14 Baseline

The script `oc14.py` obtains senses using the Oliver and Climent reimplementation.

**Arguments:**

* `--src_data`: The input source data XML.
* `--input_file`: A sheet containing a translation (e.g., output from ExpandNet Step 1).
* `--input_gold`: The file containing the source-side sense annotations.

**Example:**

```bash
python oc14.py --src_data ../inputs/semcor_en.data.dev.xml --input_file exnet_step1_out_es.tsv --input_gold ../inputs/semcor_en.gold.key.dev.txt

```

## M23 Baseline

The script `m23.py` obtains senses using the Martelli reimplementation.

**Arguments:**

* `--src_data`: The input source data XML.
* `--src_gold`: The source gold key annotations.
* `--input_file`: A sheet containing a translation (e.g., output from ExpandNet Step 1).
* `--output_file`: Where to save the output.
* `--language`: The 639-3 language code.
* `--beta`: Beta parameter (default is `0.7`; it is recommended to leave this unchanged).

**Example:**

```bash
python3 m23.py --input_file exnet_step1_out_es.tsv --src_data ../inputs/semcor_en.data.dev.xml --src_gold ../inputs/semcor_en.gold.key.dev.txt --output_file m23_out.tsv

```

## Dictionary Baseline

The script `dictbaseline.py` obtains senses using a simple dictionary baseline.

**Arguments:**

* `--src_data`: The input source data XML.
* `--src_gold`: The source gold key annotations.
* `--dictionary`: The bilingual dictionary file to use.
* `--join_char`: The character used to connect multi-word expressions.
* `--output_file`: Where to save the output.

**Example:**

```bash
python3 dictbaseline.py --src_data ../inputs/semcor_en.data.dev.xml --dictionary ../res/dicts/wikpan-en-es.tsv --src_gold ../inputs/semcor_en.gold.key.dev.txt --join_char _ --output_file dictionary_baseline_spanish.out.tsv

```
