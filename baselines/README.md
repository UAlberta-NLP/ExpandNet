## UWN Baseline

The script `uwn-to-bn.py` can be used to extract senses from Universal WordNet.

The UWN resource can be found in the `res` subfolder, though compressed. 
You will likely need to unzip it.

You will also need to identify the three letter 639-3 language code of the language you want to extract sense for.
See https://iso639-3.sil.org/code_tables/639/data or https://en.wikipedia.org/wiki/List_of_ISO_639-3_codes for information.

The script takes three arguments:
1. input_file: The .tsv file containing the contents of UWN. 
2. language: The 639-3 language code of the language to extract.
3. output_file: The file to save these output senses to.
4. data_file: The file that contains the IDs of those synsets we care about.

Altogether, it can be run as such:

```bash 
python3 uwn-to-bn.py \
--input_file res/uwn.tsv \
--language spa \
--output_file uwn-bn-out.tsv \
--data_file res/scdev_gold_senses.txt
```

## OC14 Baseline 

The script `oc14.py` can be used to obtain senses using the Oliver and Climent reimplementation.

It takes the input source data, a sheet containing a translation (as would be used as input to ExpandNet step 2, for example) and a file containing the source-side sense annotations.

For example:

```bash 
python oc14.py --src_data semcor_en.data.dev.xml --input_file exnet_step1_out_es.tsv --input_gold semcor_en.gold.key.dev.txt
```

## M23 Baseline 

The script `m23.py` can be used to obtain senses using the Martelli reimplementation.

It takes the input source data (and gold), the language, a sheet containing a translation (as would be used as input to ExpandNet step 2, for example) and a beta parameter (though please stick to the default value of 0.7)

For example:

```bash 
python m23.py --translation_df_file exnet_gpt_es.tsv --src_data semcor_en.data.dev.xml --src_gold semcor_en.gold.key.dev.txt --output_file m23_out.tsv
```

## Dictionary Baseline 

The script `dictbaseline.py` can be used to obtain senses using a dictionary baseline

It takes the input source data, the source gold, a dictionary, a join character, and the output file.

For example:

```bash 
python dictbaseline.py --src_data semcor_en.data.dev.xml --dictionary /home/dbasil1/cmput650/ExpandNet/res/dicts/wikpan-en-es.tsv --src_gold semcor_en.gold.key.dev.txt --join_char _ --output_file dictionary_baseline_spanish.out.tsv
```
