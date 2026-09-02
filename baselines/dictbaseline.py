import argparse
import xml_utils
import pandas as pd
import csv
import sys
csv.field_size_limit(sys.maxsize)

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--src_data", type=str, default="semcor_en.data.dev.xml", 
                      help="Source data.")
  parser.add_argument("--dictionary", type=str, default="/home/dbasil1/cmput650/ExpandNet/res/dicts/wikpan-en-es.tsv", 
                      help="File containing a dictionary.")
  parser.add_argument("--src_gold", type=str, default="semcor_en.gold.key.dev.txt", 
                      help="File containing the gold source annotations, like is given as input to ExpandNet step 1.")
  
  parser.add_argument("--join_char", type=str, default='_')
  parser.add_argument("--output_file", type=str, default='dictblout.tsv')
  return parser.parse_args()

args = parse_args()

def get_synonyms_by_dict(l, d):
    if l not in d:
        return []
   
    return list(d[l])
   

print("Loading dataset...")
df_src = xml_utils.process_dataset(args.src_data, args.src_gold)
print(f"Dataset loaded: {len(df_src)} rows")


def load_dict(filepaths, jc):
    """Load multiple TSV files into a dict: {english_word: set(french_words)}.
    All spaces are normalized to underscores.
    """
    dict_ = {}
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for line_num, row in enumerate(reader, start=1):
                if len(row) < 2:
                    print(f"Warning: Line {line_num} in {filepath} has fewer than 2 columns.")
                    continue
                eng_word = row[0].strip().lower().replace(' ', '_').replace('_', jc)  # Normalize English key
                fr_words = set(word.strip().lower().replace(' ', '_').replace('_', jc) for word in row[1].split())
                if eng_word in dict_:
                    dict_[eng_word].update(fr_words)  # Merge sets if key exists
                else:
                    dict_[eng_word] = fr_words
    return dict_

dictionary = load_dict([args.dictionary], args.join_char)


bn_gold_lists = (
    df_src.groupby("sentence_id")["bn_gold"]
       .apply(list)
       .reset_index(name="bn_gold")
)

lemma_gold_lists = (
    df_src.groupby("sentence_id")["lemma"]
       .apply(list)
       .reset_index(name="lemma_gold")
)

token_gold_lists = (
    df_src.groupby("sentence_id")["text"]
       .apply(list)
       .reset_index(name="token_gold")
)


merged_df = pd.merge(
    lemma_gold_lists,
    bn_gold_lists,
    on="sentence_id",
    how="inner"   # ensures only matching IDs are kept
)


senses = set() 

with open(args.output_file, 'w', encoding='utf-8') as f:
  for index, row in merged_df.iterrows():
    
    lemmas = row['lemma_gold']
    bnids = row['bn_gold']
    assert len(bnids) == len(lemmas)
    for i in range(len(bnids)):
        lem = lemmas[i]
        bn = bnids[i]
        if str(bn) != 'nan':
            for guy in get_synonyms_by_dict(lem, dictionary):
                if (bn, guy) not in senses:
                    f.write(bn + '\t' + guy + '\n')
                    senses.add((bn, guy))
                
print(f"saved {len(senses)} unique senses to {args.output_file}")