#!/usr/bin/env python3
"""
Automatic WordNet Creation from Parallel Corpora (SE13 Edition)
Based on: Oliver & Climent (2014) — "Automatic creation of WordNets from parallel corpora"

This script:
1. Loads your se13.tsv (with 'translation' and 'bn_gold_list' columns)
2. Uses spaCy to lemmatize and POS-tag the French text
3. Parses synset lists (e.g., ['nan', 'bn:00041942n', ...])
4. Applies the "most frequent translation" alignment method from the paper
5. Filters results using parameters `i` and `f` for precision
6. Saves high-confidence (synset, lemma) pairs to a TSV file

Paper parameters: i=2.5, f=5.0
"""

import pandas as pd
import spacy
from collections import defaultdict, Counter
import ast
import csv
import sys
import argparse
import xml_utils

# ===========================
# CONFIGURATION
# ===========================

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--src_data", type=str, default="semcor_en.data.dev.xml", 
                      help="Source data.")
  parser.add_argument("--input_file", type=str, default="exnet_step1_out.tsv", 
                      help="File containing a translation column, like the output of ExpandNet step 1.")
  parser.add_argument("--input_gold", type=str, default="semcor_en.gold.key.dev.txt", 
                      help="File containing the gold source annotations, like is given as input to ExpandNet step 1.")
  parser.add_argument("--i_factor", type=float,
                      help="The 'i' parameter as defined in the paper. Default of 2.5 is recommended.", default=2.5)
  parser.add_argument("--f_factor", type=float,
                      help="The 'f' parameter as defined in the paper. Default of 5 is recommended.", default=5)
  parser.add_argument("--output_file", type=str, default='oc14out.tsv',
                      help="File to save output to")

  return parser.parse_args()

args = parse_args()
INPUT_FILE = args.input_file
GOLD_FILE = args.input_gold

#OUTPUT_FILE = 'extracted_synset_lemma_pairs.tsv'
#I_THRESHOLD = 2.5   # Min ratio: freq(1st candidate) / freq(2nd candidate)
#F_THRESHOLD = 5.0   # Max ratio: freq(synset) / freq(winning lemma)
I_THRESHOLD = float(args.i_factor)  # Min ratio: freq(1st candidate) / freq(2nd candidate)
F_THRESHOLD = float(args.f_factor)  # Max ratio: freq(synset) / freq(winning lemma)
OUTPUT_FILE = args.output_file


def sentence_id_find(a):
    return a[:9]
 
 
def pos_map_this(a, m):
    if a in m:
        return m[a]
    else:
        return 'x'
    
# ===========================
# CORE FUNCTION
# ===========================
def extract_synset_lemma_pairs_from_bn_format(df, i_threshold=1.0, f_threshold=float('inf')):
    """
    Extracts (synset, lemma) pairs from SE13-style dataframe.
    - 'bn_gold_list': string repr of list like "[nan, 'bn:00041942n', ...]"
    - 'translation': raw target language text
    Uses spaCy for target language lemmatization and POS tagging.
    """
    # Load spaCy model (will error if not installed — see instructions below)
    

    # Map spaCy POS tags to WordNet-style single chars
    pos_map = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}

    # Collect all candidate lemmas for each synset
    synset_candidates = defaultdict(list)

    total_rows = len(df)
    print(f"Processing {total_rows} sentence pairs...")

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"\tProcessed {idx} / {total_rows} rows...")
        
        # Parse bn_gold_list: string → list
        # try:
        #     print("row['bn_gold_list']",row['bn_gold_list'])
        #     synset_list = ast.literal_eval(row['bn_gold_list'])
        #     print('synset_list',synset_list)
        #     if not isinstance(synset_list, list):
        #         continue
        # except (ValueError, SyntaxError, TypeError):
        #     continue  # Skip malformed or non-list rows
        # Parse bn_gold_list: space-separated string → list of tokens
        bn_gold_str = ' '.join([str(a) for a in row['bn_gold_list']])
        if not isinstance(bn_gold_str, str) or not bn_gold_str.strip():
            continue
        synset_list = bn_gold_str.split()

        # Get and validate French text
        
        
        target_lemmas = row['translation_lemma'].split()
        target_pos_tags = [pos_map_this(a, pos_map) for a in row['translation_pos'].split()]

        # For each synset in the list
        for synset in synset_list:
          
            if pd.isna(synset) or not isinstance(synset, str) or len(synset) < 2:
                continue

            # Extract POS from last char (e.g., 'n' from 'bn:00041942n')
            synset_pos = synset[-1]
            if synset_pos not in 'nvraNOUNVERBADJADV':  # Only care about open-class words
                continue
            
            
            
            # Collect all target lemmas with matching POS from this sentence
            for lemma, pos in zip(target_lemmas, target_pos_tags):
                if pos == synset_pos:
                    
                    synset_candidates[synset].append(lemma)
                

    # Select best lemma for each synset
    result_pairs = []
    total_synsets = len(synset_candidates)
    print(f"Found {total_synsets} unique synsets. Selecting best lemmas...")

    for synset, lemma_list in synset_candidates.items():
        if not lemma_list:
            print("NOT LEMMA LIST")
            continue

        lemma_counter = Counter(lemma_list)
        most_common = lemma_counter.most_common(2)
        top_lemma, top_freq = most_common[0]
        second_freq = most_common[1][1] if len(most_common) > 1 else 0

        # Calculate i_ratio (1st / 2nd candidate frequency)
        i_ratio = top_freq / second_freq if second_freq > 0 else float('inf')

        # Calculate f_ratio (total synset occurrences / top lemma frequency)
        total_synset_occurrences = len(lemma_list)
        f_ratio = total_synset_occurrences / top_freq

        # Apply paper's filtering
        if i_ratio >= i_threshold and f_ratio <= f_threshold:
            result_pairs.append((synset, top_lemma))

    return result_pairs

# ===========================
# MAIN EXECUTION
# ===========================
if __name__ == '__main__':
    print("Starting WordNet Extraction from Parallel Corpus (SE13)")
    print("=" * 60)

    # Load data
    try:
        print(f"Loading data from {INPUT_FILE}...")
        df = pd.read_csv(INPUT_FILE, sep='\t')
        print(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"File '{INPUT_FILE}' not found. Please check the filename and path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    gold_df = xml_utils.process_gold(GOLD_FILE)
    
    gold_df['sentence_id'] = gold_df.apply(
    lambda row: sentence_id_find(row['id']),
    axis=1
)
    
    df_src = xml_utils.process_dataset(args.src_data, GOLD_FILE)

    bn_gold_lists = (
        gold_df.groupby("sentence_id")["bn_gold"]
       .apply(list)
       .reset_index(name="bn_gold_list")
    )
    
    
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

   
    df_sent = xml_utils.extract_sentences(df_src)
   

    # Merge back into df_sent
    df_sent = (
        df_sent.merge(bn_gold_lists, on="sentence_id", how="left")
           .merge(lemma_gold_lists, on="sentence_id", how="left").merge(token_gold_lists, on="sentence_id", how="left")
    )
    
    df_sent['bn_gold_list'] = df_sent['bn_gold']
   
    combined = pd.concat([df_sent, df], axis=1)
    
    # Validate required columns
    required_cols = ['translation_lemma', 'bn_gold_list']
    missing_cols = [col for col in required_cols if col not in combined.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        sys.exit(1)

    # Run extraction
    print(f"\nRunning extraction with i={I_THRESHOLD}, f={F_THRESHOLD}...")
    pairs = extract_synset_lemma_pairs_from_bn_format(
        combined,
        i_threshold=I_THRESHOLD,
        f_threshold=F_THRESHOLD
    )

    # Report and save
    print(f"\nDone! Extracted {len(pairs)} high-confidence synset-lemma pairs.")

    if len(pairs) > 0:
        print(f"Saving results to {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                #writer.writerow(['synset', 'lemma'])  # header
                writer.writerows(pairs)
            print(f"Successfully saved to '{OUTPUT_FILE}'")
        except Exception as e:
            print(f"Error saving file: {e}")
            sys.exit(1)

        # Show sample
        print("\nSample of first 10 pairs:")
        for synset, lemma in pairs[:10]:
            print(f"  {synset} → {lemma}")
    else:
        print("️No pairs extracted. Try lowering i or raising f.")

