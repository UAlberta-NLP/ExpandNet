import sys
import os
import argparse
import pandas as pd

import xml_utils

from collections import defaultdict
def final(beta, cands):
    
    print()
    print(len(cands), 'possible candidates')
    # Step 1: Aggregate counts
    # Key: (english_lemma, synset_id) -> { french_lemma: count }
    sense_to_french = defaultdict(lambda: defaultdict(int))

    for line in candidates_to_remember:
        

        _, en_lemma, synset_id, fr_lemma = line
        key = (en_lemma, synset_id)
        sense_to_french[key][fr_lemma] += 1

    # Step 2: For each sense, sort candidates, normalize, and filter by β
    output_lines = []

    for (en_lemma, synset_id), fr_candidates in sense_to_french.items():
        # Sort candidates by frequency (descending)
        sorted_candidates = sorted(fr_candidates.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate total count for L1 normalization
        total_count = sum(count for _, count in sorted_candidates)
        if total_count == 0:
            continue

        # L1-normalize the scores
        normalized_scores = [(fr_word, count / total_count) for fr_word, count in sorted_candidates]

        # Apply β filtering
        cumulative_score = 0.0
        filtered_french_words = []

        for fr_word, norm_score in normalized_scores:
            if cumulative_score >= beta:
                break
            filtered_french_words.append(fr_word)
            cumulative_score += norm_score

        # Output: one line per (synset_id, french_lemma) pair
        for fr_word in filtered_french_words:
            output_lines.append(f"{synset_id}\t{fr_word}")

    # Output all results
    
    with open(output_file, 'w', encoding='utf-8') as outf:
        for line in output_lines:
            outf.write(str(line) + '\n')
            
def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--src_data", type=str, default="semcor_en.data.dev.xml",
                      help="Path to the XLWSD XML corpus file.")
  parser.add_argument("--src_gold", type=str, default="semcor_en.gold.key.dev.txt",
                      help="Path to the gold sense tagging file.")
  parser.add_argument("--translation_df_file", type=str, default="exnet_step1_out.tsv",
                      help="File to load sentences and translations from.")
  parser.add_argument("--beta", type=float, default=0.7,
                      help="Beta (strongly recommend using default 0.7)")
  parser.add_argument("--output_file", type=str, default='m23_out.tsv',
                      help="The address to save the output to")
  return parser.parse_args()


args = parse_args()

src_data = args.src_data
src_gold = args.src_gold
output_file = args.output_file

translation_df_file = args.translation_df_file


# Print argument details.

# print(f"Corpus: {src_data}")
# print(f"Gold tags: {src_gold}")

df_src = xml_utils.process_dataset(src_data, src_gold)
# print(df_src.head(30), '\n')

if os.path.exists(translation_df_file):
  # print(f'The file "{translation_df_file}" exists. Loading...')
  df_sent = pd.read_csv(translation_df_file, sep='\t')
  # print('Loading complete.')
else:
  # print(f'The file "{translation_df_file}" does not exist. Exiting...')
  sys.exit()



### Set up aligner.
from simalign import SentenceAligner
ALIGNER = SentenceAligner(model="xlmr", layer=8, token_type="bpe", matching_methods="mai")
def align(lang_src, lang_tgt, tokens_src, tokens_tgt):
    # return [(0,0)]
    alignment_links = ALIGNER.get_word_aligns(tokens_src, tokens_tgt)['itermax']
     
    return alignment_links


from tqdm import tqdm

# Create your own progress bar that writes to stderr
def apply_with_progress(df, func, axis=1):
    results = []
    with tqdm(total=len(df), file=sys.stderr, desc="Processing") as pbar:
        def wrapped_func(*args, **kwargs):
            result = func(*args, **kwargs)
            pbar.update(1)
            return result
        
        # Apply with your wrapped function
        results = df.apply(wrapped_func, axis=axis)
    return results

'''
df_sent['alignment'] = apply_with_progress(
    df_sent,
    lambda row: align('y', 
                      'x',
                      row['lemma'].split(' '),
                      row['translation_lemma'].split(' ')),
    axis=1
)
'''


# print()
# print(df_sent.head(5), '\n')

# group by sentence_id and aggregate bn_gold values into a list
bn_gold_lists = (
    df_src.groupby("sentence_id")["bn_gold"]
       .apply(lambda x: [v for v in x])  # drop NaN
       .reset_index(name="bn_gold_list")
)


df_sent = df_sent.merge(bn_gold_lists, on="sentence_id", how="left")

# print()
# print(df_sent.head(5), '\n')
# print(df_sent.iloc[1], '\n')


# print('END PART 1.')




# src_data = args.src_data
# src_gold = args.src_gold





# print()
# print(df_sent.head(5), '\n')
# print(df_sent.iloc[1], '\n')


candidates_to_remember = []

def get_alignments(alignments, i):
 
  js = [link[1] for link in alignments if link[0] == i]
  return(js)

# print()  
for j, row in tqdm(df_sent.iterrows(), total=len(df_sent), desc="Aligning"):
  sid = row['sentence_id']
  src = row['lemma'].split(' ')
  tgt = row['translation_lemma'].split(' ')
  ali = align('y',
                      'x',
                      row['lemma'].split(' '),
                      row['translation_lemma'].split(' '))
  bns = row['bn_gold_list']
  # print('SID', sid)
  # print('TXT', row['text'])
  # print('SRC', src)
  # print('TGT', tgt)
  # print('ALI', ali)
  # print('BNs', bns)
  if not (len(src) == len(bns)):
    print('SRC / BNs length mismatch.')
    
    continue


  for i, bn in enumerate(bns):
    # print("\nLENGTH OF CANDIDATES IS", len(candidates_to_remember))
    if not str(bn)[:3] == 'bn:':
      # print("continuuing", bn)
      continue
    alignment_indices = get_alignments(ali, i)
    if len(alignment_indices) > 1:
      candidates = [ '_'.join( [ tgt[j] for j in alignment_indices ] ) ]
    elif len(alignment_indices) == 1:
      candidates = [ tgt[alignment_indices[0]] ]
    else:
      # print("No cands bc align", ali, i)
      candidates = []

    if candidates:
      for candidate in candidates:
        candidates_to_remember.append(('CANDIDATE', src[i], bn, candidate))
    

  # print()




final(args.beta, candidates_to_remember)
