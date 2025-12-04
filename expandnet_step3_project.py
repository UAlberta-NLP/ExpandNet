import argparse
import ast
import csv
import pandas as pd
import sys
import xml_utils

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--src_data", type=str, default="xlwsd_se13.xml",
                      help="Path to the XLWSD XML corpus file.")
  parser.add_argument("--src_gold", type=str, default="se13.key.txt",
                      help="Path to the gold sense tagging file.")
  parser.add_argument("--dictionary", type=str, default="wikpan-en-fr.tsv",
                      help="Use a dictionary for filtering. Available options: none, bn (BabelNet), wik (WiktExtract), wikpan (WiktExtract and PanLex)")
  parser.add_argument("--alignment_file", type=str, default="expandnet_step2_align.out.tsv",
                      help="File containing the output of step 2 (alignment).")
  parser.add_argument("--output_file", type=str, default="expandnet_step3_project.out.tsv")
  parser.add_argument("--token_info_file", type=str, default="expandnet_step3_project.token_info.tsv",
                      help="(Helpful for understanding the process undergone.)")
  parser.add_argument("--join_char", type=str, default='_')
  parser.add_argument(
    "--no_pos_screen",
    action="store_false",
    dest="pos_screen",
    help="Optionally turn OFF the filtering based on part-of-speech (default: filtering is ON)."
)
  return parser.parse_args()

args = parse_args()

csv.field_size_limit(sys.maxsize)

print(f"Source data:     {args.src_data}")
print(f"Source gold:     {args.src_gold}")
print(f"Dictionary:      {args.dictionary}")
print(f"Alignment file:  {args.alignment_file}")
print(f"Output file:     {args.output_file}")

# Load the dataset and alignment data.
print("Loading dataset...")
df_src = xml_utils.process_dataset(args.src_data, args.src_gold)
print(f"Dataset loaded: {len(df_src)} rows")

print("Loading alignment data...")
df_sent = pd.read_csv(args.alignment_file, sep='\t')
print(f"Alignment loaded: {len(df_sent)} sentences")

def load_dict(filepaths):
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
                eng_word = row[0].strip().lower().replace(' ', '_')  # Normalize English key
                fr_words = set(word.strip().lower().replace(' ', '_') for word in row[1].split())
                if eng_word in dict_:
                    dict_[eng_word].update(fr_words)  # Merge sets if key exists
                else:
                    dict_[eng_word] = fr_words
    return dict_
  
def pos_match(pos_a, pos_b):
  if pos_a is None or pos_b is None:
    return False
  if len(pos_b) == 1:
    if pos_a == pos_b[0] and pos_a != 'x':
      return True
    else:
      return False
  else:
    # TODO There are multiple ways to do this. Maybe ask about it, but for now... let's say 'in' is the way.
    return pos_b in pos_a and pos_b != 'x'
  
  
def pos_map(in_pos):
  
  if in_pos.lower() in ['a', 'v', 'r', 'n', 'x']:
    return in_pos.lower()
  
  POS_DICTIONARY = {'NOUN': 'n', 'PROPN': 'n', 'PRON': 'n', 'NUM': 'n',
   'VERB': 'v', 'AUX': 'v',
   'ADJ': 'a',
   'ADV': 'r', 'ADP': 'r', 'PART': 'r', 'SCONJ': 'r',
   'CCONJ': 'x', 'INTJ': 'x', 'SYM': 'x', 'PUNCT': 'x', 'DET': 'x', 'X': 'x'}
  
  return POS_DICTIONARY[in_pos]


def is_valid_translation(eng_word, fr_word, dict_, join_char, pos1=None, pos2=None, doing_pos_screening=True):
  """Check if (eng_word, fr_word) is a valid translation pair in the dict."""
  eng_word = eng_word.lower().strip().replace(' ', '_').replace('_', join_char)
  fr_word = fr_word.lower().strip().replace(' ', '_').replace('_', join_char)
  if eng_word not in dict_:
    return False
  
  return fr_word in dict_[eng_word] and (not doing_pos_screening or pos_match(pos1, pos2))

def write_the_stuff(file, tok, source, src_pos, t_candidate, candidate, bn, t_pos, join_char, tgt_sent, w):
  file.write(tok_id + '\t' + tok.replace(join_char, ' ') + '\t' + 
             source.replace(join_char, ' ') + '\t' + 
             src_pos.replace(join_char, ' ') + '\t' + t_candidate.replace(join_char, ' ') + '\t'  + 
             candidate.replace(join_char, ' ') + '\t' + 
             bn + '\t' + 
             str(is_valid_translation(source, candidate, dict_wik, join_char, src_pos, t_pos)) + '\t' + 
             str(bool(is_valid_translation(source, candidate, dict_wik, join_char, 'n', 'n') and not is_valid_translation(source, candidate, dict_wik, join_char, src_pos, t_pos))).upper() + '\t' + 
             tgt_sent.replace(join_char, ' ') + '\t' + w + '\n')

def get_alignments(alignments, i):
  """Get all target indices aligned to source index i."""
  return [link[1] for link in alignments if link[0] == i]

# Load the dictionary.
print("Loading dictionary...")
dict_wik = load_dict([args.dictionary])
print(f"Dictionary loaded")

# Group by sentence_id and aggregate bn_gold and lemma values into lists
print("Preparing data...")
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

# Merge back into df_sent
df_sent = (
    df_sent.merge(bn_gold_lists, on="sentence_id", how="left")
           .merge(lemma_gold_lists, on="sentence_id", how="left").merge(token_gold_lists, on="sentence_id", how="left")
)
print(f"Data prepared")

# Project senses
print("Projecting senses...")
senses = set()
with open(args.token_info_file, 'w', encoding='utf-8') as f:
 f.write("Token ID" + '\t' + "Source Token" + '\t' + "Source Lemma" + '\t' + "Source POS" + '\t' + "Translated Token" + '\t'  + "Translated Lemma" + '\t' + "Synset ID" + '\t' + "Link Valid? (According to POS and Dictionary)" + '\t' + "Dict OK but POS bad" + '\t' + 'Target Sentence'+ '\t' + 'Source Sentence' + '\n')
 for _, row in df_sent.iterrows():
  tok_num = 0
  src = row['lemma_gold']
  src_tok = row['token_gold']
  # assert len(src) == len(src_tok)
  tgt = row['translation_lemma'].split(' ')
  tgt_tok = row['translation_token'].split(' ')
  if args.pos_screen:
    tgt_pos = row['translation_pos'].split(' ')
    tgt_pos = [pos_map(a) for a in tgt_pos]
  else:
    tgt_pos = ['x' for _ in tgt_tok]
  assert len(tgt) == len(tgt_tok)
  ali = ast.literal_eval(row['alignment'])
  bns = row['bn_gold']
  sent_id = row['sentence_id']
  w = row['text']
  

  for i, bn in enumerate(bns):
    source = src[i]
    tok = src_tok[i]
    tok_id = sent_id + f".s{tok_num:03d}"
    if not str(bn)[:3] == 'bn:':
      f.write('wf' + '\t' + tok.replace(args.join_char, '_') + '\t' + source.replace(args.join_char, '_') + '\t' + ' ' + '\t'  + ' ' + '\t' + ' ' + '\t' + ' ' + '\n')
      continue
    src_pos = bn[-1]
    
    tok_num += 1
    alignment_indices = get_alignments(ali, i)
    if len(alignment_indices) > 1:
      candidates = [args.join_char.join([tgt[j] for j in alignment_indices])]
      t_candidates = [args.join_char.join([tgt_tok[j] for j in alignment_indices])]
      t_pos = args.join_char.join([tgt_pos[j] for j in alignment_indices])
    elif len(alignment_indices) == 1:
      candidates = [tgt[alignment_indices[0]]]
      t_candidates = [tgt_tok[alignment_indices[0]]]
      t_pos = tgt_pos[alignment_indices[0]]
    else:
      candidates = []
      t_candidates = []
      t_pos = 'x'

    if candidates:
      for t_candidate, candidate in zip(t_candidates, candidates):
        
        
        src_pos = bn[-1].lower()
        write_the_stuff(f, tok, source, src_pos, t_candidate, candidate, bn, t_pos, args.join_char, ' '.join(tgt_tok), w)
        
        if is_valid_translation(source, candidate, dict_wik, args.join_char, src_pos, t_pos, args.pos_screen):
          senses.add((bn, candidate))

print(f"Found {len(senses)} unique sense-lemma pairs")

print(f"Saving results to {args.output_file}...")
with open(args.output_file, 'w') as f:
  for (bn, lemma) in sorted(senses):
    print(bn, lemma.replace(args.join_char, ' '), sep='\t', file=f)

print('Complete!')

