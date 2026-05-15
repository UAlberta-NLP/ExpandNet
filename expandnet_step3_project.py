import argparse
import ast
import csv
import pandas as pd
import sys
import xml_utils

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--src_data", type=str, default="semcor_en.data.dev.xml",
                      help="Path to the XLWSD XML corpus file.")
  parser.add_argument("--src_gold", type=str, default="semcor_en.gold.key.dev.txt",
                      help="Path to the gold sense tagging file.")
  parser.add_argument("--dictionary", type=str, default="res/dicts/wikpan-en-fr.tsv",
                      help="Use a dictionary for filtering. Available options: none, bn (BabelNet), wik (WiktExtract), wikpan (WiktExtract and PanLex)")
  parser.add_argument("--alignment_file", type=str, default="expandnet_step2_align.out.tsv",
                      help="File containing the output of step 2 (alignment).")
  parser.add_argument("--output_file", type=str, default="expandnet_step3_project.out.tsv")
  parser.add_argument("--pos_mapping_file", type=str, default="pos_mapping_u.tsv",
                      help="A file specifying how to convert POS tags to the 4 tags used by BN")
  parser.add_argument("--token_info_file", type=str, default="expandnet_step3_project.token_info.tsv",
                      help="(Helpful for understanding the process undergone.)")
  parser.add_argument("--source_join_char", type=str, default='_')
  parser.add_argument("--target_join_char", type=str, default='_')
  parser.add_argument(
    "--no_pos_screen",
    action="store_false",
    dest="pos_screen",
    help="Optionally turn OFF the filtering based on part-of-speech (default: filtering is ON)."
)
  parser.add_argument(
    "--no_ne_screen",
    action="store_false",
    dest="ne_screen",
    help="Optionally turn OFF the filtering of named entities (by caps) (default: filtering is ON)."
)
  parser.add_argument(
    "--no_dict_screen",
    action="store_false",
    dest="dict_screen",
    help="Optionally turn OFF the dictionary filtering (default: filtering is ON)."
)
  parser.add_argument(
    "--no_oov_screen",
    action="store_false",
    dest="oov_screen",
    help="Optionally, allow the projections whose English value isn't in the dictionary (default: OOV English terms are NOT projected)."
)
  return parser.parse_args()

args = parse_args()

max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int //= 10

print("JOIN CHAR IS '" + args.source_join_char + "' ON THE SOURCE SIDE, '" +\
  args.target_join_char + "' ON THE TARGET SIDE")
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

def load_dict(filepaths, sjc, tjc):
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
                eng_word = row[0].strip().lower().replace(' ', '_').replace('_', sjc)  # Normalize English key
                fr_words = set(word.strip().lower().replace(' ', '_').replace('_', tjc) for word in row[1].split())
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
    return pos_a == pos_b[0] and pos_a != 'x'
  
  
def pos_map(in_pos):
  
  if in_pos.lower() in ['a', 'v', 'r', 'n', 'x']:
    return in_pos.lower()
  
  try:
    
    return POS_DICTIONARY[in_pos]
  except:
    assert False, "INVALID POS: " + in_pos


def safe_replace(s, old, new):
    if old == "":
        return s
    return s.replace(old, new)

def listify(l):
  if isinstance(l, list):
    return l
  else:
    return []


def is_valid_translation(eng_orig_tok, eng_word, fr_word, dict_, source_join_char, target_join_char, mask_obj, is_mwe, pos1, pos2):
  """Check if (eng_word, fr_word) is a valid translation pair in the dict."""
  
  if mask_obj['screen_ne'] and eng_orig_tok[0].isupper():
    return False
  eng_word = eng_word.lower().strip().replace(' ', '_').replace('_', source_join_char)
  fr_word = fr_word.lower().strip().replace(' ', '_').replace('_', target_join_char)

  
  if mask_obj['screen_oov'] and eng_word not in dict_:
    return False
  elif eng_word not in dict_:
    return True
  
  if mask_obj['screen_dict'] and fr_word not in dict_[eng_word]:
    if fr_word != eng_word:
      return False
  
  if mask_obj['screen_pos'] and not pos_match(pos1, pos2):
    if not is_mwe:
      return False
    
  
  return True

def load_pos_mapping(address):
  ans = {}
  with open(address, 'r', encoding='utf-8') as f:
    for line in f:
      if line.strip():
        to, froms = line.strip().split('\t')
        for element in froms.split():
          ans[element] = to
  return ans

def write_to_file(file, tok, source, src_pos, t_pos_longer, t_candidate, candidate, bn, t_pos, source_join_char, target_join_char, tgt_sent, w, mask_ob, is_mwe):
  
  file.write(tok_id + '\t' + safe_replace(tok, source_join_char, ' ') + '\t' + 
             safe_replace(source, source_join_char, ' ') + '\t' + 
             src_pos.replace('_', ' ') + '\t' + t_pos_longer.replace('_', ' ') + '(' + t_pos.replace('_', ' ') + ')' + '\t' + safe_replace(t_candidate, target_join_char, ' ') + '\t'  + 
             safe_replace(candidate, target_join_char, ' ') + '\t' + 
             bn + '\t' + 
             str(is_valid_translation(tok, source, candidate, dict_wik, source_join_char, target_join_char, mask_ob, is_mwe, 'n', 'n')) + '\t' + 
             str(bool(pos_match(src_pos, t_pos))).upper() + '\t' + 
             safe_replace(tgt_sent, target_join_char, ' ') + '\t' + w + '\n')
 
  
def get_alignments(alignments, i):
  """Get all target indices aligned to source index i."""
  return [link[1] for link in alignments if link[0] == i]

# Load the dictionary.
print("Loading dictionary...")
dict_wik = load_dict([args.dictionary], args.source_join_char, args.target_join_char)
print(f"Dictionary loaded")

print("Loading pos mapping...")
POS_DICTIONARY = load_pos_mapping(args.pos_mapping_file)
print("pos mapping loaded")                              

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

mask_object = {'screen_ne': args.ne_screen,
               'screen_oov': args.oov_screen,
               'screen_dict': args.dict_screen,
               'screen_pos': args.pos_screen,
               }

# Project senses
print("Projecting senses...")
senses = set()
with open(args.token_info_file, 'w', encoding='utf-8') as f:
 f.write("Token ID" + '\t' + "Source Token" + '\t' + "Source Lemma" + '\t' + "Source POS" + '\t'  + "Target POS" + '\t' + "Translated Token" + '\t'  + "Translated Lemma" + '\t' + "Synset ID" + '\t' + "Link in Dictionary?" + '\t' + "POS Match?" + '\t' + 'Target Sentence'+ '\t' + 'Source Sentence' + '\n')
 for _, row in df_sent.iterrows():
  tok_num = 0
  src = row['lemma_gold']
  src_tok = row['token_gold']
  tgt = row['translation_lemma'].split(' ')
  tgt_tok = row['translation_token'].split(' ')
  if args.pos_screen:
    try:
      tgt_pos = row['translation_pos'].split(' ')
   
    except KeyError:
      print("ERROR: no translation_pos column found. " 
            "Check that you have one, or turn off the part-of-speech"
            " filter using the flag --no_pos_screen") 
      exit(-1)
  else:
    tgt_pos = ['x' for _ in tgt_tok]
  ali = ast.literal_eval(row['alignment'])
  bns = row['bn_gold']
  sent_id = row['sentence_id']
  w = row['text']
  
  for i, bn in enumerate(listify(bns)):
    source = src[i]
    tok = src_tok[i]
    tok_id = sent_id + f".s{tok_num:03d}"
   
    if not str(bn)[:3] == 'bn:':

      f.write('wf' + '\t' + tok.replace(args.source_join_char, '_').replace('_', args.source_join_char) + '\t' + source.replace(args.source_join_char, '_').replace('_', args.source_join_char) + '\t' + ' ' + '\t'  + ' ' + '\t' + ' ' + '\t' + ' ' + '\n')
    

      continue
    src_pos = bn[-1]
    
    tok_num += 1
    alignment_indices = get_alignments(ali, i)
    if len(alignment_indices) > 1:
      candidates = [args.target_join_char.join([tgt[j] for j in alignment_indices])]
      t_candidates = [args.target_join_char.join([tgt_tok[j] for j in alignment_indices])]
      t_pos = args.target_join_char.join([pos_map(tgt_pos[j]) for j in alignment_indices])
      target_pos_orig = args.target_join_char.join([str(tgt_pos[j]) for j in alignment_indices])
    elif len(alignment_indices) == 1:
      candidates = [tgt[alignment_indices[0]]]
      t_candidates = [tgt_tok[alignment_indices[0]]]
      t_pos = pos_map(tgt_pos[alignment_indices[0]])
      target_pos_orig = str(tgt_pos[alignment_indices[0]])
    else:
      candidates = []
      t_candidates = []
      t_pos = 'x'
      target_pos_orig = 'X'
      
    
    if candidates:
      for t_candidate, candidate in zip(t_candidates, candidates):
        
        
        src_pos = bn[-1].lower()
        write_to_file(f, tok, source, src_pos, target_pos_orig, t_candidate, candidate, bn, t_pos, args.source_join_char, args.target_join_char, args.target_join_char.join(tgt_tok), w, mask_object, len(alignment_indices) > 1)
       
        if is_valid_translation(tok, source, candidate, dict_wik, args.source_join_char, args.target_join_char, mask_object, len(alignment_indices) > 1, src_pos, t_pos):
       
          senses.add((bn, candidate))
    else:
      f.write(tok_id + '\t' + tok.replace(args.source_join_char, '_').replace('_', args.source_join_char) + '\t' + source.replace(args.source_join_char, '_').replace('_', args.source_join_char) + '\t' + ' ' + '\t'  + ' ' + '\t' + ' ' + '\t' + ' ' + '\n')
     
print(f"Found {len(senses)} unique sense-lemma pairs")

print(f"Saving results to {args.output_file}")
with open(args.output_file, 'w', encoding='utf-8') as f:
  for (bn, lemma) in sorted(senses):
    print(bn, safe_replace(lemma, args.target_join_char, ' '), sep='\t', file=f)

print('Complete!')

