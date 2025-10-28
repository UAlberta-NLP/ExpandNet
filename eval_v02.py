import sys
file_gold = sys.argv[1]
file_eval = sys.argv[2]


# Reads a file into a simple list of lines (if not TSV) or list of tuples (if it is. 
# Assumes that all lines are unique, so pipe input files through 'sort | uniq' first.
def file_to_pairs(f):
    pairs = []
    seen = set()
    with open(f, 'r') as fh:
        for i, line in enumerate(fh):
            fields = line.strip().split('\t')
            if len(fields) != 2:
                raise ValueError(f"File {f}, line {i+1}: expected 2 fields, got {len(fields)}")
            pair = tuple(fields)
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
    return pairs



# Read in the list of synsets to cover, and their gold contents.
import pickle
with open(file_gold, 'rb') as f:
  gold_bnid_to_lemmas = pickle.load(f)
# Read the senses to be evaluated (into a list of pairs).
senses_for_eval = file_to_pairs(file_eval) 


# Get counts and report.
num_synsets_in_gold = len(gold_bnid_to_lemmas.keys()) # denom. for synset recall
print('Source synsets to cover:', num_synsets_in_gold)
num_senses_for_eval = len(senses_for_eval) # denom. for sense precision
print('Senses to evaluate:     ', num_senses_for_eval)
num_senses_covered = len( set( [e[0] for e in senses_for_eval] ) )
print('Synsets covered:        ', num_senses_covered) # denom. for synset precision
print()


total_senses = 0 # denom. for sense recall
for bnid in gold_bnid_to_lemmas:
  total_senses += len(gold_bnid_to_lemmas[bnid])


correct_senses = 0 # num. for sense eval
synsets_with_correct_sense = set()
for (bnid, lemma) in senses_for_eval:
  if bnid in gold_bnid_to_lemmas and lemma in gold_bnid_to_lemmas[bnid]:
    print("GOOD_SENSE",bnid,lemma,sep='\t')
    correct_senses += 1
    synsets_with_correct_sense.add(bnid)
  else:
    print("BAD_SENSE",bnid,lemma,sep='\t')
num_synsets_with_correct_sense = len(synsets_with_correct_sense) # num. for synset eval


def safe_div(n, d):
  return n / d if d > 0 else 0.0

print()

### SENSE-LEVEL EVALUATION
sense_precision = safe_div(correct_senses, num_senses_for_eval)
sense_recall    = safe_div(correct_senses,  total_senses)
sense_f1 = safe_div( (2 * sense_precision * sense_recall),
                     (sense_precision + sense_recall) )

print(f"SENSE\tcorrect_senses:        {correct_senses}")
print(f"SENSE\tnum_senses_for_eval:   {num_senses_for_eval}")
print(f"SENSE\ttotal_senses:          {total_senses}")
print('SENSE', 'PRECISION', round(100*sense_precision, 1), sep='\t')
print('SENSE', 'RECALL   ', round(100*sense_recall,    1), sep='\t')
print('SENSE', 'F1       ', round(100*sense_f1,        1), sep='\t')
print()


### SYNSET-LEVEL EVALUATION
synset_precision = safe_div(num_synsets_with_correct_sense, num_senses_covered)
synset_recall    = safe_div(num_synsets_with_correct_sense, num_synsets_in_gold)
synset_f1 = safe_div( (2 * synset_precision * synset_recall), 
                      (synset_precision + synset_recall) )

print(f"SYNSET\tnum_synsets_with_correct_sense: {num_synsets_with_correct_sense}")
print(f"SYNSET\tnum_senses_covered:    {num_senses_covered}")
print(f"SYNSET\tnum_synsets_in_gold:   {num_synsets_in_gold}")
print('SYNSET', 'PRECISION', round(100*synset_precision, 1), sep='\t')
print('SYNSET', 'RECALL   ', round(100*synset_recall,    1), sep='\t')
print('SYNSET', 'F1       ', round(100*synset_f1,        1), sep='\t')
print()

