import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ExpandNet output against gold standard.")
    parser.add_argument("file_gold", type=str, help="Path to the gold standard TSV file.")
    parser.add_argument("file_eval", type=str, help="Path to the evaluation TSV file.")
    parser.add_argument("--core_synsets", type=str, default="dependencies/corebnout.txt",
                        help="Path to core synsets file (default: dependencies/corebnout.txt)")
    return parser.parse_args()

args = parse_args()

# Reads a TSV file into a list of tuples.
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

def file_to_set(f):
    """Read a file into a set of lines."""
    with open(f, 'r') as fh:
        return set(line.strip() for line in fh if line.strip())

def safe_div(n, d):
    """Safely divide n by d, returning 0.0 if d is 0."""
    return n / d if d > 0 else 0.0

print(f"Gold file: {args.file_gold}")
print(f"Eval file: {args.file_eval}")
print(f"Core synsets: {args.core_synsets}\n")

# Read in the list of synsets to cover, and their gold contents.
print("Loading gold standard...")
gold_bnid_to_lemmas = {}
with open(args.file_gold, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:  # skip empty lines
            continue
        try:
            parts = line.split('\t')
            key = parts[0]
            if len(parts) > 1 and parts[1]:  # has values and not empty
                values = parts[1].split(' ')
            else:
                values = []
            gold_bnid_to_lemmas[key] = values
        except Exception as e:
            print(f"Error processing line: {line}, error: {e}", file=sys.stderr)
            gold_bnid_to_lemmas[key] = []

# Read the senses to be evaluated (into a list of pairs).
print("Loading evaluation data...")
senses_for_eval = file_to_pairs(args.file_eval)

print("Loading core synsets...")
core_synsets = file_to_set(args.core_synsets)
print()


# Get counts and report.
num_synsets_in_gold = len(gold_bnid_to_lemmas)
print(f'Source synsets to cover: {num_synsets_in_gold}')
num_senses_for_eval = len(senses_for_eval)
print(f'Senses to evaluate:      {num_senses_for_eval}')
num_senses_covered = len(set(e[0] for e in senses_for_eval))
print(f'Synsets covered:         {num_senses_covered}')
print()

total_senses = sum(len(gold_bnid_to_lemmas[bnid]) for bnid in gold_bnid_to_lemmas)

correct_senses = 0
synsets_with_correct_sense = set()
synsets_present_in_output = set()
for (bnid, lemma) in senses_for_eval:
    if bnid in gold_bnid_to_lemmas and lemma in gold_bnid_to_lemmas[bnid]:
        # print("GOOD_SENSE", bnid, lemma, sep='\t')
        correct_senses += 1
        synsets_with_correct_sense.add(bnid)
        synsets_present_in_output.add(bnid)
    else:
        # print("BAD_SENSE", bnid, lemma, sep='\t')
        synsets_present_in_output.add(bnid)

num_synsets_with_correct_sense = len(synsets_with_correct_sense)

print()

### SENSE-LEVEL EVALUATION
sense_precision = safe_div(correct_senses, num_senses_for_eval)
sense_recall = safe_div(correct_senses, total_senses)
sense_f1 = safe_div(2 * sense_precision * sense_recall, sense_precision + sense_recall)

print(f"SENSE\tcorrect_senses:      {correct_senses}")
print(f"SENSE\tnum_senses_for_eval: {num_senses_for_eval}")
print(f"SENSE\ttotal_senses:        {total_senses}")
print(f"SENSE\tPRECISION\t{round(100 * sense_precision, 1)}")
print(f"SENSE\tRECALL\t{round(100 * sense_recall, 1)}")
print(f"SENSE\tF1\t{round(100 * sense_f1, 1)}")
print()

### SYNSET-LEVEL EVALUATION
synset_precision = safe_div(num_synsets_with_correct_sense, num_senses_covered)
synset_recall = safe_div(num_synsets_with_correct_sense, num_synsets_in_gold)
synset_f1 = safe_div(2 * synset_precision * synset_recall, synset_precision + synset_recall)
core_coverage = safe_div(len(synsets_present_in_output & core_synsets), len(core_synsets))

print(f"SYNSET\tnum_synsets_with_correct_sense: {num_synsets_with_correct_sense}")
print(f"SYNSET\tnum_senses_covered:             {num_senses_covered}")
print(f"SYNSET\tnum_synsets_in_gold:            {num_synsets_in_gold}")
print(f"SYNSET\tPRECISION\t{round(100 * synset_precision, 1)}")
print(f"SYNSET\tRECALL\t{round(100 * synset_recall, 1)}")
print(f"SYNSET\tF1\t{round(100 * synset_f1, 1)}")
print(f"SYNSET\tCORE COVERAGE\t{round(100 * core_coverage, 1)}")
print()

