import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ExpandNet output against gold standard.")
    parser.add_argument("file_gold", type=str, help="Path to the gold standard TSV file.")
    parser.add_argument("file_eval", type=str, help="Path to the evaluation TSV file.")
    parser.add_argument("--core_synsets", type=str, default="res/corebnout.txt",
                        help="Path to core synsets file (default: dependencies/corebnout.txt)")
    return parser.parse_args()

args = parse_args()

def remove_zero_width_characters(s: str) -> str:
    
    ZERO_WIDTH_CHARS = [
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u202a",  # LEFT-TO-RIGHT EMBEDDING
        "\u202b",  # RIGHT-TO-LEFT EMBEDDING
        "\u202c",  # POP DIRECTIONAL FORMATTING
        "\u202d",  # LEFT-TO-RIGHT OVERRIDE
        "\u202e",  # RIGHT-TO-LEFT OVERRIDE
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE (BOM)
    ]
    for ch in ZERO_WIDTH_CHARS:
        s = s.replace(ch, "")
    return s


def harmonic_mean(a, b):
    if a+b == 0:
        return 0
    return (2*a*b)/(a+b)

def nonempty(arg):
   
    if len(gold_bnid_to_lemmas[arg[0]]) > 0:
        return True
    return False

    
def lowercase_and_remove_zero_width(string):
    return remove_zero_width_characters(string.lower())
# Reads a TSV file into a list of tuples.
def file_to_pairs(f, bnid_dict):
    pairs = []
    nonempt_pairs = []
    seen = set()
    with open(f, 'r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            fields = lowercase_and_remove_zero_width(line).strip().split('\t')
            
            
            if len(fields) != 2:
                continue
            fields[1] = fields[1].replace(' ', '_')
            pair = tuple(fields)
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
                if len(bnid_dict[pair[0]]) > 0:
                    nonempt_pairs.append(pair)
    return pairs, nonempt_pairs

def file_to_set(f):
    """Read a file into a set of lines."""
    with open(f, 'r', encoding='utf-8') as fh:
        return set(lowercase_and_remove_zero_width(line).strip() for line in fh if line.strip())

def safe_div(n, d):
    """Safely divide n by d, returning 0.0 if d is 0."""
    return n / d if d > 0 else 0.0

print(f"Gold file: {args.file_gold}")
print(f"Eval file: {args.file_eval}")
print(f"Core synsets: {args.core_synsets}\n")

# Read in the list of synsets to cover, and their gold contents.
print("Loading gold standard...")
gold_bnid_to_lemmas = {}
with open(args.file_gold, 'r', encoding='utf-8') as f:
    for line in f:
        line = lowercase_and_remove_zero_width(line.strip())
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
senses_for_eval, nonempty_senses_for_eval = file_to_pairs(args.file_eval, gold_bnid_to_lemmas)

print("Loading core synsets...")
core_synsets = file_to_set(args.core_synsets)
print()


# Get counts and report.
num_synsets_in_gold = len(gold_bnid_to_lemmas)
num_synsets_in_gold_with_lemmas = sum(1 for v in gold_bnid_to_lemmas.values() if len(v) > 0)

print(num_synsets_in_gold_with_lemmas)

print(f'Source synsets to cover: {num_synsets_in_gold}')
num_senses_for_eval, num_nonempty_senses_for_eval = len(senses_for_eval), len(nonempty_senses_for_eval)
print(f'Senses to evaluate:      {num_senses_for_eval}')
num_senses_covered = len(set(e[0] for e in senses_for_eval))
print(f'Synsets covered:         {num_senses_covered}')
num_senses_covered_nonempty = len(set(e[0] for e in senses_for_eval if nonempty(e)))

assert num_senses_covered_nonempty <= num_senses_covered
print()

total_senses = sum(len(gold_bnid_to_lemmas[bnid]) for bnid in gold_bnid_to_lemmas)

all_lemmas = []

num_lemmas_in_gold = 0
for key in gold_bnid_to_lemmas:
    
    lems = gold_bnid_to_lemmas[key]
    num_lemmas_in_gold += len(lems)
    all_lemmas += lems
    
    
    

correct_senses = 0
pred_senses = 0
synsets_with_correct_sense = set()
synsets_present_in_output = set()
for (bnid, lemma) in senses_for_eval:
    pred_senses += 1
    if bnid in gold_bnid_to_lemmas and lemma in gold_bnid_to_lemmas[bnid]:
        # print("GOOD_SENSE", bnid, lemma, sep='\t')
        correct_senses += 1
        synsets_with_correct_sense.add(bnid)
       
        synsets_present_in_output.add(bnid)
    else:
        # print("BADSENSE", lemma, bnid, gold_bnid_to_lemmas[bnid][:4], sep='\t')
        synsets_present_in_output.add(bnid)


empty = set()
uncovered = set()


for bnid in gold_bnid_to_lemmas.keys():
    if bnid not in synsets_with_correct_sense:
        if len(gold_bnid_to_lemmas[bnid]) == 0:
            empty.add(bnid)
        else:
            if bnid not in uncovered:
                # print(bnid)
                pass
            uncovered.add(bnid)
    

assert len(empty) + len(uncovered) + len(synsets_with_correct_sense) == num_synsets_in_gold
    
num_synsets_with_correct_sense = len(synsets_with_correct_sense)
num_synsets_with_projected_sense = len(synsets_present_in_output)

print()

### SENSE-LEVEL EVALUATION
sense_precision = safe_div(correct_senses, num_senses_for_eval)
nonempty_sense_precision = safe_div(correct_senses, num_nonempty_senses_for_eval)
sense_recall = safe_div(correct_senses, total_senses)

sense_adj_recall = safe_div(pred_senses, total_senses)
sense_f1 = safe_div(2 * sense_precision * sense_recall, sense_precision + sense_recall)
bn_cov = safe_div(num_synsets_in_gold_with_lemmas, num_synsets_in_gold)
bn_avg_syn_num = safe_div(num_lemmas_in_gold, num_synsets_in_gold_with_lemmas)

concept_av_num = safe_div(num_nonempty_senses_for_eval, num_synsets_in_gold_with_lemmas)

print(f"BN: avg number of lemmas per synset in this language is: {round(bn_avg_syn_num, 1)}")
print()

print(f"BN coverage of this language is: {round(100 * bn_cov, 1)}")
print()

print(f"Number of outputs, on average, for all nonempty concepts: {round(concept_av_num, 1)}")
print()

print(f"SENSE\tcorrect_senses:      {correct_senses}")
print(f"SENSE\tnum_senses_for_eval: {num_senses_for_eval}")
print(f"SENSE\ttotal_senses:        {total_senses}")
print(f"SENSE\tPRECISION\t{round(100 * sense_precision, 1)}")
print(f"\033[94mSENSE\tNONEMPTY PRECISION\t{round(100 * nonempty_sense_precision, 1)}\033[0m")
print(f"SENSE\tRECALL\t{round(100 * sense_recall, 1)}")

print(f"SENSE\tADJUSTED RECALL\t{round(100 * sense_adj_recall, 1)}")
print(f"SENSE\tF1\t{round(100 * sense_f1, 1)}")
print()

### SYNSET-LEVEL EVALUATION
synset_precision = safe_div(num_synsets_with_correct_sense, num_senses_covered)
synset_nonempty_precision = safe_div(num_synsets_with_correct_sense, num_senses_covered_nonempty)
synset_recall = safe_div(num_synsets_with_correct_sense, num_synsets_in_gold)
synset_adj_recall = safe_div(num_synsets_with_projected_sense, num_synsets_in_gold)
synset_poss_recall = safe_div(num_synsets_with_correct_sense, num_synsets_in_gold_with_lemmas)
synset_f1 = safe_div(2 * synset_precision * synset_recall, synset_precision + synset_recall)
core_coverage = safe_div(len(synsets_present_in_output & core_synsets), len(core_synsets))

print(f"SYNSET\tnum_synsets_with_correct_sense: {num_synsets_with_correct_sense}")
print(f"SYNSET\tnum_senses_covered:             {num_senses_covered}")
print(f"SYNSET\tnum_synsets_in_gold:            {num_synsets_in_gold}")
print(f"SYNSET\tPRECISION\t{round(100 * synset_precision, 1)}")
print(f"SYNSET\tNONEMPTY PRECISION\t{round(100 * synset_nonempty_precision, 1)}")
print(f"SYNSET\tRECALL\t{round(100 * synset_recall, 1)}")
print(f"\033[94mSYNSET\tNONEMPTY RECALL\t{round(100 * synset_poss_recall, 1)}\033[0m")
print(f"SYNSET\tADJUSTED RECALL\t{round(100 * synset_adj_recall, 1)}")
print(f"SYNSET\tF1\t{round(100 * synset_f1, 1)}")
print(f"SYNSET\tCORE COVERAGE\t{round(100 * core_coverage, 1)}")
print()

ne_f1 = harmonic_mean(synset_poss_recall, nonempty_sense_precision)

print(f"\033[94mNE F1\t{round(100 * ne_f1, 1)}\033[0m")

with open("RESULTSOUT.tsv", 'w', encoding='utf8') as outf:
    outf.write('\t'.join([str(a) for a in [correct_senses, round(100 * sense_precision, 1), round(100 * nonempty_sense_precision, 1), 
       round(100 * sense_recall, 1), round(100 * sense_adj_recall, 1), num_synsets_with_correct_sense, 
       round(100 * synset_precision, 1), round(100 * synset_nonempty_precision, 1),
       round(100 * synset_recall, 1), round(100 * synset_poss_recall, 1), round(100 * synset_adj_recall, 1)]]))
    
    
