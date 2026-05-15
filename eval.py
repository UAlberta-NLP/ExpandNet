import argparse
import sys

ZERO_WIDTH_CHARS = [
    "\u200b", "\u200c", "\u200d", "\u200e", "\u200f",
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\ufeff"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ExpandNet output against gold standard.")
    parser.add_argument("file_gold", type=str, help="Path to the gold standard TSV file.")
    parser.add_argument("file_eval", type=str, help="Path to the evaluation TSV file.")
    parser.add_argument("--core_synsets", type=str, default="res/corebnout.txt",
                        help="Path to core synsets file (default: res/corebnout.txt)")
    return parser.parse_args()


def remove_zero_width_characters(s: str) -> str:
    for ch in ZERO_WIDTH_CHARS:
        s = s.replace(ch, "")
    return s


def lowercase_and_remove_zero_width(string: str) -> str:
    return remove_zero_width_characters(string.lower())


def harmonic_mean(a: float, b: float) -> float:
    return (2 * a * b) / (a + b) if (a + b) != 0 else 0.0


def safe_div(n: float, d: float) -> float:
    """Safely divide n by d, returning 0.0 if d is 0."""
    return n / d if d > 0 else 0.0


def file_to_pairs(filepath, bnid_dict):
    """Reads a TSV file into a list of tuples."""
    pairs = []
    nonempt_pairs = []
    seen = set()
    
    with open(filepath, 'r', encoding='utf-8') as fh:
        for line in fh:
            fields = lowercase_and_remove_zero_width(line).strip().split('\t')
            if len(fields) != 2:
                continue
            
            fields[1] = fields[1].replace(' ', '_')
            pair = tuple(fields)
            
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
                # Use .get() to prevent KeyErrors if pair[0] is not in bnid_dict
                if len(bnid_dict.get(pair[0], [])) > 0:
                    nonempt_pairs.append(pair)
                    
    return pairs, nonempt_pairs


def file_to_set(filepath):
    """Read a file into a set of non-empty cleaned lines."""
    with open(filepath, 'r', encoding='utf-8') as fh:
        return {lowercase_and_remove_zero_width(line).strip() for line in fh if line.strip()}


def main():
    args = parse_args()

    print(f"Gold file: {args.file_gold}")
    print(f"Eval file: {args.file_eval}")
    print(f"Core synsets: {args.core_synsets}\n")

    # Load gold standard
    print("Loading gold standard...")
    gold_bnid_to_lemmas = {}
    with open(args.file_gold, 'r', encoding='utf-8') as f:
        for line in f:
            line = lowercase_and_remove_zero_width(line.strip())
            if not line:
                continue
            try:
                parts = line.split('\t')
                key = parts[0]
                values = parts[1].split(' ') if len(parts) > 1 and parts[1] else []
                gold_bnid_to_lemmas[key] = values
            except Exception as e:
                print(f"Error processing line: {line}, error: {e}", file=sys.stderr)
                gold_bnid_to_lemmas[key] = []

    # Load evaluation and core synset data
    print("Loading evaluation data...")
    senses_for_eval, nonempty_senses_for_eval = file_to_pairs(args.file_eval, gold_bnid_to_lemmas)

    print("Loading core synsets...")
    core_synsets = file_to_set(args.core_synsets)
    print()

    # Get counts and report
    num_synsets_in_gold = len(gold_bnid_to_lemmas)
    num_synsets_in_gold_with_lemmas = sum(1 for v in gold_bnid_to_lemmas.values() if len(v) > 0)
    
    print(num_synsets_in_gold_with_lemmas)
    print(f'Source synsets to cover: {num_synsets_in_gold}')
    
    num_senses_for_eval = len(senses_for_eval)
    num_nonempty_senses_for_eval = len(nonempty_senses_for_eval)
    
    print(f'Senses to evaluate:      {num_senses_for_eval}')
    
    num_senses_covered = len(set(e[0] for e in senses_for_eval))
    print(f'Synsets covered:         {num_senses_covered}')
    
    # Replaced 'nonempty(e)' with inline dictionary check
    num_senses_covered_nonempty = len({e[0] for e in senses_for_eval if len(gold_bnid_to_lemmas.get(e[0], [])) > 0})
    assert num_senses_covered_nonempty <= num_senses_covered
    print()

    total_senses = sum(len(lems) for lems in gold_bnid_to_lemmas.values())
    num_lemmas_in_gold = total_senses  # Simplified from original loop

    correct_senses = 0
    synsets_with_correct_sense = set()
    synsets_present_in_output = set()
    
    for (bnid, lemma) in senses_for_eval:
        if bnid in gold_bnid_to_lemmas and lemma in gold_bnid_to_lemmas[bnid]:
            correct_senses += 1
            synsets_with_correct_sense.add(bnid)
            synsets_present_in_output.add(bnid)
        else:
            synsets_present_in_output.add(bnid)

    empty = set()
    uncovered = set()

    for bnid, lemmas in gold_bnid_to_lemmas.items():
        if bnid not in synsets_with_correct_sense:
            if len(lemmas) == 0:
                empty.add(bnid)
            else:
                uncovered.add(bnid)

    assert len(empty) + len(uncovered) + len(synsets_with_correct_sense) == num_synsets_in_gold
        
    num_synsets_with_correct_sense = len(synsets_with_correct_sense)
    num_synsets_with_projected_sense = len(synsets_present_in_output)

    print()

    ### SENSE-LEVEL EVALUATION
    sense_precision = safe_div(correct_senses, num_senses_for_eval)
    nonempty_sense_precision = safe_div(correct_senses, num_nonempty_senses_for_eval)
    sense_recall = safe_div(correct_senses, total_senses)
    sense_adj_recall = safe_div(num_senses_for_eval, total_senses) # Fixed: pred_senses was just a counter for loop length
    sense_f1 = safe_div(2 * sense_precision * sense_recall, sense_precision + sense_recall)
    
    bn_cov = safe_div(num_synsets_in_gold_with_lemmas, num_synsets_in_gold)
    bn_avg_syn_num = safe_div(num_lemmas_in_gold, num_synsets_in_gold_with_lemmas)
    concept_av_num = safe_div(num_nonempty_senses_for_eval, num_synsets_in_gold_with_lemmas)

    print(f"BN: avg number of lemmas per synset in this language is: {round(bn_avg_syn_num, 1)}\n")
    print(f"BN coverage of this language is: {round(100 * bn_cov, 1)}\n")
    print(f"Number of outputs, on average, for all nonempty concepts: {round(concept_av_num, 1)}\n")

    print(f"SENSE\tcorrect_senses:      {correct_senses}")
    print(f"SENSE\tnum_senses_for_eval: {num_senses_for_eval}")
    print(f"SENSE\ttotal_senses:        {total_senses}")
    print(f"SENSE\tPRECISION\t{round(100 * sense_precision, 1)}")
    print(f"\033[94mSENSE\tNONEMPTY PRECISION\t{round(100 * nonempty_sense_precision, 1)}\033[0m")
    print(f"SENSE\tRECALL\t{round(100 * sense_recall, 1)}")
    print(f"SENSE\tADJUSTED RECALL\t{round(100 * sense_adj_recall, 1)}")
    print(f"SENSE\tF1\t{round(100 * sense_f1, 1)}\n")

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
    print(f"SYNSET\tCORE COVERAGE\t{round(100 * core_coverage, 1)}\n")

    ne_f1 = harmonic_mean(synset_poss_recall, nonempty_sense_precision)
    print(f"\033[94mNE F1\t{round(100 * ne_f1, 1)}\033[0m")

    # Grouped results for cleaner file writing
    results = [
        correct_senses,
        round(100 * sense_precision, 1),
        round(100 * nonempty_sense_precision, 1),
        round(100 * sense_recall, 1),
        round(100 * sense_adj_recall, 1),
        num_synsets_with_correct_sense,
        round(100 * synset_precision, 1),
        round(100 * synset_nonempty_precision, 1),
        round(100 * synset_recall, 1),
        round(100 * synset_poss_recall, 1),
        round(100 * synset_adj_recall, 1)
    ]

    with open("RESULTSOUT.tsv", 'w', encoding='utf8') as outf:
        outf.write('\t'.join(map(str, results)) + '\n')


if __name__ == "__main__":
    main()