import sys
from collections import defaultdict

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 aggregate_filter.py <beta>", file=sys.stderr)
        print("  beta: float between 0.0 and 1.0 (e.g., 0.9)", file=sys.stderr)
        sys.exit(1)

    try:
        beta = float(sys.argv[1])
        if not (0.0 <= beta <= 1.0):
            raise ValueError
    except ValueError:
        print("Error: beta must be a float between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    # Step 1: Aggregate counts
    # Key: (english_lemma, synset_id) -> { french_lemma: count }
    sense_to_french = defaultdict(lambda: defaultdict(int))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 3:
            continue  # Skip malformed lines

        en_lemma, synset_id, fr_lemma = parts
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
    for line in output_lines:
        print(line)

if __name__ == "__main__":
    main()
