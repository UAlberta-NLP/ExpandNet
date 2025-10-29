#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python filter_by_sentences.py <sentence_ids.txt> <data.txt>")
        sys.exit(1)

    sentence_ids_file = sys.argv[1]
    data_file = sys.argv[2]

    # Load all allowed sentence IDs into a set for O(1) lookup
    with open(sentence_ids_file, 'r') as f:
        allowed_sentences = set(line.strip() for line in f if line.strip())

    # Process the data file
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Extract sentence ID: everything before the first ".t"
            first_dot_t = line.find('.t')
            if first_dot_t == -1:
                continue  # skip malformed lines
            sent_id = line[:first_dot_t]

            if sent_id in allowed_sentences:
                print(line)

if __name__ == '__main__':
    main()
