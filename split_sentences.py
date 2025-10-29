# split_sentences.py
input_file = "semcor_en.sentences.txt" # List of sentece IDs.
dev_file   = "semcor_en.sentences.dev.txt"  # 20%
test_file  = "semcor_en.sentences.test.txt" # 80%

with open(input_file, 'r') as f:
    sentences = [line.strip() for line in f if line.strip()]

with open(dev_file, 'w') as dev_f, open(test_file, 'w') as test_f:
    for i, sent_id in enumerate(sentences):
        if (i % 5) == 0:  # e.g., indices 0, 5, 10, 15, ...
            dev_f.write(sent_id + '\n')
        else:
            test_f.write(sent_id + '\n')
