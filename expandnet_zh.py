#!/usr/bin/env python3
"""
ExpandNet for Chinese (en→zh)

This script runs the ExpandNet pipeline for English-to-Chinese synset expansion using:
- Pre-computed GPT-5 translations from xlwsd_semcor_en_zh_gpt-5-chat-latest.tsv
- Gold BabelNet synsets from scdev_gold_zh.pkl
- SemCor development set
"""

# built-in
import sys
import os
import csv
import argparse
import logging
# public
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
# private
import xml_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Run ExpandNet for Chinese (en→zh)")
    parser.add_argument("--src_data", type=str, default="semcor_en.data.dev.xml",
                        help="Path to the source XML corpus file.")
    parser.add_argument("--src_gold", type=str, default="semcor_en.gold.key.dev.txt",
                        help="Path to the gold sense tagging file.")
    parser.add_argument("--translation_df_file", type=str,
                        default="xlwsd_semcor_en_zh_gpt-5-chat-latest.tsv",
                        help="Path to pre-computed translation TSV file.")
    parser.add_argument("--lang_src", type=str, default="en",
                        help="Source language (default: en).")
    parser.add_argument("--lang_tgt", type=str, default="zh",
                        help="Target language (default: zh).")
    parser.add_argument("--aligner", type=str, default="simalign",
                        help="Aligner to use ('simalign' or 'dbalign').")
    parser.add_argument("--aligndict", type=str, default="none",
                        help="Dictionary for alignment (dbalign only): none, bn, wik, wikpan, cedict, ecdict, ceec, cow (default: none)")
    parser.add_argument("--filterdict", type=str, default="none",
                        help="Dictionary for filtering candidates: none, bn, wik, wikpan, cedict, ecdict, ceec (default: none)")
    parser.add_argument("--output_file", type=str, default="expandnet_zh_output.txt",
                        help="Output file for full processing log.")
    parser.add_argument("--nb_workers", type=int, default=5,
                        help="Number of parallel workers (default: 5).")
    parser.add_argument("--alignment_file", type=str, default=None,
                        help="File to save/load alignment results (TSV format). If exists, will load instead of recomputing.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Log file to save job logs. If not specified, uses {output_file}.log")

    return parser.parse_args()


def tokenize_sentence(sentence: str, lang: str, lemmatize: bool = False, pipelines: dict = None) -> list:
    """
    Tokenize sentence using spaCy.

    For Chinese, we use zh_core_web_lg or zh_core_web_sm.
    Note: Chinese doesn't have traditional lemmatization like English.
    """
    import spacy

    if pipelines is None:
        pipelines = {}

    if lang not in pipelines:
        model_map = {
            'en': 'en_core_web_lg',
            'zh': 'zh_core_web_lg',
        }

        model_name = model_map.get(lang, f"{lang}_core_web_lg")

        try:
            nlp = spacy.load(model_name)
            print(f"Loaded spaCy model: {model_name}", file=sys.stderr)
        except OSError:
            # Fallback to small model
            fallback_model = model_name.replace('_lg', '_sm')
            print(f"Model {model_name} not found, trying {fallback_model}", file=sys.stderr)
            nlp = spacy.load(fallback_model)

        pipelines[lang] = nlp

    doc = pipelines[lang](sentence)
    if lemmatize:
        return [token.lemma_ for token in doc]
    else:
        return [token.text for token in doc]


def get_alignments(alignments, i):
    """Get all target indices aligned to source index i"""
    js = [link[1] for link in alignments if link[0] == i]
    return js

def load_dict(filepaths):
    """Load multiple TSV files into a dict: {english_word: set(french_words)}.
    All spaces are normalized to underscores.
    """
    # increase max field size limit to system max
    csv.field_size_limit(sys.maxsize)
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

def is_valid_translation(src_word, tgt_word, dict_):
    """Check if tgt_word is a valid translation of src_word using the provided dictionary."""
    src_word = src_word.lower().strip().replace(' ', '_')
    tgt_word = tgt_word.lower().strip().replace(' ', '_')
    if src_word not in dict_:
        return False
    else:
        # print("Wiki dict hit:", src_word, "->", dict_[src_word], file=sys.stderr)
        return tgt_word in dict_[src_word]

def main():
    args = parse_args()

    # Configuration
    lang_src = args.lang_src
    lang_tgt = args.lang_tgt
    src_data = args.src_data
    src_gold = args.src_gold
    translation_df_file = args.translation_df_file
    aligner = args.aligner
    aligndict = args.aligndict
    filterdict = args.filterdict
    output_file = args.output_file
    sense_file = output_file + '.senses'

    # Setup logging
    log_file = args.log_file if args.log_file else output_file + '.log'

    # Create file handler with immediate flushing
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Setup logger with both handlers
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False  # Prevent duplicate logging to root logger

    # Chinese doesn't use lemmatization
    lemmatize = False if lang_tgt in ['zh'] else True

    # Construct dictionary file paths based on usedict setting
    dict_file_wik = f'wiktextract-{lang_src}-{lang_tgt}.tsv'
    dict_file_wikpan = f'wikpan-{lang_src}-{lang_tgt}.tsv'
    dict_file_panlex = f'panlex-{lang_src}-{lang_tgt}.tsv'
    dict_file_cedict = f'cedict-{lang_src}-{lang_tgt}.tsv'
    dict_file_ecdict = f'ecdict-{lang_src}-{lang_tgt}.tsv'
    dict_file_ceec = f'ceec-{lang_src}-{lang_tgt}.tsv'
    dict_file_cow = f'cow-{lang_src}-{lang_tgt}.tsv'

    logger.info(f"Running on: {lang_src} -> {lang_tgt}")
    logger.info(f"Corpus: {src_data}")
    logger.info(f"Gold tags: {src_gold}")
    logger.info(f"Translation file: {translation_df_file}")
    logger.info(f"Aligner: {aligner}")
    logger.info(f"Alignment dictionary: {aligndict}")
    logger.info(f"Filter dictionary: {filterdict}")
    logger.info(f"Lemmatize target: {lemmatize}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Log file: {log_file}")

    # 1. Load source data
    logger.info("Loading source data...")
    df_src = xml_utils.process_dataset(src_data, src_gold)
    logger.info(f"Total tokens: {len(df_src)}")

    # 2. Load pre-computed translations
    logger.info(f"Loading translations from {translation_df_file}...")
    if not os.path.exists(translation_df_file):
        logger.error(f"Translation file {translation_df_file} not found!")
        sys.exit(1)

    df_sent = pd.read_csv(translation_df_file, sep='\t')
    logger.info(f"Translations loaded: {len(df_sent)} sentences")

    # 3. Setup tokenizer
    logger.info("Setting up tokenizer...")
    pipelines = {}

    # Test Chinese tokenization
    test_zh = df_sent['translation'].iloc[0] if len(df_sent) > 0 else "测试"
    test_tokens = tokenize_sentence(test_zh, lang_tgt, lemmatize=lemmatize, pipelines=pipelines)
    logger.info(f"Test tokenization: {test_zh}")
    logger.info(f"Tokens: {test_tokens}")

    # 4. Setup aligner
    logger.info(f"Setting up aligner: {aligner}")

    if aligner == 'simalign':
        from simalign import SentenceAligner
        ali = SentenceAligner(model="xlmr", layer=8, token_type="bpe", matching_methods="i")

        def align(tokens_src, tokens_tgt):
            alignment_links = ali.get_word_aligns(tokens_src, tokens_tgt)['itermax']
            return alignment_links

        logger.info("Using simalign with XLM-R")

    elif aligner == 'dbalign':
        from babelnetalign import DBAligner

        if aligndict == 'bn':
            logger.info("Initializing DBAlign with BabelNet.")
            ali = DBAligner(lang_src, lang_tgt)
        elif aligndict == 'wik':
            logger.info(f"Initializing DBAlign with WiktExtract: {dict_file_wik}")
            ali = DBAligner(lang_src, lang_tgt, 'custom', dict_file_wik)
        elif aligndict == 'wikpan':
            logger.info(f"Initializing DBAlign with Wik+Pan: {dict_file_wikpan}")
            ali = DBAligner(lang_src, lang_tgt, 'custom', dict_file_wikpan)
        elif aligndict == 'cedict':
            logger.info(f"Initializing DBAlign with CC-CEDICT: {dict_file_cedict}")
            ali = DBAligner(lang_src, lang_tgt, 'custom', dict_file_cedict)
        elif aligndict == 'ecdict':
            logger.info(f"Initializing DBAlign with EC-ECDICT: {dict_file_ecdict}")
            ali = DBAligner(lang_src, lang_tgt, 'custom', dict_file_ecdict)
        elif aligndict == 'ceec':
            logger.info(f"Initializing DBAlign with CEEC: {dict_file_ceec}")
            ali = DBAligner(lang_src, lang_tgt, 'custom', dict_file_ceec)
        elif aligndict == 'cow':
            logger.info(f"Initializing DBAlign with COW: {dict_file_cow}")
            ali = DBAligner(lang_src, lang_tgt, 'custom', dict_file_cow)
        else:
            logger.info("Initializing DBAlign with default settings.")
            ali = DBAligner(lang_src, lang_tgt)

        def spans_to_links(span_string):
            span_string = span_string.strip()
            span_list = span_string.split(' ')
            links = set()
            for s in span_list:
                try:
                    (x_start, x_end, y_start, y_end) = s.split('-')
                    for x in range(int(x_start), int(x_end)+1):
                        for y in range(int(y_start), int(y_end)+1):
                            links.add((x, y))
                except:
                    pass
            return sorted(links)

        def align(tokens_src, tokens_tgt):
            alignment_spans = ali.new_align(tokens_src, tokens_tgt)
            return spans_to_links(alignment_spans)

        logger.info("Using dbalign")


    # 5. Perform word alignment
    # Determine alignment file path
    if args.alignment_file:
        alignment_file = args.alignment_file
    else:
        # alignment_file = f"gpt_{lang_src}_{lang_tgt}_{aligner}_{aligndict}.tsv"
        # alignment_file = f"helsinki_{lang_src}_{lang_tgt}_{aligner}_{aligndict}.tsv"
        alignment_file = f"gt_{lang_src}_{lang_tgt}_{aligner}_{aligndict}.tsv"

    # Check if alignment file exists
    if os.path.exists(alignment_file):
        logger.info(f"Loading existing alignments from {alignment_file}...")
        df_sent_with_alignment = pd.read_csv(alignment_file, sep='\t')

        # Convert alignment column from string back to list of tuples
        import ast
        df_sent_with_alignment['alignment'] = df_sent_with_alignment['alignment'].apply(ast.literal_eval)

        # Merge alignment back into df_sent
        df_sent = df_sent_with_alignment
        logger.info(f"Loaded alignments for {len(df_sent)} sentences")
    else:
        logger.info(f"Computing word alignments (will save to {alignment_file})...")
        pandarallel.initialize(progress_bar=False, nb_workers=args.nb_workers)

        def apply_with_progress(df, func, axis=1):
            results = []
            with tqdm(total=len(df), file=sys.stderr, desc="Aligning") as pbar:
                def wrapped_func(*args, **kwargs):
                    result = func(*args, **kwargs)
                    pbar.update(1)
                    return result

                results = df.apply(wrapped_func, axis=axis)
            return results

        df_sent['alignment'] = apply_with_progress(
            df_sent,
            lambda row: align(row['lemma'].split(' '),
                              tokenize_sentence(row['translation'], lang_tgt, lemmatize=lemmatize, pipelines=pipelines)),
            axis=1
        )

        logger.info("Alignment complete!")

        # Save alignment results
        logger.info(f"Saving alignment results to {alignment_file}...")
        df_sent.to_csv(alignment_file, sep='\t', index=False)
        logger.info("Save complete.")
    
    # 6. Prepare BabelNet gold annotations
    logger.info("Preparing BabelNet gold annotations...")
    bn_gold_lists = (
        df_src.groupby("sentence_id")["bn_gold"]
           .apply(lambda x: [v for v in x])
           .reset_index(name="bn_gold_list")
    )

    df_sent = df_sent.merge(bn_gold_lists, on="sentence_id", how="left")
    logger.info("BabelNet gold annotations merged.")

    # 7. Extract synset expansions
    logger.info("Extracting synset expansions...")
    logger.info(f"Total sentences to process: {len(df_sent)}")

    # Load dictionary if needed for filtering
    if filterdict == 'wik':
        logger.info(f"Loading WiktExtract dictionary from: {dict_file_wik}")
        dict_filter = load_dict([dict_file_wik])
    elif filterdict == 'wikpan':
        dict_files = [dict_file_wik, dict_file_panlex]
        logger.info(f"Loading WiktExtract+PanLex dictionaries from: {dict_files}")
        dict_filter = load_dict(dict_files)
    elif filterdict == 'cedict':
        logger.info(f"Loading CC-CEDICT dictionary from: {dict_file_cedict}")
        dict_filter = load_dict([dict_file_cedict])
    elif filterdict == 'ecdict':
        logger.info(f"Loading EC-ECDICT dictionary from: {dict_file_ecdict}")
        dict_filter = load_dict([dict_file_ecdict])
    elif filterdict == 'ceec':
        logger.info(f"Loading CEEC dictionary from: {dict_file_ceec}")
        dict_filter = load_dict([dict_file_ceec])
    elif filterdict == 'cow':
        logger.info(f"Loading COW dictionary from: {dict_file_cow}")
        dict_filter = load_dict([dict_file_cow])

    # from bn_utils import bnsyn, synonyms, bnsyn_save

    # # Save initial cache
    # bnsyn_save()

    # Progress tracking
    total_sentences = len(df_sent)
    log_interval = max(100, total_sentences // 20)  # Log every 5% or 100 sentences
    processed_count = 0

    # Open output files
    with open(output_file, 'w', encoding='utf-8') as f_out:
        with open(sense_file, 'w', encoding='utf-8') as f_sense:

            for idx, row in tqdm(df_sent.iterrows(), total=len(df_sent), desc="Processing sentences", file=sys.stderr):
                processed_count += 1
                sid = row['sentence_id']
                src = row['lemma'].split(' ')
                tgt = tokenize_sentence(row['translation'], lang_tgt, lemmatize=lemmatize, pipelines=pipelines)
                ali = row['alignment']
                bns = row['bn_gold_list']

                # Log progress periodically
                if processed_count % log_interval == 0:
                    progress_pct = (processed_count / total_sentences) * 100
                    logger.info(f"Progress: {processed_count}/{total_sentences} sentences ({progress_pct:.1f}%)")
                    # Force flush to ensure immediate write
                    for handler in logger.handlers:
                        handler.flush()

                # Print to stdout and file
                output_lines = []
                output_lines.append(f'SID\t{sid}')
                output_lines.append(f'TXT\t{row["text"]}')
                output_lines.append(f'SRC\t{src}')
                output_lines.append(f'TGT\t{tgt}')
                output_lines.append(f'ALI\t{ali}')
                output_lines.append(f'BNs\t{bns}')

                # Check length match
                if not (len(src) == len(bns)):
                    output_lines.append('ERROR\tSRC / BNs length mismatch.')
                    for line in output_lines:
                        f_out.write(line + '\n')
                    f_out.write('\n')
                    continue

                # Process each token with BabelNet ID
                for i, bn in enumerate(bns):
                    if not str(bn)[:3] == 'bn:':
                        continue

                    alignment_indices = get_alignments(ali, i)

                    # Build candidates
                    if len(alignment_indices) > 1:
                        if lang_tgt in ['zh']:
                            # For Chinese: first token + concatenated version
                            # candidates = [tgt[alignment_indices[0]]]
                            candidates = [''.join([tgt[j] for j in alignment_indices])]
                            # candidates += ['+'.join([tgt[j] for j in alignment_indices])]
                        else:
                            candidates = ['_'.join([tgt[j] for j in alignment_indices])]
                    elif len(alignment_indices) == 1:
                        candidates = [tgt[alignment_indices[0]]]
                    else:
                        candidates = []

                    if candidates:
                        for candidate in candidates:
                            source = src[i]

                            # Apply dictionary filter if specified
                            if filterdict == 'bn':
                                filter_pass = False
                                if source.lower() == candidate.lower():
                                    filter_pass = True
                                elif bnsyn(lang_src, lang_tgt, source, candidate):
                                    filter_pass = True
                            elif filterdict in ['wik', 'wikpan', 'cedict', 'ecdict', 'ceec', 'cow']:
                                filter_pass = is_valid_translation(source, candidate, dict_filter)
                            else:
                                # No filtering
                                filter_pass = 'n/a'

                            line = f'\t{i}\t{bn}\t{src[i]}\t{alignment_indices}\t{candidate}\t{filter_pass}'
                            output_lines.append(line)

                            if bool(filter_pass):
                                sense_line = f'SENSE\t{bn}\t{candidate}'
                                output_lines.append(sense_line)
                                f_sense.write(sense_line + '\n')
                    else:
                        line = f'\t{i}\t{bn}\t{src[i]}\t{alignment_indices}\tNO_CANDIDATES\tFalse'
                        output_lines.append(line)

                # Write all lines for this sentence
                for line in output_lines:
                    f_out.write(line + '\n')
                f_out.write('\n')

    # Save BabelNet synonym cache
    # logger.info("Saving BabelNet synonym cache...")
    # bnsyn_save()
    logger.info('Synset extraction DONE!')
    logger.info(f'Full output saved to: {output_file}')
    logger.info(f'SENSE lines saved to: {sense_file}')

    # 8. Post-process and deduplicate
    logger.info("Post-processing and deduplicating...")

    with open(sense_file, 'r', encoding='utf-8') as f:
        sense_lines = f.readlines()

    logger.info(f"Total SENSE lines: {len(sense_lines)}")

    # Extract unique (bnid, lemma) pairs
    unique_senses = set()
    for line in sense_lines:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            bnid = parts[1]
            lemma = parts[2]
            unique_senses.add((bnid, lemma))

    logger.info(f"Unique SENSE pairs: {len(unique_senses)}")

    # Save deduplicated results
    dedup_file = sense_file.replace('.senses', '.dedup.senses')

    with open(dedup_file, 'w', encoding='utf-8') as f:
        for bnid, lemma in sorted(unique_senses):
            f.write(f'{bnid}\t{lemma}\n')

    logger.info(f"Deduplicated results saved to: {dedup_file}")

    # 9. Optional evaluation
    gold_file = 'scdev_gold_zh.pkl'
    if os.path.exists(gold_file):
        logger.info(f"Evaluating against {gold_file}...")
        logger.info("=" * 60)
        import subprocess
        result = subprocess.run(
            ['python', 'eval_v02.py', gold_file, dedup_file],
            capture_output=True,
            text=True
        )

        # Log evaluation results line by line for better formatting
        if result.stdout:
            logger.info("Evaluation Results:")
            for line in result.stdout.strip().split('\n'):
                logger.info(line)

        if result.stderr:
            logger.warning(f"Evaluation errors: {result.stderr}")

        logger.info("=" * 60)
    else:
        logger.info(f"Gold file {gold_file} not found. Skipping evaluation.")

    logger.info("All processing complete!")


if __name__ == '__main__':
    main()
