import argparse
import ast
import csv
import logging
import sys
from pathlib import Path
import pandas as pd
import xml_utils

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def listify(l):
    if isinstance(l, list):
        return l
    else:
        return []
    
def set_csv_limit():
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int //= 10

class SenseProjector:
    def __init__(self, args):
        self.args = args
        self.pos_map = self.load_pos_mapping(args.pos_mapping_file)
        self.dictionary = self.load_dictionary([args.dictionary])
        
    def load_pos_mapping(self, path):
        mapping = {}
        if not Path(path).exists():
            return mapping
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    target, sources = line.strip().split('\t')
                    for s in sources.split():
                        mapping[s] = target
        return mapping

    def load_dictionary(self, paths):
        """Loads and normalizes dictionary keys/values."""
        compiled_dict = {}
        for path in paths:
            if not Path(path).exists(): continue
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) < 2: continue
                    # Normalize to space internally, convert to join_char only on output
                    eng = row[0].strip().lower().replace('_', ' ')
                    fr = set(w.strip().lower().replace('_', ' ') for w in row[1].split())
                    compiled_dict.setdefault(eng, set()).update(fr)
        return compiled_dict

    def is_valid(self, eng_tok, eng_lemma, fr_lemma, is_mwe, src_pos, tgt_pos):
        # 1. Named Entity Screen
        if self.args.ne_screen and eng_tok and eng_tok[0].isupper():
            return False
        
        eng_key = eng_lemma.lower().replace(self.args.source_join_char, ' ')
        fr_key = fr_lemma.lower().replace(self.args.target_join_char, ' ')

        # 2. OOV Screen
        if self.args.oov_screen and eng_key not in self.dictionary:
            return False
        
        # 3. Dictionary Screen
        if self.args.dict_screen and eng_key in self.dictionary:
            if fr_key not in self.dictionary[eng_key] and fr_key != eng_key:
                return False

        # 4. POS Screen
        if self.args.pos_screen and not is_mwe:
            if src_pos != tgt_pos or src_pos == 'x':
                return False
                
        return True

    def get_alignments(self, ali, idx):
        """Extract target indices mapped to a source index safely."""
        if isinstance(ali, dict):
            return ali.get(idx, [])
        elif isinstance(ali, list):
            # Assume list of (src, tgt) tuples
            if len(ali) > 0 and isinstance(ali[0], (tuple, list)) and len(ali[0]) >= 2:
                return [tgt for src, tgt in ali if src == idx]
            # Assume list of lists where index is source
            elif len(ali) > idx and isinstance(ali[idx], list):
                return ali[idx]
        return []

    def process(self):
        print("Loading dictionary...")
        print(f"Dictionary loaded")

        print("Loading pos mapping...")
        print("pos mapping loaded")                              

        # Project senses
        print("Projecting senses...")
        args = self.args

        print("Loading dataset...")
        df_src = xml_utils.process_dataset(args.src_data, args.src_gold)
        print(f"Dataset loaded: {len(df_src)} rows")

        print("Loading alignment data...")
        df_sent = pd.read_csv(args.alignment_file, sep='\t')
        print(f"Alignment loaded: {len(df_sent)} sentences")

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
                   .merge(lemma_gold_lists, on="sentence_id", how="left")
                   .merge(token_gold_lists, on="sentence_id", how="left")
        )
        print("Data prepared")

        mask_object = {'screen_ne': args.ne_screen,
                       'screen_oov': args.oov_screen,
                       'screen_dict': args.dict_screen,
                       'screen_pos': args.pos_screen,
                       }

        senses = set()
        with open(args.token_info_file, 'w', encoding='utf-8') as f:
            headers = ["Token ID", "Source Token", "Source Lemma", "Source POS", "Target POS", 
                       "Translated Token", "Translated Lemma", "Synset ID", "Link in Dictionary?", 
                       "POS Match?", "Target Sentence", "Source Sentence"]
            f.write('\t'.join(headers) + '\n')
            
            for _, row in df_sent.iterrows():
                tok_num = 0
                src = listify(row.get('lemma_gold', []))
                src_tok = listify(row.get('token_gold', []))
                
                # Check for NaNs and properly split
                tgt = str(row['translation_lemma']).split(' ') if pd.notnull(row.get('translation_lemma')) else []
                tgt_tok = str(row['translation_token']).split(' ') if pd.notnull(row.get('translation_token')) else []
                
                if args.pos_screen:
                    try:
                        tgt_pos = str(row['translation_pos']).split(' ')
                    except KeyError:
                        print("ERROR: no translation_pos column found. " 
                              "Check that you have one, or turn off the part-of-speech filter using the flag --no_pos_screen") 
                        exit(-1)
                else:
                    tgt_pos = ['x' for _ in tgt_tok]
                    
                ali = ast.literal_eval(row['alignment']) if pd.notnull(row.get('alignment')) else []
                bns = row.get('bn_gold', [])
                sent_id = row['sentence_id']
                w = str(row['text'])

                for i, bn in enumerate(listify(bns)):
                    source = src[i]
                    tok = src_tok[i]
                    tok_id = f"{sent_id}.s{tok_num:03d}"

                    if not str(bn)[:3] == 'bn:':
                        f.write(f"wf\t{tok.replace(args.source_join_char, '_').replace('_', args.source_join_char)}\t"
                                f"{source.replace(args.source_join_char, '_').replace('_', args.source_join_char)}"
                                f"\t \t \t \t \n")
                        continue
                        
                    src_pos = bn[-1]
                    tok_num += 1
                    alignment_indices = self.get_alignments(ali, i)
                    
                    if len(alignment_indices) > 1:
                        candidates = [args.target_join_char.join([tgt[j] for j in alignment_indices if j < len(tgt)])]
                        t_candidates = [args.target_join_char.join([tgt_tok[j] for j in alignment_indices if j < len(tgt_tok)])]
                        t_pos = args.target_join_char.join([self.pos_map.get(tgt_pos[j], 'x') for j in alignment_indices if j < len(tgt_pos)])
                        target_pos_orig = args.target_join_char.join([str(tgt_pos[j]) for j in alignment_indices if j < len(tgt_pos)])
                    elif len(alignment_indices) == 1:
                        idx = alignment_indices[0]
                        candidates = [tgt[idx]] if idx < len(tgt) else []
                        t_candidates = [tgt_tok[idx]] if idx < len(tgt_tok) else []
                        t_pos = self.pos_map.get(tgt_pos[idx], 'x') if idx < len(tgt_pos) else 'x'
                        target_pos_orig = str(tgt_pos[idx]) if idx < len(tgt_pos) else 'X'
                    else:
                        candidates = []
                        t_candidates = []
                        t_pos = 'x'
                        target_pos_orig = 'X'

                    if candidates:
                        for t_candidate, candidate in zip(t_candidates, candidates):
                            src_pos_lower = bn[-1].lower()
                            is_mwe = len(alignment_indices) > 1
                            
                            # Determine flags for logging output
                            eng_key = source.lower().replace(args.source_join_char, ' ')
                            fr_key = candidate.lower().replace(args.target_join_char, ' ')
                            in_dict = str(fr_key in self.dictionary.get(eng_key, set()))
                            pos_match_str = str(src_pos_lower == t_pos)

                            f.write(f"{tok_id}\t{tok}\t{source}\t{src_pos_lower}\t{target_pos_orig}\t"
                                    f"{t_candidate}\t{candidate}\t{bn}\t{in_dict}\t{pos_match_str}\t"
                                    f"{args.target_join_char.join(tgt_tok)}\t{w}\n")
                            
                            if self.is_valid(tok, source, candidate, is_mwe, src_pos_lower, t_pos):
                                senses.add((bn, candidate))
                    else:
                        f.write(f"{tok_id}\t{tok.replace(args.source_join_char, '_').replace('_', args.source_join_char)}\t"
                                f"{source.replace(args.source_join_char, '_').replace('_', args.source_join_char)}"
                                f"\t \t \t \t \n")

        print(f"Found {len(senses)} unique sense-lemma pairs")

        print(f"Saving results to {args.output_file}")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for (bn, lemma) in sorted(senses):
                print(bn, lemma.replace(args.target_join_char, ' '), sep='\t', file=f)

        print('Complete!')

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

if __name__ == "__main__":
    set_csv_limit()
    cmd_args = parse_args()
    projector = SenseProjector(cmd_args)
    projector.process()