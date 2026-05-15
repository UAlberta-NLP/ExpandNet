import argparse
import pandas as pd
import xml_utils
from tqdm import tqdm

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--translation_df_file", type=str, default="expandnet_step1_translate.out.tsv",
                      help="Path to the TSV file containing tokenized translated sentences.")
  parser.add_argument("--src_data", type=str, default="semcor_en.data.dev.xml",
                      help="Path to the XLWSD XML corpus file.")
  parser.add_argument("--lang_src", type=str, default="en", 
                      help="Source language (default: en).")
  parser.add_argument("--lang_tgt", type=str, default="fr", 
                      help="Target language (default: fr).")
  parser.add_argument("--dictionary", type=str, default="res/dicts/wikpan-en-fr.tsv",
                      help="Use a dictionary with DBAlign. This argument should be a path, the string 'bn' if you are using babelnet, or can be none if you are using simalign.")
  parser.add_argument("--aligner", type=str, default="dbalign",
                      help="Aligner to use ('simalign' or 'dbalign').")
  parser.add_argument("--output_file", type=str, default="expandnet_step2_align.out.tsv",
                      help="Output file to save the file with alignments to.")
  parser.add_argument("--num_workers", type=int, default=1,
                      help="Number of workers to paralellize the alignment computation over. More than one is not recommended on Windows or less powerful machines. (Default: 1)")
  parser.add_argument("--source_join_char", type=str, default='_')
  parser.add_argument("--target_join_char", type=str, default='_')
  
  return parser.parse_args()

def replace_lemma_with_gold(in_df, gold_file_path):
    """Syncs the input DF with gold XML data using sentence_id as the key."""
    df_src = xml_utils.process_xml(gold_file_path)
    df_gold = xml_utils.extract_sentences(df_src)

    if len(in_df) != len(df_gold):
        raise ValueError(f"Row mismatch: in_df ({len(in_df)}) vs gold ({len(df_gold)})")

    # Map both columns at once by setting the index to the join key
    mapping = df_gold.set_index('sentence_id')[['lemma', 'text']]
    
    # Update in_df using the mapping
    in_df = in_df.set_index('sentence_id')
    in_df.update(mapping)
    return in_df.reset_index()

class AlignerFactory:
    """Handles the setup and execution of different alignment backends."""
    def __init__(self, args):
        self.args = args
        self.aligner_type = args.aligner
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.aligner_type == 'simalign':
            from simalign import SentenceAligner
            return SentenceAligner(model="xlmr", layer=8, token_type="bpe", matching_methods="i")
        
        elif self.aligner_type == 'dbalign':
            from align_utils import DBAligner
            if self.args.dictionary == 'bn':
                return DBAligner(self.args.lang_src, self.args.lang_tgt)
            return DBAligner(
                self.args.lang_src, self.args.lang_tgt, 'custom', 
                self.args.dictionary, self.args.source_join_char, self.args.target_join_char
            )

    def align(self, row):
        # Pre-processing
        src_tokens = row['text'].split()
        tgt_tokens = row['translation_token'].split()
        src_lemmas = row['lemma'].split()
        tgt_lemmas = row['translation_lemma'].split()

        if self.aligner_type == 'simalign':
            return self.model.get_word_aligns(src_tokens, tgt_tokens)['itermax']
        
        # dbalign logic
        tgt_tokens = [t.replace(self.args.target_join_char, " ") for t in tgt_tokens]
        spans = self.model.new_align(src_tokens, tgt_tokens, src_lemmas, tgt_lemmas)
        return self._spans_to_links(spans)

    @staticmethod
    def _spans_to_links(span_string):
        links = []
        for s in span_string.strip().split():
            try:
                x_start, x_end, y_start, y_end = map(int, s.split('-'))
                for x in range(x_start, x_end + 1):
                    for y in range(y_start, y_end + 1):
                        links.append((x, y))
            except (ValueError, IndexError):
                continue
        return sorted(list(set(links)))

def main():
    args = parse_args()
    
    print(f"Loading data from {args.translation_df_file}...")
    df = pd.read_csv(args.translation_df_file, sep='\t')
    df = replace_lemma_with_gold(df, args.src_data)

    # Initialize Parallelism
    if args.num_workers > 1:
        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=True, nb_workers=args.num_workers)
        apply_method = df.parallel_apply
    else:
        tqdm.pandas()
        apply_method = df.progress_apply

    # Run Alignment
    engine = AlignerFactory(args)
    print(f"Aligning {len(df)} sentences using {args.aligner}...")
    df['alignment'] = apply_method(engine.align, axis=1)

    df.to_csv(args.output_file, sep='\t', index=False)
    print(f"Complete! Saved to {args.output_file}")

if __name__ == "__main__":
    main()