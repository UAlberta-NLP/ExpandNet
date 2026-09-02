import pandas as pd
import argparse
from tqdm import tqdm

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--input_file", type=str, default="res/uwn.tsv", 
                      help="File containing UWN.")
  parser.add_argument("--language", type=str, required=True,
                      help="Three letter 639-3 language code. See https://iso639-3.sil.org/code_tables/639/data or https://en.wikipedia.org/wiki/List_of_ISO_639-3_codes to find 639-3 language codes.")
  parser.add_argument("--output_file", type=str, default="uwn-to-bn.tsv", 
                      help="File to save outputs to.")
  parser.add_argument("--data_file", type=str, default='res/scdev_gold_senses.txt',
                      help="File containing sysnet ids for all the senses which we care about (i.e., all the concepts in SemCor20)")

  return parser.parse_args()

args = parse_args()

def get_synsets_in_data(file_addr):
  ans = set()
  with open(file_addr, 'r', encoding='utf-8') as f:
    for row in f:
      ans.add(row.strip())
  return ans
  


language = args.language
output_file = args.output_file
input_file = args.input_file
data_file = args.data_file

# Read the data, select only the rows and columns we want.
df = pd.read_csv(input_file, sep='\t', names=['col1', 'relation', 'col2', 'score'])
df = df.loc[ df['col1'].str.startswith(f't/{language}') ]
df = df.loc[ df['relation'] == 'rel:means' ]
df = df.loc[ df['col2'].str.startswith('s/') ]
df = df[['col1','col2']].rename( columns={ 'col1':'lemma', 'col2':'pwn_synset' } )


# Reset indices -- we're done filtering rows.
df.reset_index(drop=True, inplace=True)


# Process the columns to get plain lemmas, and the offset and pos of the synset.
df['lemma'] = df['lemma'].str.replace(f't/{language}/', '', regex=True)
df[['pwn_synset_pos','pwn_synset_offset']] \
  = df['pwn_synset'].str.extract(r'^s/([a-z])(\d+)$')
df['pwn_synset_offset'] = df['pwn_synset_offset'].astype('str').str.zfill(8)
df.drop(columns='pwn_synset', inplace=True)


# Now convert the pwn_synset columns into proper wn synset offsets.
df['wn_synset'] = 'wn:' + df['pwn_synset_offset'] + df['pwn_synset_pos']
df.drop(columns=['pwn_synset_offset', 'pwn_synset_pos'], inplace=True)


# Now map those PWN synsets to BN synsets!
import babelnet as bn
from babelnet.resources import WordNetSynsetID

synsets_in_data = get_synsets_in_data(data_file)

def wn_to_bn(wn: str):
  return( str( bn.get_synset(WordNetSynsetID(wn)).id ) )

tqdm.pandas()

df['bn_synset'] = df['wn_synset'].progress_apply(wn_to_bn)


# Prep for final output.
df = df[['bn_synset','lemma']]
with open(output_file, 'w', encoding='utf-8') as f:
 for index, row in df.iterrows():
  if row['bn_synset'] in synsets_in_data:
    f.write(row['bn_synset'] + '\t' + row['lemma'] + '\n')
    
print(df.sample(10))
