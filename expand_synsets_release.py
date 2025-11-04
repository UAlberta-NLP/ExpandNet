import sys



lang_tgt    = sys.argv[1]
file_input  = sys.argv[2]
file_output = sys.argv[3]

import babelnet	as bn
from babelnet.resources import BabelSynsetID
from babelnet import Language
def get_synset(bnid, language):
  language = Language.from_iso(language)
  synset = bn.get_synset(BabelSynsetID(bnid))
  try:
    lemmas = [str(l) for l in synset.lemmas(language)]
  except:
    lemmas = []
  return(lemmas)


bn_lemmas = {}

with open(file_input, 'r') as f:
    total_lines = sum(1 for _ in f)
    
with open(file_input, 'r') as file:
  for i, line in enumerate(file):
    bnid = line.strip()
    pct = i / total_lines * 100
    
    print(f"Processed {i}/{total_lines} lines ({pct:.1f}%)")
    bn_lemmas[bnid] = get_synset(bnid, lang_tgt)
    # print(bn_lemmas[bnid])

with open(file_output, 'w') as f:
  for key in bn_lemmas.keys():
    
    f.write(key + '\t' + ' '.join(bn_lemmas[key]) + '\n')
        
      
