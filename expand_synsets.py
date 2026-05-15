import sys
from tqdm import tqdm


lang_tgt    = sys.argv[1]
file_input  = sys.argv[2]
file_output = sys.argv[3]

import babelnet	as bn
from babelnet.resources import BabelSynsetID
from babelnet import Language

CACHE = {}
def get_synset(bnid, language):
  
  key = (bnid, language)
  if key not in CACHE:
    language = Language.from_iso(language)
    synset = bn.get_synset(BabelSynsetID(bnid))
    try:
    # print("LEMMAS:", lemmas)
      lemmas = [str(l) for l in synset.lemmas(language)]
    
    except:
      lemmas = []
    CACHE[key] = lemmas
  return CACHE[key]


bn_lemmas = {}

    
with open(file_input, 'r') as file:
  for line in tqdm(file.readlines()):
      x = line.strip()
      bnid = x.strip()
      
      if len(bnid.split()) == 1:
        pass
      else:
        bnid = bnid.split()[1]
      
    
      bn_lemmas[bnid] = get_synset(bnid, lang_tgt)
      # print(bn_lemmas[bnid])


with open(file_output, 'w') as f:
  for key in sorted(bn_lemmas.keys()):
    
    f.write(key + '\t' + ' '.join(bn_lemmas[key]) + '\n')
        
      