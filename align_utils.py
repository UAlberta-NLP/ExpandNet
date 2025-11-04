# from alignment_evaluation_script import evaluate, stringify_aligns
import pickle

import spacy
from simalign import SentenceAligner
# from babelnet.data.relation import BabelPointer
from pathlib import Path
HERE = Path(__file__).parent



SIMALIGNER = SentenceAligner(model="xlmr", layer=8, token_type="bpe", matching_methods="i")
# 



STEM_CACHE = {}

POS_CACHE = {}

LEMMATIZERS = {'es': spacy.load("es_core_news_lg"), 
               #'it': spacy.load("it_core_news_lg"), 
               'fr': spacy.load("fr_core_news_lg"), 
               'en': spacy.load("en_core_web_lg"), 
               # 'ro': spacy.load("ro_core_news_lg"), 
               'zh': spacy.load("zh_core_web_lg"),
            #    'xx': spacy.load('xx_ent_wiki_sm')
                }

TAGDICT = {'NP': 'n', 'VBD': 'v', 'SPACE': 'x', 'DET': 'x', 'JJ': 'a', 'NN': 'n', 'NNS': 'n', 'ADV': 'r', 'ADJ': 'a', 'PRON': 'n', 'AUX': 'v', 'CCONJ': 'x', 'SCONJ': 'x', 'X': 'x',
           'VB': 'v', 'PP$': 'x', 'RB': 'r', 'VBN': 'v', 'IN': 'x', ',': 'x', 'SENT': 'x', 'PP': 'x', 'VERB': 'v', 'PROPN': 'n', 'NOUN': 'n', 'SYM': 'x', 'INTJ': 'x',
           'CD': 'x', 'VBZ': 'v', 'CC': 'x', 'MD': 'x', 'TO': 'x', 'PDT': 'x', 'WRB': 'x', 'POS': 'x', 'ADP': 'x', 'PUNCT': 'x', 'NUM': 'x', 'PART': 'x'}

PUNCTUATION = {'¿', '`', '.', ',', '!', '?', '.', ':', ';', '*', ' ', " ", '-', '«', 
               '.', ',', '!', '?', ':', ';', '*', '-', '—', '(', ')', '[', ']', '{', '}', 
    '"', "'", '«', '»', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '_', '+', '«', '»', '¿', '?', 
    '°', '¡', '.', ',', '!', '?', '.', ':', ';', '*', ' ', " ", '-', '«', 
               '.', ',', '!', '?', ':', ';', '*', '-', '—', '(', ')', '[', ']', '{', '}', 
    '"', "'", '«', '»', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '_', '+', 
    '=', '<', '>', '~', '`', '’', '、', '。', '。', '、', '·'
    '=', '<', '>', '~', '`'
}
DO_NOT_CALL_BABELNET = {'«', '»', '¿', '?', '°', '¡', '.', ',', '!', '?', '.', ':', ';', '*', ' ', " ", '-', '«', 
               '.', ',', '!', '?', ':', ';', '*', '-', '—', '(', ')', '[', ']', '{', '}', 
    '"', "'", '«', '»', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '_', '+', "''"
    '=', '<', '>', '~', '`', '’', '、', '。', '。', '、', '·', '©', '·', '¢', '``', '``'} 



ALIGNMENT_CACHE = {}
SYNSET_CACHED_DICT = {}

    
class DBAligner:
    def __init__(self, src_l, tar_l, dictionary='BN', path_to_dict=None):
        self.source_language = src_l
        self.target_language = tar_l
        if dictionary.lower() == 'BN'.lower():
            self.dict_in_use = 'bn'
            self.dict = {}
            self.load_cache()
            global LANGS
            import babelnet as bn
            from babelnet import Language
            from babelnet.pos import POS
            LANGS = {"tr": Language.TR, "es": Language.ES, "it": Language.IT, "ar": Language.AR, "de": Language.DE,
           "fr": Language.FR, "en": Language.EN, "ja": Language.JA, "th": Language.TH, "zh": Language.ZH,
          " ko": Language.KO, "ro": Language.RO }
            global POS_TAGS
            POS_TAGS = {"VERB": POS.VERB, "NOUN": POS.NOUN, "ADV": POS.ADV, "ADJ": POS.ADJ, "PROPN": POS.NOUN}
            
        elif dictionary.lower() == 'custom':
            self.dict_in_use = 'custom'
            self.dict = {}
            self.load_dict(path_to_dict)
            
        else:
            assert False, "please pass third argument as either 'BN' or 'Custom'"
            
    def are_synonyms_by_dictionary(self, word_one, word_two, lang_two, lang_one):
        if self.dict_in_use == 'custom':
            return self.are_synonyms_by_custom(word_one, word_two, lang_two, lang_one)
        elif self.dict_in_use == 'bn':
            return are_synonyms_by_bn(word_one, word_two, lang_two, lang_one)
        else:
            assert False
            
    def are_synonyms_by_custom(self, first_word, second_word, second_lang, first_lang):
      if first_word in self.dict and second_word in self.dict[first_word]:
          # print('according to custom dictionary,', first_word, ' is a synonym of ', second_word)
          return 'strict'
      if first_word in ENGLISH_FUNCTION_WORDS and first_lang == 'en':
        lemma1 = first_word
        lemma2 = second_word
      elif '_' in second_word or '_' in first_word:
        for guy in first_word.split('_') + second_word.split('_'):
            
                if guy in PUNCTUATION:
                    # print(guy, "WAS PUNCTUATION SO FALSE")
                    return False
                
       
        lemma1 = '_'.join([get_lemma(a, first_lang) for a in first_word.split('_')])
        lemma2 = '_'.join([get_lemma(a, second_lang) for a in second_word.split('_')])
      else:
        lemma1 = get_lemma(first_word, first_lang)
        lemma2 = get_lemma(second_word, second_lang)
        
      
      if lemma1 in self.dict and (lemma2 in self.dict[lemma1] or second_word in self.dict[lemma1] or (first_word in self.dict and lemma2 in self.dict[first_word])):
          # print('according to custom dictionary,', first_word, ' is a synonym of ', second_word)
          return 'loose'
      return False
  
    def using_custom(self):
        if self.dict_in_use == 'custom':
            return True
        return False
    
    def load_cache(self):
        global SYNSET_CACHED_DICT
        try:
            with open(HERE / "pickles/bninfo.pkl", "rb") as f:
                print('load synset cache')
            
                SYNSET_CACHED_DICT = pickle.load(f)
        except FileNotFoundError:
            print('making new synset cache')
            SYNSET_CACHED_DICT = {}
    

    def save_cache(self):    
        with open(HERE / "pickles/bninfo.pkl", "wb") as f:
            pickle.dump(SYNSET_CACHED_DICT, f)
        # with open("pickles/aligns.pkl", "wb") as f:
            # pickle.dump(ALIGNMENT_CACHE, f)
            
    
    def load_dict(self, addr):
        found_words = set()
        with open(addr, 'r', encoding='utf-8') as infile:
            for line in infile:
                stripped_line = line.strip()
                assert len(stripped_line.split('\t')) == 2 or len(stripped_line) == 0, "Expected each line in dictionary file to have one tab character. The offending line is: " + str(line)
                input_word, translations = stripped_line.split('\t')
                input_word = input_word.replace(' ', '_')
                    # print("Warning: Expected word", input_word, "to appear in dictionary file only once. Parsing will continue, but you should check your file")
                found_words.add(input_word)
                for translation in translations.split():
                    if input_word in self.dict:
                        self.dict[input_word].append(translation)
                    else:
                        self.dict[input_word] = [translation]
                        
    def new_align(self, src, tgt, steps=3):
    # print(source)
    # print(target)
    
    
        if isinstance(src, str):
            source_words = src.split()
        elif isinstance(src, list):
            source_words = src.copy()
        
        if isinstance(tgt, str):
            target_words = tgt.split()
        elif isinstance(tgt, list):
            target_words = tgt.copy()
    
        aligns = AlignlistObj([], source_words, target_words)
    
        unal_source_indices = list(range(len(source_words)))
        unal_target_indices = list(range(len(target_words)))
    
    
    
        if steps > 0:
            # intersection_pass(source_words, target_words, aligns, unal_source_indices, unal_target_indices, True)
            intersection_pass(self.source_language, self.target_language, source_words, target_words, aligns, unal_source_indices, unal_target_indices, False, self)
        
            pass

    # 
        if steps > 1:
            babelmwe_pass(self.source_language, self.target_language, source_words, target_words, aligns, unal_source_indices, unal_target_indices, False, self)
            babelnet_pass(self.source_language, self.target_language, source_words, target_words, aligns, unal_source_indices, unal_target_indices, False, self)
        
            pass
        so_far = aligns.string_version()
        if steps > 2:
            simalign_pass(self.source_language, self.target_language, source_words, target_words, aligns, unal_source_indices, unal_target_indices, self)   
        # print(aligns.string_version()) 
    
    # answer = string_subtract(aligns.string_version(), so_far)
        answer = aligns.string_version()
    # print(stringify_aligns(answer, source, target))
        # print("ANSWER:", answer)
        return answer
                
    
    
def sort_by_first_number(s: str) -> str:
    groups = s.split()  # split by spaces
    # sort by the first number in each group
    groups.sort(key=lambda g: int(g.split('-')[0]))
    return ' '.join(groups) 
    
class AlignlistObj:
  def __init__(self, list_start, source, target):
      # print(list_start)
      # print(source)
      # print(target)
      self.source = source
      self.target = target
      self.alignments = list_start
    
  def conflict(self, quadrad):
      # print(quadrad)
      if len(quadrad) == 4:
        s_start, s_end, t_start, t_end = quadrad
      else:
          assert len(quadrad) == 2
          s_start, s_end, t_start, t_end = quadrad[0], quadrad[0], quadrad[1], quadrad[1]
      for other_quadrad in self.alignments:
          # print(other_quadrad, 'VS', quadrad)
          if other_quadrad[0] <= s_start <= other_quadrad[1] or other_quadrad[0] <= s_end <= other_quadrad[1]:
              return True
          if s_start <= other_quadrad[0] <= s_end or s_start <= other_quadrad[1] <= s_end:
              return True
          if other_quadrad[2] <= t_start <= other_quadrad[3] or other_quadrad[2] <= t_end <= other_quadrad[3]:
              return True
          if t_start <= other_quadrad[2] <= t_end or t_start <= other_quadrad[3] <= t_end:
              return True
      return False
  
  
  
      
        
  def print_this(self, pair):
      ans = ''
      if isinstance(pair[0], int) and isinstance(pair[1], int):
          s_start, s_end, t_start, t_end = pair[0], pair[0], pair[1], pair[1]
        
      elif isinstance(pair[0], int) and isinstance(pair[1], list):
            s_start, s_end = pair[0], pair[0]
            t_start, t_end = min(pair[1]), max(pair[1])
      elif isinstance(pair[0], list) and isinstance(pair[1], int):
            s_start, s_end = min(pair[0]), max(pair[0])
            t_start, t_end = pair[1], pair[1]
      elif isinstance(pair[0], list) and isinstance(pair[1], list):
          s_start, s_end = min(pair[0]), max(pair[0])
          t_start, t_end = min(pair[1]), max(pair[1])
      else:
          # print(pair)
          assert False
      for i in range(s_start, s_end + 1):
        ans += self.source[i] + ' '
        
      ans += ' ALIGNED WITH'
      for j in range(t_start, t_end + 1):
        ans += ' ' + self.target[j] 
      # print(ans)
    
  def add(self, pair):
      
      if isinstance(pair[0], int) and isinstance(pair[1], int):
          s_start, s_end, t_start, t_end = pair[0], pair[0], pair[1], pair[1]
        
      elif isinstance(pair[0], int) and isinstance(pair[1], list):
            s_start, s_end = pair[0], pair[0]
            t_start, t_end = min(pair[1]), max(pair[1])
      elif isinstance(pair[0], list) and isinstance(pair[1], int):
            s_start, s_end = min(pair[0]), max(pair[0])
            t_start, t_end = pair[1], pair[1]
      elif isinstance(pair[0], list) and isinstance(pair[1], list):
          s_start, s_end = min(pair[0]), max(pair[0])
          t_start, t_end = min(pair[1]), max(pair[1])
      # assert not self.conflict((s_start, s_end, t_start, t_end))
      self.alignments.append((s_start, s_end, t_start, t_end))
      
  
  def string_version(self):
      ans = ''
      for guy in self.alignments:
          assert len(guy) == 4
          ans += str(guy[0]) + '-' + str(guy[1]) + '-' + str(guy[2]) + '-' + str(guy[3]) + ' '
      return sort_by_first_number(ans)
          
        
      
    
def read_function_words(file):
        engans = set()
        with open(file, mode='r', encoding='utf-8') as reader:
            for a, line in enumerate(reader):
                if a > 3:
                    if len(line.split()) < 1:
                        break
                    word = line.split()[0].strip()
                    engans.add(word.replace('"', ''))
        engans.add("'s")
        return engans
    
ENGLISH_FUNCTION_WORDS = read_function_words(HERE / 'dependencies/WList.pm')
        
def get_lemma(word, lang):
    if not word.strip():
        return word
    if lang == 'zh':
        return word
    if lang in LEMMATIZERS:
        proper_nlp = LEMMATIZERS[lang]
    else:
        proper_nlp = LEMMATIZERS['xx']
    cachekey = word + '<>' + lang
    if cachekey in STEM_CACHE:
        return STEM_CACHE[cachekey]
    doc1 = proper_nlp(word)

    STEM_CACHE[cachekey] = doc1[0].lemma_
    return doc1[0].lemma_

def convert_quad_to_tuple(ours):
    ans = []
    for align in ours.split():
        sb, se, tb, te = align.split('-')
        sb, se, tb, te = int(sb), int(se), int(tb), int(te)
        for x in range(sb, se + 1):

            for y in range(tb, te + 1):
                ans.append((x, y))
    return ans

def erase_cache():
    SYNSET_CACHED_DICT.clear()
    ALIGNMENT_CACHE.clear()
    STEM_CACHE.clear()
    
def get_synsets_from_id_cachable(id):
    
    if id in SYNSET_CACHED_DICT:
        pass
    else:
        import babelnet as bn
        SYNSET_CACHED_DICT[id] = bn.get_synset(id)
    return SYNSET_CACHED_DICT[id]
    
def get_synsets_cachable(word, lang, pos=None):
    
    # from babelnet import Language
    # from babelnet.pos import POS
    # print("TRYNA GET SYNSETS FOR", word)
    if word in DO_NOT_CALL_BABELNET:
        return set()
    import babelnet as bn
    if word.startswith('bn:'):
        if word in SYNSET_CACHED_DICT:
            return SYNSET_CACHED_DICT[word]
        SYNSET_CACHED_DICT[word] = bn.get_synset(bn.BabelSynsetID(word))
        # await asyncio.sleep(0.2)
        # print("SYNSET IS", SYNSET_CACHED_DICT[word])
        return SYNSET_CACHED_DICT[word]
    global POS_TAGS
    global LANGS
    if pos is not None and pos in POS_TAGS:
        index = word + lang + pos
        if index in SYNSET_CACHED_DICT:
            return SYNSET_CACHED_DICT[index]
        real_pos = POS_TAGS[pos]
        SYNSET_CACHED_DICT[index] = {s.id for s in bn.get_synsets(word, from_langs=[LANGS[lang]], poses=[real_pos])}
        # await asyncio.sleep(0.2)
        return SYNSET_CACHED_DICT[index]
    else:
        index = word + lang + 'nopos'
        if index in SYNSET_CACHED_DICT:
            return SYNSET_CACHED_DICT[index]
        # print("WORD IS", word)
        SYNSET_CACHED_DICT[index] = {s.id for s in bn.get_synsets(word, from_langs=[LANGS[lang]])}
        # await asyncio.sleep(0.2)
        return SYNSET_CACHED_DICT[index]
    
    
'''def hypernyms_x_levels(token, language, level):
    # print(token)
    key = 'hypernyms' + str(level) + language + token
    if key in SYNSET_CACHED_DICT:
        return SYNSET_CACHED_DICT[key]
    lemma = get_lemma(token, language)
    synsets1 = get_synsets_cachable(token, language)
    
    synsets3 = get_synsets_cachable(lemma, language)
    
    
    synsets = synsets1.union(synsets3)
    
    synsets_changed = set([get_synsets_from_id_cachable(s) for s in synsets])
    curr_beam = synsets_changed.copy()
    all_concepts = synsets_changed.copy()
    for i in range(level):
        # print(curr_beam)
        new_beam = set()
        # print(i)
        for synset in curr_beam:
            by = synset
            for edge in by.outgoing_edges(BabelPointer.ANY_HYPERNYM):
                new_concept = get_synsets_from_id_cachable(edge.id_target)
                new_beam.add(new_concept)
                break
        
        curr_beam = new_beam
        all_concepts = all_concepts.union(curr_beam)
    SYNSET_CACHED_DICT[key] = all_concepts
    return all_concepts
        

def are_translations_by_dictionary(word1, word2, lang2, lang1="en"):
    # overrule and make sure it's all no poses
    
    if word1 in DO_NOT_CALL_BABELNET or word2 in DO_NOT_CALL_BABELNET:
        return False
    for element in DO_NOT_CALL_BABELNET:
        if element in word1 or element in word2:
            return False
        
        
    # synsets1 = get_synsets_cachable(word1, lang1)
    # synsets2 = get_synsets_cachable(word2, lang2)
    # synsets3 = get_synsets_cachable(lemma1, lang1)
    # synsets4 = get_synsets_cachable(lemma2, lang2)
    
    # src_synsets = synsets1.union(synsets3)
    # tgt_synsets = synsets2.union(synsets4)
    if word1 in ENGLISH_FUNCTION_WORDS:
        return False
    
    src_hypernyms = hypernyms_x_levels(word1, lang1, 1)
    tgt_hypernyms = hypernyms_x_levels(word2, lang2, 1)
    
    # print(src_hypernyms)
    intersect = src_hypernyms.intersection(tgt_hypernyms)
    # print(intersect)
    if intersect:
        return True
    return False
'''

def are_synonyms_by_bn(word1, word2, lang2, lang1="en"):
    # overrule and make sure it's all no poses
    # if 'com' in word1 and 'down' in word1:
        # print(word2 + " x " + word1)
    word1 = word1.replace(' ', '_')
    word2 = word2.replace(' ', '_')
    if word1 in DO_NOT_CALL_BABELNET or word2 in DO_NOT_CALL_BABELNET:
        # print("DECIDED NOT TO CALL BC OF ", word1, 'OR', word2)
        return False
    for element in DO_NOT_CALL_BABELNET:
        if element in word1 or element in word2:
            if element != '_':
                # print("DECIDED NOT TO CALL BC OF ", element, "BEING IN ONE OF", word1, word2)
                return False
    if word1 in ENGLISH_FUNCTION_WORDS and lang1 == 'en':
        lemma1 = word1
        lemma2 = word2
    elif '_' in word2 or '_' in word1:
        for guy in word1.split('_') + word2.split('_'):
            
                if guy in PUNCTUATION:
                    # print(guy, "WAS PUNCTUATION SO FALSE")
                    return False
                
       
        lemma1 = '_'.join([get_lemma(a, lang1) for a in word1.split('_')])
        lemma2 = '_'.join([get_lemma(a, lang2) for a in word2.split('_')])
    else:
        lemma1 = get_lemma(word1, lang1)
        lemma2 = get_lemma(word2, lang2)
    
    # print(lemma2, 'is lemma2', word2, 'is word2')
        
        
    # print(word1, 'is word1', word2, 'is word2')
    synsets1 = get_synsets_cachable(word1, lang1)
    synsets2 = get_synsets_cachable(word2, lang2)
    synsets3 = get_synsets_cachable(lemma1, lang1)
    synsets4 = get_synsets_cachable(lemma2, lang2)
    

    # print(synsets1, 'is the first')
    # print(synsets2, 'is the second')
    
    word_intersection = set.intersection(synsets1, synsets2)
    lem_intersection = set.intersection(synsets3, synsets4)
    # print(lem_intersection)
    # print(intersection, 'is the intersection')
    if word_intersection:
        return 'strict'
    elif lem_intersection:
        return 'loose'

    return False        
 

    
def simalign_cachable(source, target):
    key = source + '<and>' + target
    if key in ALIGNMENT_CACHE:
        pass
    else:
        ALIGNMENT_CACHE[key] = SIMALIGNER.get_word_aligns(source, target)
    return ALIGNMENT_CACHE[key]['itermax']

def token_pos_and_morph_tag(sentence, language):
    # print(word_list)
    # print(word_list)
    tokens = []
    poses = []
    morphs = []
    
    key = language + '<>' + sentence + '<>' + 'WITHMORPH'
    if key in POS_CACHE:
        return POS_CACHE[key]
    if language in LEMMATIZERS:
        doc = LEMMATIZERS[language](sentence)
    else:
        doc = LEMMATIZERS['xx'](sentence)
   
    for token in doc:
        tokens.append(token.text)
        poses.append(token.pos_)
        morphs.append(token.morph.to_dict())
        
    ans = (tokens, poses, morphs)
    # POS_CACHE[key] = ans
    POS_CACHE[key] = ans
    # print(ans, word_list)
    # assert len(ans) == len(word_list)
    return POS_CACHE[key]

def can_have_same_pos(sl, tl, word1, word2, p1, p2, alignobj):
    if alignobj.using_custom():
        return False
   
    synsets1 = get_synsets_cachable(word1, sl)
    synsets2 = get_synsets_cachable(word2, tl).union(get_synsets_cachable(get_lemma(word2, tl), tl)).union(TAGDICT[p2])
    
    poses1 = {TAGDICT[p1]}
    for synset in synsets1:
        poses1.add(str(synset)[-1])
    for synset in synsets2:
        letter = str(synset)[-1]
        if letter in poses1:
            return True
    
    return False
    
def pos_match(src_lang, tgt_lang, pos1, pos2, w1, w2, alignobj):
    
    if pos1 == pos2:
        return True
    if w1[:5] == w2[:5]:
        return True
    if can_have_same_pos(src_lang, tgt_lang, w1, w2, pos1, pos2, alignobj):
        return True
    if pos1 in ['VERB', 'AUX'] and pos2 in ['VERB', 'AUX']: # because this is basically the same
        return True
    if pos1 in ['PRON', 'DET'] and pos2 in ['PRON', 'DET']: # in the case of 'its' can be the same
        return True
    if pos1 in ['VERB', 'ADJ'] and pos2 in ['VERB', 'ADJ']: # in the case of 'we are surrounded' this can be the same
        return True
    if pos1 in ['PART', 'ADP', 'SCONJ'] and pos2 in ['PART', 'ADP', 'SCONJ']: # basically the same, no?
        return True
    if pos1 in ['PROPN', 'NOUN'] and pos2 in ['PROPN', 'NOUN']: # proper nouns are nouns i think this works:
        return True
    if pos1 in ['ADJ', 'DET'] and pos2 in ['ADJ', 'DET']: # in the context of 'other people'
        return True
    if pos1 in ['NOUN', 'VERB'] and pos2 in ['NOUN', 'VERB']: # really losing the plot of this function making any sense, but this is needed sometimes
        return True
    if alignobj.are_synonyms_by_dictionary(w1, w2, tgt_lang, src_lang):
        return True
    return False

def screen_alignments(a, s, t, len1, len2, slan, tlan, al_obj):
    
    answer = []
    t1, p1, _ = token_pos_and_morph_tag(s, slan)
    t2, p2, _ = token_pos_and_morph_tag(t, tlan)
    if not (len(t1) == len1 and len(t2) == len2):
        return a    
    for pair in a:
        if pos_match(slan, tlan, p1[pair[0]], p2[pair[1]], t1[pair[0]], t2[pair[1]], al_obj):
            answer.append(pair)
        
    return answer
    
def simalign_cachable_screened(source, target, len1, len2, src_lan, tar_lan, aligner):
    alignments = simalign_cachable(source, target)
    return screen_alignments(alignments, source, target, len1, len2, src_lan, tar_lan, aligner)

def add_nonvetoed_alignment(a, s, sentone, senttwo):
    if not veto(s[0], s[1], sentone, senttwo):
        a.add(s)

def add_alignment(a, s):
    a.add(s)
    
def mean(list_of_vals):
    return sum(list_of_vals) / len(list_of_vals)
    
def choose_diagonally(possible_inds, index, known_index_length, alt_index_length):
    if len(possible_inds) == 1:
        return index, possible_inds[0]
   
    # print(index)
    flat_possible_indices = []
    list_possibilities = []
    for item in possible_inds:
        if isinstance(item, int):
            flat_possible_indices.append(item)
        elif isinstance(item, list):
            list_possibilities.append(item)
            for guy in item:
                flat_possible_indices.append(guy)
    
    # print(index, possible_inds, known_index_length)
    if isinstance(index, list):
        position = mean(index)
    else:
        position = index
    src_distance_through = position / known_index_length
    tgt_distances_through = [a / alt_index_length for a in flat_possible_indices]
    
    
    best_closeness = 99999
    best_ans = None
    for i, tgt_dist in enumerate(tgt_distances_through):
        closeness = abs(tgt_dist - src_distance_through)
        if closeness < best_closeness:
            best_closeness = closeness
            best_ans = flat_possible_indices[i]
            
    for guy in list_possibilities:
        if best_ans in guy:
           
            return index, best_ans
    return index, best_ans
    
    
def mwe_intersectalign(index, simals, pis, iwk):
    # print(simals)
    # print(iwk)
    # print(index)
    # print(pis)
    
    good = True
    for pi in pis + [index]:
        if isinstance(pi, list):
            good = False
    if isinstance(index, int):
        s_side_inds_flat = [index]
    else:
        s_side_inds_flat = index
        
    if good:
        
        return [a for a in simals if a[iwk] == index and a[1 - iwk] in pis]
    else:
        ans = []
        for pi in pis:
            this_pi_valid = True
            if isinstance(pi, int):
                for guy in s_side_inds_flat:
                    if (guy, pi) in simals:
                        pass
                        # print("YEA PASSING ONE")
                    else:
                        this_pi_valid = False
                
            elif isinstance(pi, list):
                for x in pi:
                    for guy in s_side_inds_flat:
                        if (x, guy) in simals:
                            pass
                            # print("PASSING TWOOOO")
                        else:
                            this_pi_valid = False
            if this_pi_valid:
                    # print("this pi valid")
                    ans.append((index, pi))
                        
        return ans
     
        
    
def choose_alignment_from_bn_options(poss_indices, index, src_wds, tgt_wds, intersect_pass, choosing):
    
    # print(ans, poss_indices, index, src_wds, tgt_wds)
    # TODO maybe check if meanign is precisely the same instead of just possible synonym?
    simaligns = simalign_cachable(' '.join(src_wds), ' '.join(tgt_wds))
    
    if choosing == 'source':
        relevant_wds = tgt_wds
        ind_we_know = 1
        ind_we_dont = 0
        other_wds = src_wds
    elif choosing == 'target':
        relevant_wds = src_wds
        other_wds = tgt_wds
        ind_we_know = 0
        ind_we_dont = 1
    # print(src_wds)
    # print(tgt_wds)
    intersectaligns_of_token = mwe_intersectalign(index, simaligns, poss_indices, ind_we_know) # intersection of alignments
    
    # print(choosing, intersectaligns_of_token)
        
    intersectaligns_of_token = [a for a in simaligns if a[ind_we_know] == index and a[ind_we_dont] in poss_indices]
        
    if len(intersectaligns_of_token) == 1:
    
        # only one of the possibilities is in both the babelnet align and the simalign
        # print('intersect of one', relevant_wds[index])
    
       
        return intersectaligns_of_token[0]
    elif len(intersectaligns_of_token) > 1:
        # print('intersect of multiple', relevant_wds[index])
        # some of the alignments are supported by simalign, so let's choose between them by virtue of diagonality
        # print(intersectaligns_of_token, src_wds, tgt_wds, src_wds[index])
        
        confirmed_poss_indices = [a[ind_we_dont] for a in intersectaligns_of_token]
        
        most_diagonal_alignment = choose_diagonally(confirmed_poss_indices, index, len(relevant_wds), len(other_wds))

        # print(most_diagonal_alignment)
     
        return most_diagonal_alignment[ind_we_know], most_diagonal_alignment[ind_we_dont]
    elif intersect_pass:
        return None, None
    elif len(intersectaligns_of_token) == 0 and len(poss_indices) > 0:
        # print('intersect of zero', relevant_wds[index])
        # none of the alignments are supported by simalign, but let's choose one by virtue of diagonality
        most_diagonal_alignment = choose_diagonally(poss_indices, index, len(relevant_wds), len(other_wds))
      
        return most_diagonal_alignment[ind_we_know], most_diagonal_alignment[ind_we_dont]
    
    
    return None, None
    

def accept_all_alignments(proposed, alignments, usi, uti, toks1, toks2):
    # print(proposed, alignments.alignments)
    unfinished = False
    to_add = []
    for pair in proposed:
        if not alignments.conflict(pair):
            to_add.append(pair)
    for pair in to_add:
        add_nonvetoed_alignment(alignments, pair, toks1, toks2)
        safe_remove(usi, pair[0])
        safe_remove(uti, pair[1])
          
        
    # print(proposed, alignments)
    # print(alignments.alignments)
    return unfinished

def safe_remove(s, element):
    if isinstance(element, int):
        
        if element in s:
            s.remove(element)
    elif isinstance(element, list):
      
        for guy in element:
        
            s.remove(guy)
        
       
    
def flatten(mixed_list):
    result = []
    for item in mixed_list:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result

def alignflatten(prop):
    ans = []
    # print(prop)
    for pair in prop:
        if isinstance(pair, tuple) and isinstance(pair[0], int) and isinstance(pair[1], int):
            ans.append(pair)
        elif isinstance(pair, tuple) and isinstance(pair[0], list) and isinstance(pair[1], int):
            for guy in pair[0]:
                ans.append((guy, pair[1]))
        elif isinstance(pair, tuple) and isinstance(pair[0], int) and isinstance(pair[1], list):
            for guy in pair[1]:
                ans.append((pair[0], guy))
        else:
            # print(pair)
            # print(prop)
            assert False
        
        
    return ans
    
def get_claimed(mylist):
    ans_s, ans_t = [], []
    # print(mylist)
    for guy in mylist:
        # print(guy)
        assert isinstance(guy, tuple) and len(guy) == 2
        if isinstance(guy[0], int) and isinstance(guy[1], int):
            ans_s.append(guy[0])
            ans_t.append(guy[1])
        else:
            if isinstance(guy[0], list):
                for fella in guy[0]:
                    ans_s.append(fella)
            if isinstance(guy[1], list):
                for fella in guy[1]:
                    ans_t.append(fella) 
            if isinstance(guy[0], int):
                ans_s.append(guy[0])
            if isinstance(guy[1], int):
                ans_t.append(guy[1])
    return ans_s, ans_t
  
def claimed_by(arg):
    claimed = []
    if isinstance(arg, int):
        claimed.append(arg)
    elif isinstance(arg, list):
        for guy in arg:
            claimed.append(guy)  
    return claimed
   
def claims(p, s, t):
    # print(p, s, t)
    if isinstance(p, tuple) and len(p) == 2:
        if isinstance(p[0], int) and isinstance(p[1], int):
            # print("GONNA CHECK IF", p[0], p[1], "ARE IN", s, t, "RESPECTIVELY")
            return p[0] in s or p[1] in t
        else:
            claimed_by_first = claimed_by(p[0])
            for guy in claimed_by_first:
                if guy in s:
                    return True
            claimed_by_second = claimed_by(p[1])
            for guy in claimed_by_second:
                if guy in t:
                    return True
            return False
    else:  
        assert False 
    
def subsumes(other_match, match):
    
    if isinstance(other_match[0], int):
        o_claim_s = [other_match[0]]
    else:
        o_claim_s = other_match[0]
        
    if isinstance(other_match[1], int):
        o_claim_t = [other_match[1]]
    else:
        o_claim_t = other_match[1]
        
    if isinstance(match[1], int):
        claim_t = [match[1]]
    else:
        claim_t = match[1]
    
    if isinstance(match[0], int):
        claim_s = [match[0]]
    else:
        claim_s = match[0]
    for guy in claim_s:
        if guy not in o_claim_s:
            # print(other_match, 'doesnt subsume', match)
            # a = input("")
            return False
    for guy in claim_t:
        if guy not in o_claim_t:
            # print(other_match, 'doesnt subsume', match)
            # a = input("")
            return False
    
    return True
    
    
def generalize_if_possible(proposed):
    # print(proposed)
    for match in proposed:
        for other_match in proposed:
            if other_match != match and subsumes(other_match, match):
                proposed.remove(match)
                break
    return proposed
    
    
def accept_unconflicting_alignments(proposed, alignments, usi, uti, toks1, toks2):
    # print(proposed, alignments)
    # print(usi)
    # print(uti)
    proposed = generalize_if_possible(proposed)
    # print("PROPOSED", proposed)
    
    unfinished = False
    claimed_src_indices, claimed_tar_indices = get_claimed(proposed)
    
    # print(claimed_tar_indices)
    # print(claimed_src_indices)
    conflicted_tgt_indices = [i for i in set(claimed_tar_indices) if claimed_tar_indices.count(i) > 1]
    conflicted_src_indices = [i for i in set(claimed_src_indices) if claimed_src_indices.count(i) > 1]
   
    for pair in proposed:
       
        if not claims(pair, conflicted_src_indices, conflicted_tgt_indices):
            add_nonvetoed_alignment(alignments, pair, toks1, toks2)
            
            safe_remove(usi, pair[0])
            safe_remove(uti, pair[1])
        else:
            # print(pair, 'claims', conflicted_src_indices, conflicted_tgt_indices)
            # print(care_abt)
            
            unfinished = True
    # print(proposed, alignments)
    # a = input("Check this")
    return unfinished
    
def consecutive_subsequences(nums):
   
    if not nums:
        return []

    nums = sorted(set(nums))  # Sort and remove duplicates
    subsequences = []
    start = 0

    for i in range(1, len(nums) + 1):
        # If end of list or break in adjacency
        if i == len(nums) or nums[i] != nums[i - 1] + 1:
            segment = nums[start:i]
            # Generate all subsequences of length >= 2 from the segment
            for j in range(len(segment)):
                for k in range(j + 2, len(segment) + 1):
                    subsequences.append(segment[j:k])
            start = i

    return subsequences


    
def intersection_pass(src_lang, tar_lang, src_wds, tgt_wds, align_ans, unaligned_source_indices, unaligned_target_indices, strict_lemma, aligner):
    babelnet_pass(src_lang, tar_lang, src_wds, tgt_wds, align_ans, unaligned_source_indices, unaligned_target_indices, strict_lemma, aligner, True)
    
    
def lemmatize(list_of_words, language):
    return [get_lemma(a, language) for a in list_of_words]
    
    
def babelmwe_pass(sl, tl, src_wds, tgt_wds, align_ans, unaligned_source_indices, unaligned_target_indices, strict_lemma, alignobj, strict_intersect=False, use_lemmas=False):
  unfinished = True
  if use_lemmas:
      src_wds = lemmatize(src_wds, sl)
      tgt_wds = lemmatize(tgt_wds, tl)
  while unfinished:
    # first pass using babelnet align
    # Forward pass : align the source words
    proposed_alignments = []
    for i in unaligned_source_indices + consecutive_subsequences(unaligned_source_indices):
        if isinstance(i, int):
            word_one = src_wds[i]
        elif isinstance(i, list):
            word_one = '_'.join([src_wds[el] for el in i])
        # if i == [2, 3]:
        #     input("X")
        
        possible_alignment_indices_for_this = []
        less_strict_possible_alignment_indices_for_this = []
        for j in unaligned_target_indices + consecutive_subsequences(unaligned_target_indices):
            if isinstance(i, int) and isinstance(j, int):
                continue
            if isinstance(j, int):
                
                word_two = tgt_wds[j]
            elif isinstance(j, list):
                word_two = '_'.join([tgt_wds[el] for el in j])
            # print(word_two)
            if alignobj.are_synonyms_by_dictionary(word_one, word_two, tl, sl) == 'strict': # this is true if theyre the same without lemmatizing them
             
                possible_alignment_indices_for_this.append(j)
                if isinstance(j, int):
                    assert j in unaligned_target_indices
                elif isinstance(j, list):
                    for guy in j:
                        assert guy in unaligned_target_indices
            elif alignobj.are_synonyms_by_dictionary(word_one, word_two, tl, sl) == 'loose': # if they're the same only when lemmatized
                less_strict_possible_alignment_indices_for_this.append(j)
           
        
            
        for x in possible_alignment_indices_for_this:
            if isinstance(x, int):
            
                assert x in unaligned_target_indices
            else:
                for sub_x in x:
                    assert sub_x in unaligned_target_indices
        
        # we prefer to use the strict ones if possible, but if there aren't any, use either
        if len(possible_alignment_indices_for_this) == 0 and not strict_lemma:
            possible_alignment_indices_for_this += less_strict_possible_alignment_indices_for_this
        
            
        # print(word_one, possible_alignment_indices_for_this, [tgt_wds[a] for a in possible_alignment_indices_for_this])
        new_al_src_ind, new_al_tgt_ind = choose_alignment_from_bn_options(possible_alignment_indices_for_this, i, src_wds, tgt_wds, strict_intersect, 'target')   
        
        # if we made a new alignment, remove these from the list of indices that can be aligned still
        if new_al_src_ind is not None and new_al_tgt_ind is not None:
            # print(new_al_src_ind, new_al_tgt_ind)
            # print('aligning', src_wds[new_al_src_ind], tgt_wds[new_al_tgt_ind])
            
                
            proposed_alignments.append((new_al_src_ind, new_al_tgt_ind))
             
    unfinished = accept_unconflicting_alignments(proposed_alignments, align_ans, unaligned_source_indices, unaligned_target_indices, src_wds, tgt_wds)
    proposed_alignments = []
    
    # backwards pass: align target words   
    
    for j in unaligned_target_indices + consecutive_subsequences(unaligned_target_indices):
        
        if isinstance(j, int):
                
                word_two = tgt_wds[j]
        elif isinstance(j, list):
                word_two = '_'.join([tgt_wds[el] for el in j])
   
        possible_alignment_indices_for_this = []
        less_strict_possible_alignment_indices_for_this = []
        for i in unaligned_source_indices + consecutive_subsequences(unaligned_source_indices):
            if isinstance(i, int) and isinstance(j, int):
                continue
            if isinstance(i, int):
                word_one = src_wds[i]
            elif isinstance(i, list):
                word_one = '_'.join([src_wds[el] for el in i])
            # print(word_two)
            if alignobj.are_synonyms_by_dictionary(word_one, word_two, tl, sl) == 'strict': # this is true if theyre the same without lemmatizing them
                possible_alignment_indices_for_this.append(i)
                # print(word_one, word_two)
                if isinstance(i, int):
                    assert i in unaligned_source_indices
                elif isinstance(i, list):
                    for guy in i:
                        assert guy in unaligned_source_indices
            elif alignobj.are_synonyms_by_dictionary(word_one, word_two, tl, sl) == 'loose': # if they're the same only when lemmatized
                less_strict_possible_alignment_indices_for_this.append(i)
        for x in possible_alignment_indices_for_this:
            if isinstance(x, int):
                assert x in unaligned_source_indices
            elif isinstance(x, list):
                for guy in x:
                    assert guy in unaligned_source_indices
        
        # we prefer to use the strict ones if possible, but if there aren't any, use either
        if len(possible_alignment_indices_for_this) == 0 and not strict_lemma:
            possible_alignment_indices_for_this += less_strict_possible_alignment_indices_for_this
        
            
        # print(word_one, possible_alignment_indices_for_this, [tgt_wds[a] for a in possible_alignment_indices_for_this])
        new_al_src_ind, new_al_tgt_ind = choose_alignment_from_bn_options(possible_alignment_indices_for_this, j, src_wds, tgt_wds, strict_intersect, 'source')   
       
        # if we made a new alignment, remove these from the list of indices that can be aligned still
        if new_al_src_ind is not None and new_al_tgt_ind is not None:
         
            assert new_al_src_ind in unaligned_source_indices or (isinstance(new_al_src_ind, list) and [d in unaligned_source_indices for d in unaligned_source_indices])
            proposed_alignments.append((new_al_src_ind, new_al_tgt_ind))
            # print(unaligned_source_indices, unaligned_target_indices)
    unfinished = accept_unconflicting_alignments(proposed_alignments, align_ans, unaligned_source_indices, unaligned_target_indices, src_wds, tgt_wds)
    previous_cycle = proposed_alignments.copy()
    if proposed_alignments == previous_cycle:
        unfinished = False
                    
def babelnet_pass(sl, tl, src_wds, tgt_wds, align_ans, unaligned_source_indices, unaligned_target_indices, strict_lemma, alignobj, strict_intersect=False):
  unfinished = True
  while unfinished:
    # first pass using babelnet align
    # Forward pass : align the source words
    proposed_alignments = []
    for i in unaligned_source_indices.copy():
        if isinstance(i, int):
            word_one = src_wds[i]
        elif isinstance(i, list):
            word_one = '_'.join([src_wds[el] for el in i])
     
        possible_alignment_indices_for_this = []
        less_strict_possible_alignment_indices_for_this = []
        for j in unaligned_target_indices:
            if isinstance(j, int):
                
                word_two = tgt_wds[j]
            
            # print(word_two)
            if alignobj.are_synonyms_by_dictionary(word_one, word_two, tl, sl) == 'strict': # this is true if theyre the same without lemmatizing them
                possible_alignment_indices_for_this.append(j)
                if isinstance(j, int):
                    assert j in unaligned_target_indices
                elif isinstance(j, list):
                    for guy in j:
                        assert guy in unaligned_target_indices
            elif alignobj.are_synonyms_by_dictionary(word_one, word_two, tl, sl) == 'loose': # if they're the same only when lemmatized
                less_strict_possible_alignment_indices_for_this.append(j)
        
            
        for x in possible_alignment_indices_for_this:
            if isinstance(x, int):
            
                assert x in unaligned_target_indices
            else:
                for sub_x in x:
                    assert sub_x in unaligned_target_indices
        
        # we prefer to use the strict ones if possible, but if there aren't any, use either
        if len(possible_alignment_indices_for_this) == 0 and not strict_lemma:
            possible_alignment_indices_for_this += less_strict_possible_alignment_indices_for_this
        
            
        # print(word_one, possible_alignment_indices_for_this, [tgt_wds[a] for a in possible_alignment_indices_for_this])
        new_al_src_ind, new_al_tgt_ind = choose_alignment_from_bn_options(possible_alignment_indices_for_this, i, src_wds, tgt_wds, strict_intersect, 'target')   
        
        # if we made a new alignment, remove these from the list of indices that can be aligned still
        if new_al_src_ind is not None and new_al_tgt_ind is not None:
            # print(new_al_src_ind, new_al_tgt_ind)
            # print('aligning', src_wds[new_al_src_ind], tgt_wds[new_al_tgt_ind])
            if isinstance(new_al_tgt_ind, int):
                assert new_al_tgt_ind in unaligned_target_indices
                proposed_alignments.append((new_al_src_ind, new_al_tgt_ind))

            elif isinstance(new_al_tgt_ind, list):
                for k in new_al_tgt_ind:
                    proposed_alignments.append((new_al_src_ind, k))
    unfinished = accept_unconflicting_alignments(proposed_alignments, align_ans, unaligned_source_indices, unaligned_target_indices, src_wds, tgt_wds)
    proposed_alignments = []
    
    # backwards pass: align target words   
    
    for j in unaligned_target_indices.copy():
        
        if isinstance(j, int):
                
                word_two = tgt_wds[j]
        elif isinstance(j, list):
                word_two = '_'.join([tgt_wds[el] for el in j])
   
        possible_alignment_indices_for_this = []
        less_strict_possible_alignment_indices_for_this = []
        for i in unaligned_source_indices:
            if isinstance(i, int):
                word_one = src_wds[i]
            
            # print(word_two)
            if alignobj.are_synonyms_by_dictionary(word_one, word_two, tl, sl) == 'strict': # this is true if theyre the same without lemmatizing them
                possible_alignment_indices_for_this.append(i)
                # print(word_one, word_two)
                if isinstance(i, int):
                    assert i in unaligned_source_indices
                
            elif alignobj.are_synonyms_by_dictionary(word_one, word_two, tl, sl) == 'loose': # if they're the same only when lemmatized
                less_strict_possible_alignment_indices_for_this.append(i)
        for x in possible_alignment_indices_for_this:
            if isinstance(x, int):
                assert x in unaligned_source_indices
            elif isinstance(x, list):
                for guy in x:
                    assert guy in unaligned_source_indices
        
        # we prefer to use the strict ones if possible, but if there aren't any, use either
        if len(possible_alignment_indices_for_this) == 0 and not strict_lemma:
            possible_alignment_indices_for_this += less_strict_possible_alignment_indices_for_this
        
            
        # print(word_one, possible_alignment_indices_for_this, [tgt_wds[a] for a in possible_alignment_indices_for_this])
        new_al_src_ind, new_al_tgt_ind = choose_alignment_from_bn_options(possible_alignment_indices_for_this, j, src_wds, tgt_wds, strict_intersect, 'source')   
       
        # if we made a new alignment, remove these from the list of indices that can be aligned still
        if new_al_src_ind is not None and new_al_tgt_ind is not None:
            assert new_al_src_ind in unaligned_source_indices
            proposed_alignments.append((new_al_src_ind, new_al_tgt_ind))
            # print(unaligned_source_indices, unaligned_target_indices)
    unfinished = accept_unconflicting_alignments(proposed_alignments, align_ans, unaligned_source_indices, unaligned_target_indices, src_wds, tgt_wds)
    previous_cycle = proposed_alignments.copy()
    if proposed_alignments == previous_cycle:
        unfinished = False
  
      
def are_aligned(ind_one, ind_two, alignments):
    for al in alignments:
        if al[0] == ind_one and al[1] == ind_two:
            return True
    return False
    
     
def get_adjacent_value_sequences(lst):
    s = set(lst)
    visited = set()
    sequences = []

    for num in s:
        if num in visited:
            continue
        # Grow the sequence in both directions
        seq = [num]
        visited.add(num)

        # Expand downwards
        down = num - 1
        while down in s:
            seq.append(down)
            visited.add(down)
            down -= 1

        # Expand upwards
        up = num + 1
        while up in s:
            seq.append(up)
            visited.add(up)
            up += 1

        sequences.append(sorted(seq))
    # print(sorted(sequences, key=lambda x: x[0]))
    ans_almost = sorted(sequences, key=lambda x: x[0])
    ans = [guy for guy in ans_almost if len(guy) > 1]
    return ans



        
def veto(i, j, src_ws, tgt_ws):
    if isinstance(i, int):
        w1 = src_ws[i]
    else:
        w1 = ' '.join([src_ws[k] for k in i])
    # print(tgt_ws)
    if isinstance(j, int):
        w2 = tgt_ws[j]
    else:
        w2 = ' '.join([tgt_ws[k] for k in j])
    
    # print(w1, w2)
    # a = input("We good?")
    if w1 in PUNCTUATION and w2 not in PUNCTUATION:
        return True
    if w2 in PUNCTUATION and w1 not in PUNCTUATION:
        return True
    if w1 in ['(', ')'] and w2 not in ['(', ')']:
        return True
    if w2 in ['(', ')'] and w1 not in ['(', ')']:
        return True
    return False
         
def simalign_pass(sl, tl, src_wds, tgt_wds, align_ans, unaligned_source_indices, unaligned_target_indices, a):
  unfinished = True
  previous_cycle = None
  while unfinished:
    # first pass using babelnet align
    proposed_alignments = []
    simaligns = simalign_cachable_screened(' '.join(src_wds), ' '.join(tgt_wds), len(src_wds), len(tgt_wds), sl, tl, a)
    # print('SIMALIGNS:', simaligns)
    # print(src_wds, tgt_wds)
    for i in unaligned_source_indices.copy():
        possible_alignment_indices_for_this = []
        for j in unaligned_target_indices:

            if are_aligned(i, j, simaligns) and not veto(i, j, src_wds, tgt_wds):
                possible_alignment_indices_for_this.append(j)
                assert j in unaligned_target_indices
        for x in possible_alignment_indices_for_this:
            assert x in unaligned_target_indices
            

        most_diagonal = choose_diagonally(possible_alignment_indices_for_this, i, len(src_wds), len(tgt_wds))
        
        
        new_al_src_ind, new_al_tgt_ind = most_diagonal[0], most_diagonal[1]
        
        # if we made a new alignment, remove these from the list of indices that can be aligned still
        if new_al_src_ind is not None and new_al_tgt_ind is not None:
            
            assert new_al_tgt_ind in unaligned_target_indices
            # print(unaligned_source_indices, unaligned_target_indices, 'removing', new_al_src_ind, new_al_tgt_ind)
            proposed_alignments.append(most_diagonal)
            # print('aligning', src_wds[new_al_src_ind], tgt_wds[new_al_tgt_ind])
            # print(unaligned_source_indices, unaligned_target_indices)
    
    
    unfinished = accept_unconflicting_alignments(proposed_alignments, align_ans, unaligned_source_indices, unaligned_target_indices, src_wds, tgt_wds)
    previous_cycle = proposed_alignments.copy()
    if proposed_alignments == previous_cycle:
        # print("TIME TO BREAK")
        accept_all_alignments(proposed_alignments, align_ans, unaligned_source_indices, unaligned_target_indices, src_wds, tgt_wds)
        return 
   
    
def simalign(source, target, steps=3):
    # print(source)
    # print(target)
    aligns = AlignlistObj([], source.split(), target.split())
    source_words = source.split()
    target_words = target.split()
    
    alignments = SIMALIGNER.get_word_aligns(source, target)['itermax']
    for guy in alignments:
        aligns.add(guy)
    
    answer = aligns.string_version()
    # print(stringify_aligns(answer, source, target))
    
    return answer
    
def string_subtract(a, b):
    assert a.startswith(b), f"String '{a}' does not start with '{b}'"
    return a[len(b):]
    

    
def read_this_address(filepath):
    ans = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for row in file:
            source, target, gold_aligns = row.strip().split('\t') 
            ans.append({'src': source, 'tgt': target, 'gold': gold_aligns})
    return ans

def save_to_tsv(to_be_saved, filepath):
    all_rows = ['\t'.join(['source', 'target', 'output', 'legible output', 'gold', 'legible gold']) + '\n']
    for tbs in to_be_saved:
        all_rows.append('\t'.join([tbs['src'], tbs['tgt'], tbs['output'], tbs['read_output'], tbs['gold'], tbs['read_gold']])+ '\n')
    # print(all_rows)
    with open(filepath, 'w', encoding='utf-8') as file:
        file.writelines(all_rows)












def align_and_validate(aligner_to_use, input_file, output_file, num_steps=3):
    info = read_this_address(input_file)
    # print(num_steps, 'steps')
    for m, row in enumerate(info):
    # print(row)
    
        # print(f"{m}/{len(info)}")
        # babel_aligns = simalign(row['src'], row['tgt'], num_steps)
        babel_aligns = aligner_to_use.new_align(row['src'], row['tgt'], num_steps)
        babel_aligns_legible = stringify_aligns(babel_aligns, row['src'], row['tgt'])
       
     
        gold_aligns_legible = stringify_aligns(row['gold'], row['src'], row['tgt'])
        
        # assert babel_aligns
        row['output'] = babel_aligns
        row['read_output'] = babel_aligns_legible
        row['read_gold'] = gold_aligns_legible
       
        # save_caches()
    

    save_to_tsv(info, output_file)
    evaluate(input_file, output_file)
    print()

    
