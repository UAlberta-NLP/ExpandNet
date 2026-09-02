UWN 2010-10
Gerard de Melo
http://www.mpi-inf.mpg.de/yago-naga/uwn/


== DESCRIPTION ==

UWN is an automatically constructed multilingual lexical knowledge base
based on the structure of Princeton WordNet. Please see the web site above
for more information.

UWN is best used in conjunction with the original WordNet created at
at Princeton University. 
http://wordnet.princeton.edu


== DATA FORMAT ==

The gzip-compressed TSV file is best decompressed on the fly
while reading for best performance. Each line contains subject, predicate,
object, and weight, separated by tabs.
Words and other terms are listed as "t/<iso-639-3-language-code>/<string>", e.g.
"t/eng/house" for the English term "house".
Predicates include "rel:means" for the relationship between a term and its meanings.
WordNet senses are given as "s/<wordnet-pos-tag><wordnet-3.0-synset-offset>"
(e.g. "s/n1740" for WordNet 3.0's entity synset).


== CREDITS AND LICENSE ==

Gerard de Melo
http://icsi.berkeley.edy/~demelo/

For academic use, please cite:
Gerard de Melo and Gerhard Weikum (2009). Towards a Universal Wordnet by Learning from Combined Evidence.
In: Proc. 18th ACM Conference on Information and Knowledge Management (CIKM 2009),
Hong Kong, China. ACM, New York, USA.

License: CC BY-NC-SA 3.0
http://creativecommons.org/licenses/by-nc-sa/3.0/
