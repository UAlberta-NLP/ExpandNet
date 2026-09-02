import pandas as pd
import xml.etree.ElementTree as ET

def process_xml(xml_file_path):
  rows = []
  tree = ET.parse(xml_file_path)
  root = tree.getroot()
  
  for text in root.findall('text'):
    for sentence in text.findall('sentence'):
      sentence_id = sentence.get('id', '')
      
      for elem in sentence:
        elem_type = elem.tag
        lemma = elem.get('lemma', '')
        pos = elem.get('pos', '')
        text_content = elem.text.strip() if elem.text else ''
        elem_id = elem.get('id', '')
        
        # Extract token ID parts only if present
        # doc_id = sent_id = tok_id = ''
        # if elem_id:
        #   parts = elem_id.split('.')
        #   doc_id = parts[0] if len(parts) > 0 else ''
        #   sent_id = parts[1] if len(parts) > 1 else ''
        #   tok_id = parts[2] if len(parts) > 2 else ''
        
        rows.append({
          'type': elem_type,
          'lemma': lemma,
          'pos': pos,
          'text': text_content,
          'id': elem_id,
          # 'doc_id': doc_id,
          # 'sent_id': sent_id,
          # 'tok_id': tok_id,
          'sentence_id': sentence_id,
        })
  
  return(pd.DataFrame(rows))


def process_gold(file_name):
  rows = []
  
  with open(file_name, 'r') as f:
    for line in f:
      fields = line.strip().split(' ')
      
      if len(fields) < 2: # Should be at least two cols: instance ID and bn synset ID.
        continue
      
      # Add this row.
      rows.append({
        'id': fields[0],
        'bn_gold': fields[1], # Could be multiple golds -- take the first.
      })
 
  # Create a Pandas dataframe from the rows
  df = pd.DataFrame(rows)
  return(df)


def process_dataset(xml_file, gold_file):
  # Load the XML data into a dataframe
  df_xml = process_xml(xml_file)
  #return(df_xml)
  
  # Load the TSV data into a dataframe
  df_gold = process_gold(gold_file)
  
  # Merge the two dataframes on document_id, sentence_id, and token_id
  df_combined = pd.merge(df_xml, df_gold, on='id', how='left')
  
  return(df_combined)


def extract_sentences(df):
  # Group by document_id and sentence_id, then join the raw_token values for each group
  sentence_df = df.groupby(['sentence_id'])['text'].apply(' '.join).reset_index()
  lemma_df = df.groupby(['sentence_id'])['lemma'].apply(' '.join).reset_index()
  sentence_df = sentence_df.merge(lemma_df, on='sentence_id')
  return(sentence_df)


def merge_translations(df1, df2):
  return( pd.merge(df1, df2, on=['document_id', 'sentence_id'], how='inner') )


# Example usage:
# df = pd.read_csv("your_dataframe.csv")  # Your original dataframe
# fixed_df = fix_sentence_ids(df, "fixfile.tsv")
# fixed_df.to_csv("fixed_dataframe.csv", index=False)
def fix_sentence_ids(df, remap_file):
    # Step 1: Load the remap file into a dictionary
    remap_df = pd.read_csv(remap_file, sep='\t', header=None, names=['universal', 'source', 'target'])

    # Create a dictionary to map wrong sentence IDs to correct ones
    remap_dict = dict(zip(remap_df['source'], remap_df['universal']))


    # remap_dict_src = {}
    # remap_dict_tgt = {}
    # with open(remap_file, 'r') as fh:
    #     for l in fh.readlines():
    #         [universal, source, target] = l.strip().split('\t')
    #         source    = source.split(';')
    #         target    = target.split(';')
            
   
    # Step 2: Create a new column to hold the updated sentence IDs
    def update_sentence_id(row):
        # Combine document_id and sentence_id to check the mapping
        doc_sent_id = f"{row['document_id']}.{row['sentence_id']}"
        
        # If the document_sent_id exists in the remap_dict, update sentence_id
        if doc_sent_id in remap_dict:
            # Get the correct sentence ID from the remap_dict
            return remap_dict[doc_sent_id].split('.')[1]  # Only extract the sentence part
        else:
            # Return None if there is no mapping (will drop row later)
            return None
    
    # Step 3: Apply the update function to the dataframe
    df['new_sentence_id'] = df.apply(update_sentence_id, axis=1)
    
    # Step 4: Drop rows where sentence_id could not be updated
    df = df.dropna(subset=['new_sentence_id']).copy()  # Remove rows with None in new_sentence_id column
    
    # Step 5: Update the sentence_id column with the new sentence IDs
    df['sentence_id'] = df['new_sentence_id']
    
    # Drop the auxiliary column
    df.drop(columns=['new_sentence_id'], inplace=True)
    
    return df




if __name__ == '__main__':
  src_data = '/home/bmhauer/data/se15/SemEval-2015-task-13-v1.0/data/semeval-2015-task-13-es.xml'
  src_gold = '/home/bmhauer/data/se15/SemEval-2015-task-13-v1.0/keys/gold_keys/ES/semeval-2015-task-13-es.key'

  tgt_data = '/home/bmhauer/data/se15/SemEval-2015-task-13-v1.0/data/semeval-2015-task-13-en.xml'
  tgt_gold = '/home/bmhauer/data/se15/SemEval-2015-task-13-v1.0/keys/gold_keys/EN/semeval-2015-task-13-en.key'

  df_src = process_dataset(src_data, src_gold)
  print(df_src.head(20))
  print()
  df_tgt = process_dataset(tgt_data, tgt_gold)
  print(df_tgt.head(20))
  print()

  df_sent_src = extract_sentences(df_src)
  print(df_sent_src.head(10))
  print()
  df_sent_tgt = extract_sentences(df_tgt)
  print(df_sent_tgt.head(10))
  print()

  df_sent_trans = merge_translations(df_sent_src, df_sent_tgt)
  print(df_sent_trans.head(10))

