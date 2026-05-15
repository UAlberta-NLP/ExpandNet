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
        
        rows.append({
          'type': elem_type,
          'lemma': lemma,
          'pos': pos,
          'text': text_content,
          'id': elem_id,
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
 
  
  # Load the TSV data into a dataframe
  df_gold = process_gold(gold_file)
  
  # Merge the two dataframes on document_id, sentence_id, and token_id
  df_combined = pd.merge(df_xml, df_gold, on='id', how='left')
  
  return(df_combined)


def extract_sentences(df):
    
    df['tokens'] = df['text'].str.replace(' ', '_')
    

    sentence_df = df.groupby('sentence_id').agg(
        text=('text', ' '.join),           # Creates a column named 'text'
        tokens=('tokens', ' '.join), # Creates a column named 'tokens'
        lemma=('lemma', ' '.join)          # Creates a column named 'lemma'
    ).reset_index()
    
    return sentence_df


def merge_translations(df1, df2):
  return( pd.merge(df1, df2, on=['document_id', 'sentence_id'], how='inner') )


def fix_sentence_ids(df, remap_file):
    # Step 1: Load the remap file into a dictionary
    remap_df = pd.read_csv(remap_file, sep='\t', header=None, names=['universal', 'source', 'target'])

    # Create a dictionary to map wrong sentence IDs to correct ones
    remap_dict = dict(zip(remap_df['source'], remap_df['universal']))
            
   
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
