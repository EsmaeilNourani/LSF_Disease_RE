import os
import pandas as pd
from collections import defaultdict
from Bio import Entrez
import argparse

def count_relations_and_entities(corpus_folder):
    # Define subfolders
    subfolders = ['train-set', 'dev-set', 'test-set']
    
    results = defaultdict(lambda: {
        'folder': '',
        'count_lifestyle_factor': 0,
        'unique_lifestyle_factor': set(),
        'count_disease': 0,
        'unique_disease': set(),
        'relations': 0,
        'abstract_text': ''
    })
    
    # Iterate through each subfolder
    for subfolder in subfolders:
        folder_path = os.path.join(corpus_folder, subfolder)
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist.")
            continue
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ann'):
                pmid = file_name.replace('.ann', '')
                ann_file_path = os.path.join(folder_path, file_name)
                txt_file_path = ann_file_path.replace('.ann', '.txt')
                
                # Count the number of relations in the .ann file
                with open(ann_file_path, 'r', encoding='utf-8') as ann_file:
                    ann_lines = ann_file.readlines()
                    relations_count = sum(1 for line in ann_lines if line.startswith('R') and 'Out-of-scope' not in line)
                    results[pmid]['relations'] = relations_count
                    results[pmid]['folder'] = subfolder
                
                # Read abstract text
                if os.path.exists(txt_file_path):
                    with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
                        results[pmid]['abstract_text'] = txt_file.read().replace('\n', ' ')
                
                # Process entities
                with open(ann_file_path, 'r', encoding='utf-8') as ann_file:
                    for line in ann_file:
                        parts = line.strip().split('\t')
                        if len(parts) > 1:
                            entity_details = parts[1].split(maxsplit=1)
                            entity_type = entity_details[0]
                            entity_text = parts[-1]
                            if entity_type == 'lifestyle_factor':
                                results[pmid]['count_lifestyle_factor'] += 1
                                results[pmid]['unique_lifestyle_factor'].add(entity_text)
                            elif entity_type == 'disease':
                                results[pmid]['count_disease'] += 1
                                results[pmid]['unique_disease'].add(entity_text)
    
    return results

def fetch_pubmed_info(pmid):
    
    Entrez.email = 'your_email@example.com'
    Entrez.api_key = 'your_api_key_here'  # Personal API key


    try:
        handle = Entrez.esummary(db="pubmed", id=pmid, retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        pub_info = record[0]
        return pub_info.get('FullJournalName', None), pub_info.get('PubDate', '')[:4], pub_info.get('Title', None)
    except Exception as e:
        print(f"Error fetching PubMed info for PMID {pmid}: {e}")
        return None, None, None

def create_metadata_with_progress(results, output_file):
    rows = []
    total_pmids = len(results)
    processed_pmids = 0

    for pmid, data in results.items():
        journal, year, title = fetch_pubmed_info(pmid)
        
        rows.append({
            'Publication_ID': pmid,
            'Article_Title': title,
            'Journal_Name': journal,
            'Publication_Year': year,
            'Data_Set': data['folder'],
            'Lifestyle_Factor_Count': data['count_lifestyle_factor'],
            'Disease_Count': data['count_disease'],
            'Relations_Count': data['relations'],
            'Abstract_Text': data['abstract_text'],
            'Unique_Lifestyle_Factors': ', '.join(sorted(data['unique_lifestyle_factor'])),
            'Unique_Diseases': ', '.join(sorted(data['unique_disease']))
        })
        
        # Increment processed PMIDs and show progress
        processed_pmids += 1
        progress = (processed_pmids / total_pmids) * 100
        print(f"Fetched PubMed info for {processed_pmids}/{total_pmids} PMIDs. Progress: {progress:.2f}%")
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Final output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate consolidated data from corpus.')
    parser.add_argument('corpus_folder', help='Path to the corpus folder containing train, dev, and test subfolders.')
    parser.add_argument('output_file_name', help='Name of the output TSV file.')
    
    args = parser.parse_args()
    
    # Determine the output file path to be in the same directory as corpus_folder
    output_path = os.path.join(args.corpus_folder, args.output_file_name)
    
    # Run the process
    results = count_relations_and_entities(args.corpus_folder)
    create_metadata_with_progress(results, output_path)
