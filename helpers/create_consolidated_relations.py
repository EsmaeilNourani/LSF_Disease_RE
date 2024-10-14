import os
import pandas as pd
import sys

def extract_relation_data_from_ann_file(ann_file_path, txt_file_path, category):
    relations_data = []
    
    # Read the .ann file
    with open(ann_file_path, 'r', encoding='utf-8') as ann_file:
        ann_lines = ann_file.readlines()
        
    # Read the .txt file to get the abstract text
    with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
        abstract_text = txt_file.read().strip()
    
    # Parse entities and relations
    entities = {}
    relations = []
    
    for line in ann_lines:
        parts = line.strip().split('\t')
        if line.startswith('T'):  # Entity line
            entity_id = parts[0]
            entity_info = parts[1].split(' ')
            entity_type = entity_info[0]
            start_idx = entity_info[1]
            end_idx = entity_info[-1]
            entity_text = parts[2]
            
            # Store entity details in the dictionary
            entities[entity_id] = {
                'type': entity_type,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'text': entity_text
            }
        elif line.startswith('R'):  # Relation line
            relation_id = parts[0]
            relation_info = parts[1].split(' ')
            relation_type = relation_info[0]
            arg1 = relation_info[1].split(':')[1]
            arg2 = relation_info[2].split(':')[1]
            
            # Exclude Out-of-scope relations
            if relation_type == 'Out-of-scope':
                continue
            
            # Store relation details in the list
            relations.append({
                'relation_id': relation_id,
                'relation_type': relation_type,
                'arg1': arg1,
                'arg2': arg2
            })
    
    # Create records for each relation
    for relation in relations:
        arg1 = entities.get(relation['arg1'])
        arg2 = entities.get(relation['arg2'])
        
        # Ensure both arguments are available and valid
        if not arg1 or not arg2:
            continue
        
        # Check if either argument is a Disease and the other is a Lifestyle_factor
        if (arg1['type'] == 'disease' and arg2['type'] == 'lifestyle_factor') or \
           (arg1['type'] == 'lifestyle_factor' and arg2['type'] == 'disease'):
            
            # Append relation data to the list
            relations_data.append({
                'Disease_Entity': arg1['text'] if arg1['type'] == 'disease' else arg2['text'],
                'LSF_Entity': arg2['text'] if arg1['type'] == 'disease' else arg1['text'],
                'Relationship_Type': relation['relation_type'],
                'Publication_ID': os.path.basename(ann_file_path).replace('.ann', ''),
                'Category': category,
                'Disease_BRAT_ID': relation['arg1'] if arg1['type'] == 'disease' else relation['arg2'],
                'LSF_BRAT_ID': relation['arg2'] if arg1['type'] == 'disease' else relation['arg1'],
                'Disease_Entity_Span': f"{arg1['start_idx']}:{arg1['end_idx']}" if arg1['type'] == 'disease' else f"{arg2['start_idx']}:{arg2['end_idx']}",
                'LSF_Entity_Span': f"{arg2['start_idx']}:{arg2['end_idx']}" if arg1['type'] == 'disease' else f"{arg1['start_idx']}:{arg1['end_idx']}",
                'Abstract_Text': abstract_text
            })
    
    return relations_data

def create_consolidated_dataset(corpus_folder, output_file):
    # Define subfolders
    subfolders = ['train-set', 'dev-set', 'test-set']
    
    # List to store all relation data
    all_relation_data = []
    
    # Iterate through each subfolder
    for subfolder in subfolders:
        folder_path = os.path.join(corpus_folder, subfolder)
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist.")
            continue
        
        # Iterate through each file in the subfolder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ann'):
                ann_file_path = os.path.join(folder_path, file_name)
                txt_file_path = os.path.join(folder_path, file_name.replace('.ann', '.txt'))
                
                # Check if corresponding .txt file exists
                if not os.path.exists(txt_file_path):
                    print(f"Corresponding .txt file for {file_name} not found.")
                    continue
                
                # Extract relation data from the .ann file
                relation_data = extract_relation_data_from_ann_file(ann_file_path, txt_file_path, subfolder)
                all_relation_data.extend(relation_data)
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_relation_data)
    
    # Save the DataFrame to a TSV file in the same folder as root
    output_path = os.path.join(corpus_folder, output_file)
    df.to_csv(output_path, sep='\t', index=False)

    print(f"Consolidated dataset saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_consolidated_relations.py <corpus_folder> <output_file_name>")
        sys.exit(1)

    corpus_folder = sys.argv[1]
    output_file_name = sys.argv[2]

    create_consolidated_dataset(corpus_folder, output_file_name)
