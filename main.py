import os
import glob
import json

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Load the spaCy English model
nlp = spacy.load("en_core_web_lg")


# Function to clean strings
def clean_string(s):
    if isinstance(s, str):
        # Remove special characters, convert to lowercase, and trim spaces
        s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s).lower().strip()
        doc = nlp(s)
        return ' '.join([token.lemma_ for token in doc])


# Function to clean DataFrame strings except those with '_id' in their name
def clean_dataframe_strings(df):
    string_columns = df.select_dtypes(include='object').columns
    for col in string_columns:
        if not col.endswith('_id'):
            df[col] = df[col].apply(clean_string)
    return df


# Function to process files and extract data
def process_files():
    files = glob.glob('./Preprocessing/Fix structure/rows/*.json')
    all_records, all_entity_relationships, all_categories = [], [], []
    all_symptoms, all_conditions, all_body_parts, all_organs, all_organisms = [], [], [], [], []

    # Gender mapping dictionary
    gender_map = {'male': 'm', 'female': 'f', '': None}

    category_id = 1  # Initial category ID

    for index_id, file in enumerate(tqdm(files), start=1):
        record = {}
        with open(file, 'r') as f:
            json_data = json.load(f)

        # Add index_id and file_name to record
        record['index_id'] = index_id
        record['file_name'] = file
        record['preprocessed_review'] = " ".join(json_data.get('preprocessed_review', []))

        # Handle gender more efficiently
        gender_type = json_data.get('gender', {}).get('type', '').strip().lower()
        record['gender'] = gender_map.get(gender_type, None)

        # Append the record to the list of records
        all_records.append(record)

        # Normalize entity_relationships if present
        entity_relationships = json_data.get('entity_relationships', [])
        if entity_relationships:
            entity_relationships_df = pd.json_normalize(entity_relationships)
            entity_relationships_df['index_id'] = index_id  # Add foreign key column
            all_entity_relationships.append(entity_relationships_df)

        # Normalize categories if present
        for category in json_data.get('categories', []):
            if category:
                category_dict = {
                    'category_id': category_id,
                    'index_id': index_id,
                    'name': category.get('name', None),
                }
                all_categories.append(pd.Series(category_dict))

                for field, all_entries in zip(
                        ['medical_symptoms', 'medical_conditions', 'body_parts', 'organs', 'organisms'],
                        [all_symptoms, all_conditions, all_body_parts, all_organs, all_organisms]
                ):
                    items = category.get(field, [])
                    for item in items:
                        all_entries.append(pd.Series({
                            'category_id': category_id,
                            'index_id': index_id,
                            'value': item
                        }))
                category_id += 1

    # Convert lists to DataFrames
    main_df = pd.DataFrame(all_records)
    entity_relationships_df = pd.concat(all_entity_relationships,
                                        ignore_index=True) if all_entity_relationships else pd.DataFrame()
    categories_df = pd.DataFrame(all_categories)
    symptoms_df = pd.DataFrame(all_symptoms)
    conditions_df = pd.DataFrame(all_conditions)
    body_parts_df = pd.DataFrame(all_body_parts)
    organs_df = pd.DataFrame(all_organs)
    organisms_df = pd.DataFrame(all_organisms)

    # Clean all string columns except those with '_id' in their name
    # main_df = clean_dataframe_strings(main_df)
    entity_relationships_df = clean_dataframe_strings(entity_relationships_df)
    categories_df = clean_dataframe_strings(categories_df)
    symptoms_df = clean_dataframe_strings(symptoms_df)
    conditions_df = clean_dataframe_strings(conditions_df)
    body_parts_df = clean_dataframe_strings(body_parts_df)
    organs_df = clean_dataframe_strings(organs_df)
    organisms_df = clean_dataframe_strings(organisms_df)

    return main_df, entity_relationships_df, categories_df, symptoms_df, conditions_df, body_parts_df, organs_df, organisms_df


def break_to_files():
    main_df, relationships_df, categories_df, symptoms_df, conditions_df, body_parts_df, organs_df, organisms_df = process_files()

    # Save DataFrames to CSV
    main_df.to_csv('data/main.csv', index=False)
    relationships_df.to_csv('data/entity_relationships.csv', index=False)
    categories_df.to_csv('data/categories.csv', index=False)
    symptoms_df.to_csv('data/symptoms.csv', index=False)
    conditions_df.to_csv('data/conditions.csv', index=False)
    body_parts_df.to_csv('data/body_parts.csv', index=False)
    organs_df.to_csv('data/organs.csv', index=False)
    organisms_df.to_csv('data/organisms.csv', index=False)

    print("DataFrames saved successfully.")


def compute_and_save_embeddings():
    # Load the saved DataFrames
    data_files = {
        # 'main': 'data/main.csv',
        'entity_relationships': 'data/entity_relationships.csv',
        # 'categories': 'data/categories.csv',
        'symptoms': 'data/symptoms.csv',
        'conditions': 'data/conditions.csv',
        # 'body_parts': 'data/body_parts.csv',
        # 'organs': 'data/organs.csv',
        # 'organisms': 'data/organisms.csv'
    }
    output_dir = 'vectors'
    os.makedirs(output_dir, exist_ok=True)

    for file_name, file_path in data_files.items():
        print(f"Processing {file_name}...")
        df = pd.read_csv(file_path)

        columns_to_embed = [col for col in df.columns if col in ['relation', 'value', 'entity1', 'entity2', ]]

        for column in columns_to_embed:
            # Drop NaN, empty strings, and duplicates, and convert to strings
            unique_values_df = df[[column]].dropna().drop_duplicates().reset_index(drop=True)
            unique_values_df[column] = unique_values_df[column].astype(str).replace('', pd.NA).dropna()

            # Add a unique ID for each unique value
            unique_values_df['unique_id'] = range(1, len(unique_values_df) + 1)

            # Initialize a list to store valid embeddings
            valid_embeddings = []
            valid_values = []

            # Compute embeddings with progress tracking
            for value in tqdm(unique_values_df[column], desc=f"Embedding {column} in {file_name}"):
                if value:  # Ensure value is not empty or NaN
                    embedding = model.encode(value).tolist()
                    if embedding:  # Ensure that embedding is valid
                        valid_embeddings.append(embedding)
                        valid_values.append(value)

            # Create a DataFrame with the valid values and embeddings
            final_df = pd.DataFrame({
                'unique_id': range(1, len(valid_values) + 1),
                column: valid_values,
                f'vector_{column}': valid_embeddings
            })

            # Save the DataFrame with unique_id, original values, and vector columns
            output_file_path = os.path.join(output_dir, f'vector_{file_name}_{column}.csv')
            final_df.to_csv(output_file_path, index=False)
            # print(f"Saved {output_file_path}")

    print("All DataFrames with embeddings saved successfully.")


if __name__ == "__main__":
    # filename = 'vector_conditions_value.csv'
    # column = filename.split('_')[-1].split('.')[0]
    # vector_column = f'vector_{column}'
    # header = filename.split('_')[1]
    # df_relations = pd.read_csv(f'vectors/{filename}')
    # break_to_files()
    # compute_and_save_embeddings()
    print('hi')
# # Load the CSV file containing the embeddings
# file_path = 'vectors/vector_organisms.csv'  # Update this with the correct path to your CSV file
# df = pd.read_csv(file_path)
# # Extract the embeddings for the first 10 values
# values = df['value'][:10].tolist()  # Get the original values (strings)
# embeddings = df['vector_value'][:10].apply(eval).apply(np.array).tolist()
# # Convert to a numpy array for easier manipulation
# embeddings = np.array(embeddings)
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Step 3: Calculate pairwise cosine similarities
# similarity_matrix = cosine_similarity(embeddings)
#
# # Step 4: Loop through the similarity matrix and print the results
# print("Pairwise similarity results:")
# for i in range(len(values)):
#     for j in range(i + 1, len(values)):
#         print(f"({values[i]}) and ({values[j]}) = {similarity_matrix[i][j]:.4f}")
