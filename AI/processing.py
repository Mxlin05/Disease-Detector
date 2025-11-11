import numpy as np
import google.generativeai as genai

def clean_data(df, index_name, column_mapping):
    """
    Cleans a DataFrame by setting the index, renaming columns, and filling empty values.
    
    Args:
        df (pd.DataFrame): The DataFrame to clean.
        index_name (str): The name for the index column.
        column_mapping (dict): A dictionary mapping the column names to the new names.
    """
    df.set_index(0, inplace=True)
    df.index.name = index_name
    df.rename(columns=column_mapping, inplace=True)

    # print(df.head())
    # print(df.columns)
    #clean the data
    df.fillna('', inplace=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.lower()
    df.index = df.index.str.strip().str.lower()
    
    if df.index.duplicated().sum() > 0:
        print("removing: ", df.index.duplicated().sum())
        df = df[~df.index.duplicated(keep='first')]
    if df.duplicated().sum() > 0:
        print("removing: ", df.duplicated().sum())
        df = df[~df.duplicated(keep='first')]

    return df

def format_data(df, id_vars, value_vars, new_value_name):
    """
    Melts a DataFrame from wide to long format.
    
    Args:
        df (pd.DataFrame): The DataFrame to melt.
        id_vars (list): List of columns to keep as identifiers.
        value_vars (list): List of columns to unpivot.
        new_value_name (str): The name for the new column holding the values.
    """
    # print(df.info())
    df_unpivot = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='source',  # Temporary column
        value_name=new_value_name
    )
    
    # Drop the temporary 'source' column
    df_unpivot.drop(columns=['source'], inplace=True)
    
    # Drop rows that were originally empty strings
    df_unpivot = df_unpivot[df_unpivot[new_value_name] != '']
    
    return df_unpivot

def prep_RAG(dfs, base_df):
    """
    Prepares a dictionary of DataFrames for RAG.
    
    Args:
        dfs (dict): A dictionary of DataFrames to prepare.
    """

    aggregated_dfs = [base_df]

    for key, df in dfs.items():
        # print("--------------------------------")
        # print(f"Aggregating: {key}s")       
        agg_series = df.groupby('disease_name')[key].apply(', '.join)
        # print(agg_series.head(10))
        agg_series.name = f"{key}"
        
        aggregated_dfs.append(agg_series)
    
    rag_df = base_df.join(aggregated_dfs[1:])
    rag_df.fillna('N/A', inplace=True)

    return rag_df

def retrieve_context(rag_df, disease_query):
    """
    Retrieves the context document for a given disease query.
    
    Args:
        rag_df (pd.DataFrame): Your aggregated RAG DataFrame.
        disease_query (str): The name of the disease to search for.
    """
    # Clean the query just like your index
    query = disease_query.strip().lower()
    
    # 1. Try to find a match in the main 'disease_name' index
    if query in rag_df.index:
        # .loc[query] pulls the entire row as a Series
        context_series = rag_df.loc[query]
        return context_series
        
    # 2. If not found, try to find a match in the 'alt_name' column
    # This is why adding 'alt_name' was so important!
    alt_match = rag_df[rag_df['alt_name'] == query]
    
    if not alt_match.empty:
        # .iloc[0] gets the first (and likely only) matching row
        return alt_match.iloc[0]

    # 3. If no match, return None
    return None

def generate_answer(context, user_question, model):
    """
    Augments a prompt with context and generates an answer.
    
    Args:
        context (pd.Series): The row of data for the disease.
        user_question (str): The original question from the user.
        model (genai.GenerativeModel): The model to use for generation.
    """
    
    # --- This is the "Augment" part ---
    # Convert the pandas Series (our context) into a formatted string
    context_str = f"""--- CONTEXT ---
        Disease: {context.name}
        Description: {context.description}
        Alternative Name: {context.alt_name}
        Symptoms: {context.symptom}
        Causes: {context.cause}
        Treatments: {context.treatment}
        Diagnosis: {context.diagnosis}
        Complications: {context.complication}
        Prognosis: {context.prognosis}
        Severity: {context.severity}
        Region: {context.region}
    --- END CONTEXT ---"""
    
    # Create the final prompt
    prompt = f"""
    You are a helpful medical assistant. Based *only* on the context provided,
    answer the user's question. Do not use any outside knowledge.
    If the context does not contain the answer, say so.

    {context_str}

    User Question: {user_question}
    
    Answer:
    """
    
    # --- This is the "Generate" part ---
    # (Assuming 'model' is your genai.GenerativeModel("gemini-2.5-flash"))
    response = model.generate_content(prompt)
    return response.text

def generate_differential_answer(context_df, user_query, model):
    """
    Generates a differential diagnosis-style answer.
    
    Args:
        context_df (pd.DataFrame): The DataFrame of top N matching diseases.
        user_query (str): The original user query.
        model (genai.GenerativeModel): The model to use.
    """
    context_str = ""
    context_str += f"""--- CONTEXT ---
    Here is a list of potential diseases that match the user's symptoms,
    along with their causes and treatments:\n"""
    # Loop through each matching disease (each row in the context_df)
    for disease_name, data in context_df.iterrows():
        context_str += f"""
        Disease: {disease_name}
        Symptoms: {data['symptom']}
        Causes: {data['cause']}
        Treatments: {data['treatment']}
        Complications: {data['complication']}
        Prognosis: {data['prognosis']}
        Severity: {data['severity']}
        Region: {data['region']}
        ---
        """
    context_str += """--- END CONTEXT ---"""

    # Create the final prompt
    prompt = f"""
    You are a helpful medical assistant. Based *only* on the context provided,
    analyze the potential diseases and present them to the user.
    Do not use any outside knowledge.
    
    Start by listing the diseases that match the user's query,
    then provide common symptoms as well as their potential causes, treatments, complications, prognosis, severity, and region as listed.
    
    {context_str}

    User Question: {user_query}
    
    Answer:
    """
    
    # --- This is the "Generate" part ---
    response = model.generate_content(prompt)
    return response.text

def setup_embedding(rag_df, embedding_model, document_text_file):
    """
    Sets up the embedding model for the RAG dataframe.
    Args:
        rag_df (pd.DataFrame): The RAG dataframe.
        embedding_model (str): The embedding model to use.
    """
    rag_df = create_document_text(rag_df)
    rag_df.to_json(document_text_file, orient='index')
    documents_list = rag_df['document_text'].tolist()
    doc_embeddings = embed_documents(documents_list, embedding_model)

    return doc_embeddings

def create_document_text(rag_df):
    """
    Creates a single "document" string for each disease
    for embedding.
    """
    print("Creating text documents for embedding...")
    # Combine all useful text fields into one string
    def combine_texts(row):
        # We use .get(col, 'N/A') to be safe
        text = f"Disease: {row.name}. " \
               f"Description: {row.get('description', 'N/A')}. " \
               f"Symptoms: {row.get('symptom', 'N/A')}. " \
               f"Causes: {row.get('cause', 'N/A')}. " \
               f"Treatments: {row.get('treatment', 'N/A')}. " \
               f"Complications: {row.get('complication', 'N/A')}. " \
               f"Prognosis: {row.get('prognosis', 'N/A')}. " \
               f"Severity: {row.get('severity', 'N/A')}. " \
               f"Diagnosis: {row.get('diagnosis', 'N/A')}. " \
               f"Region: {row.get('region', 'N/A')}."
        return text

    rag_df['document_text'] = rag_df.apply(combine_texts, axis=1)
    print("Text documents created.")
    return rag_df

def embed_documents(documents_list, embedding_model):
    """
    Calls the Google AI API to embed a batch of documents.
    NOTE: This requires the 'model' object to be passed in.
    """
    print(f"Embedding {len(documents_list)} documents... (This may take a moment)")
    
    # Use the embedding model
    # Note: In a real app, you'd handle API errors, etc.
    try:
        result = genai.embed_content(
            model=embedding_model,
            content=documents_list,
            task_type="RETRIEVAL_DOCUMENT" # Critical for search
        )
        print("Embeddings created successfully.")
        return result['embedding']
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None

def embed_query(query, model):
    """
    Embeds a single user query.
    """
    try:
        result = genai.embed_content(
            model=model, # Use the embedding model string
            content=query,
            task_type="RETRIEVAL_QUERY" # Critical for search
        )
        return result['embedding']
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None

def find_nearest_neighbors(query_vector, doc_embeddings_matrix, all_disease_names, top_n=3):
    """
    Finds the top_n closest document vectors to the query vector
    using cosine similarity (calculated with numpy).
    """
    
    #similarity equation: dot product of query and embeded data / (norm of query * norm of data)
    # Calculate dot products
    dot_products = np.dot(doc_embeddings_matrix, query_vector)
    
    # Calculate norms
    doc_norms = np.linalg.norm(doc_embeddings_matrix, axis=1)
    query_norm = np.linalg.norm(query_vector)
    
    # Calculate similarities
    similarities = dot_products / (doc_norms * query_norm) #provides a similarity score for each disease row
    
    top_n_indices = np.argsort(similarities)[-top_n:][::-1] #sort lowest to highest, grab last 3 and reverses it to get top 3
    
    top_n_diseases = [all_disease_names[i] for i in top_n_indices] #get the disease names via matching indices
    
    return top_n_diseases