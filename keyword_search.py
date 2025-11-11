from collections import Counter

STOP_WORDS = set([
    'a', 'an', 'the', 'i', 'me', 'my', 'is', 'am', 'are', 'have', 'had',
    'and', 'or', 'but', 'with', 'of', 'in', 'on', 'at', 'to', 'for', 
    'feeling', 'symptoms', 'pains', 'feels', 'like'
])

def process_query(query):
    """
    Splits a query into a list of words.
    
    Args:
        query (str): The query to split.
    """
    query = set(query.strip().lower().split(" "))
    # print("after split: ", query)
    query = query - STOP_WORDS
    # print("after stop words: ", query)
    if not query:
        return "no keywords found"
    return query

def count_keywords(processed_query, rag_df, tiny_df, key, top_n=3):
    """
    Counts the keywords in the processed query, and returns the top N matching diseases.
    
    Args:
        processed_query (set): The processed query.
        rag_df (pd.DataFrame): The RAG DataFrame.
        tiny_df (pd.DataFrame): The melted DataFrame containing disease_name and key columns.
        key (str): The column name to search in (e.g., 'symptom').
        top_n (int): Number of top matching diseases to return.
    """
    matching_disease_names = []
    # print(tiny_df)
    # print(tiny_df[key].head(10))
    for keyword in processed_query:
        matches = tiny_df[tiny_df[key].str.contains(keyword, na=False)]
        # Extract disease names from the matches
        if not matches.empty:
            disease_names = matches['disease_name'].tolist()
            matching_disease_names.extend(disease_names)
    
    if not matching_disease_names:
        return None
    
    # Count occurrences of each disease
    d_score = Counter(matching_disease_names)
    top_diseases = [disease for disease, score in d_score.most_common(top_n)]

    # print(top_diseases)
    # print("________________________________")
    try:
        context = rag_df.loc[top_diseases]
        return context
    except KeyError:
        found_diseases = [d for d in top_diseases if d in rag_df.index]
        if not found_diseases:
            return None
        context = rag_df.loc[found_diseases]
        return context

