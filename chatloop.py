from processing import retrieve_context, generate_answer, embed_query, find_nearest_neighbors, generate_differential_answer

def chatloop(new_df, model, semantic_search_ready, 
             doc_embeddings_matrix, all_disease_names, embedding_model_name):
    """
    Main chat loop for the symptom checker.
    
    Args:
        new_df (pd.DataFrame): The RAG DataFrame.
        model (genai.GenerativeModel): The model for generation.
        semantic_search_ready (bool): Flag if embeddings are loaded.
        doc_embeddings_matrix (np.array): The matrix of document vectors.
        all_disease_names (list): List of all disease names.
        embedding_model_name (str): The name of the embedding model.
    """
    while True:
        print("\n" + "="*80)
        print("What would you like to do?")
        print("1: Search by Disease Name (e.g., 'malaria')")
        if semantic_search_ready:
            print("2: Search by Symptoms (e.g., 'i feel hot and my head hurts')")
        print("Type 'quit' to exit.")
        print("="*80)
        
        mode = input("Enter your choice (1 or 2): ")
        
        if mode.lower() == 'quit':
            break
        
        # --- MODE 1: Search by Disease Name ---
        if mode == '1':
            disease_query = input("Enter the disease name: ").strip().lower()
            if not disease_query: continue
            
            context = retrieve_context(new_df, disease_query)
            
            if context is None:
                print(f"Sorry, I have no information on '{disease_query}'.")
            else:
                full_question = f"Tell me about {context.name}. What are its symptoms, causes, and treatments?"
                print("Generating answer...")
                answer = generate_answer(context, full_question, model)
                print("\n--- ANSWER ---")
                print(answer)
                print("---------------\n")

        # --- MODE 2: Semantic Search (Replaced Keyword Search) ---
        elif mode == '2' and semantic_search_ready:
            user_query = input("Describe how you feel (full sentences are ok): ").strip()
            if not user_query: continue
            
            print("Running semantic search...")
            
            # 1. Embed the query
            query_vector = embed_query(user_query, embedding_model_name)
            
            if query_vector:
                # 2. Find "Nearest Neighbors"
                top_disease_names = find_nearest_neighbors(
                    query_vector,
                    doc_embeddings_matrix,
                    all_disease_names,
                    top_n=3
                )
                
                # 3. Retrieve & Generate
                context_df = new_df.loc[top_disease_names]
                
                print("Generating answer...")
                answer = generate_differential_answer(context_df, user_query, model)
                print("\n--- ANSWER ---")
                print(answer)
                print("---------------\n")
            else:
                print("Sorry, I couldn't process your search query.")
        
        else:
            print("Invalid choice. Please try again.")