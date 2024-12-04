import numpy as np
from sentence_transformers import SentenceTransformer
from src.vector_database import VectorDatabase
import requests
import re

def preprocess_query(query):
    """
    Preprocesses the user query to extract the main topic.

    Parameters:
        query (str): The user's input query.

    Returns:
        str: A simplified query for searching.
    """
    query = re.sub(r"(what is|who is|explain|describe|definition of|meaning of)", "", query, flags=re.IGNORECASE)
    return query.strip().title()

def is_relevant_suggestion(original, suggested):
    """
    Determines if the suggested topic is relevant to the original query.

    Parameters:
        original (str): The original query.
        suggested (str): The suggested topic.

    Returns:
        bool: True if relevant, False otherwise.
    """
    original_keywords = set(original.lower().split())
    suggested_keywords = set(suggested.lower().split())
    overlap = original_keywords & suggested_keywords
    return len(overlap) > 0

def fetch_wikipedia_articles(topic, num_articles=5, seen_topics=None, max_attempts=5):
    """
    Fetches Wikipedia articles for a given topic using the MediaWiki API.

    Parameters:
        topic (str): The topic to search for.
        num_articles (int): Number of paragraphs to fetch.
        seen_topics (set): Set of topics already processed to avoid loops.
        max_attempts (int): Maximum number of recursive attempts.

    Returns:
        list: List of paragraphs from the Wikipedia article.
    """
    if seen_topics is None:
        seen_topics = set()
    if topic in seen_topics:
        print(f"Topic '{topic}' already processed. Skipping to avoid loop.")
        return []
    if len(seen_topics) >= max_attempts:
        print("Reached maximum recursion limit. Stopping further attempts.")
        return []

    # Mark this topic as processed
    seen_topics.add(topic)

    # Preprocess the topic to improve matching
    original_topic = topic
    topic = preprocess_query(topic)

    # Define the MediaWiki API endpoint
    url = "https://en.wikipedia.org/w/api.php"

    # Step 1: Try fetching the page content by exact title
    params = {
        "action": "query",
        "format": "json",
        "titles": topic,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Parse the response to extract text content
        pages = data.get("query", {}).get("pages", {})
        content = next(iter(pages.values())).get("extract", "")

        if content:
            paragraphs = content.split("\n\n")
            print(f"Successfully fetched content for topic: '{topic}'.")
            return paragraphs[:num_articles]
    except Exception as e:
        print(f"Error fetching content from Wikipedia: {e}")
        return []

    # Step 2: If no exact match, use the search API to find related pages
    print(f"Exact match for '{original_topic}' not found. Searching for related topics...")
    search_params = {
        "action": "opensearch",
        "format": "json",
        "search": topic,
        "limit": 5,
    }

    try:
        search_response = requests.get(url, params=search_params)
        search_response.raise_for_status()
        search_data = search_response.json()

        # Select the most relevant suggestion
        for suggested_title in search_data[1]:
            if is_relevant_suggestion(original_topic, suggested_title):
                print(f"Using suggested topic: '{suggested_title}'")
                return fetch_wikipedia_articles(suggested_title, num_articles, seen_topics, max_attempts)

        # Default to the main keyword if no relevant suggestions are found
        main_keyword = preprocess_query(original_topic)
        if main_keyword and main_keyword not in seen_topics:
            print(f"No relevant suggestions found. Retrying with main keyword: '{main_keyword}'")
            return fetch_wikipedia_articles(main_keyword, num_articles, seen_topics, max_attempts)

        print(f"No related topics found for '{original_topic}'.")
        return []
    except Exception as e:
        print(f"Error searching for related topics: {e}")
        return []

def generate_embeddings(documents, model):
    """
    Generates embeddings for a list of documents.

    Parameters:
        documents (list): List of document texts.
        model (SentenceTransformer): The embedding model.

    Returns:
        np.ndarray: Numpy array of embeddings.
    """
    return model.encode(documents)


def interactive_query():
    """
    Interactive query system that dynamically fetches documents from Wikipedia,
    generates embeddings, and performs similarity search.
    """
    # Load a pre-trained SentenceTransformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Interactive query loop
    print("You can now enter queries interactively. Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your query: ").strip()
        if user_query.lower() == "exit":
            print("Exiting interactive query system.")
            break

        # Step 1: Fetch documents dynamically
        print(f"Fetching documents for topic: '{user_query}'...")
        documents = fetch_wikipedia_articles(user_query, num_articles=5)
        if not documents:
            print("No documents found for the given query. Try a different topic.")
            continue

        # Step 2: Generate embeddings for the documents
        print("Generating embeddings for fetched documents...")
        embeddings = generate_embeddings(documents, model)

        # Step 3: Initialize the Vector Database with the generated embeddings
        embedding_dim = embeddings.shape[1]
        vector_db = VectorDatabase(embedding_dim)
        vector_db.add_embeddings(embeddings)

        # Step 4: Encode the query and perform similarity search
        try:
            query_embedding = model.encode(user_query)
            distances, indices = vector_db.query(query_embedding, top_k=5)
            
            # Display results
            print("\nTop Results:")
            for idx, (distance, index) in enumerate(zip(distances[0], indices[0]), start=1):
                print(f"\n{idx}. Document Index: {index}, Distance: {distance:.4f}")
                print(f"Document Text: {documents[index][:300]}...")  # Display the first 300 characters
        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == "__main__":
    interactive_query()