import pandas as pd
from click import command, option
from sklearn.feature_extraction.text import TfidfVectorizer

# Load preprocessed data
data = pd.read_csv("dblp-v10.csv")

# Option 1: Filter Out NaN Values (recommended for missing data)
# data = data.dropna(subset=['title', 'abstract'])

# Option 2: Fill NaN Values with Empty Strings
data["title"] = data["title"].fillna("")
data["abstract"] = data["abstract"].fillna("")

@command()
@option("--query", "-q", prompt="Enter your research paper search query:", required=True)
def search(query):
  """Search research papers in the dataset based on your query."""

  # Lowercase query and convert to search terms
  search_terms = query.lower().split()

  # Prepare text data (combine title and abstract for TF-IDF)
  text = data["title"] + " " + data["abstract"]

  # Create TF-IDF vectorizer
  vectorizer = TfidfVectorizer()

  # Generate TF-IDF vectors for all papers and the query
  tf_idf_matrix = vectorizer.fit_transform(text)
  query_vec = vectorizer.transform([query])

  # Calculate cosine similarity between query and each paper
  cosine_similarities = (query_vec * tf_idf_matrix.T).toarray().squeeze()

  # Sort papers by cosine similarity in descending order
  sorted_results = pd.DataFrame({'title': data['title'], 'authors': data['authors'], 'similarity': cosine_similarities})
  sorted_results = sorted_results.sort_values(by='similarity', ascending=False)

  # Limit results to top 5
  top_5_results = sorted_results.head(5)

  # Print results (modify as needed)
  if top_5_results.empty:
    print("No results found for your query.")
  else:
    print(f"\nSearch Results (Top 5) based on TF-IDF similarity:")
    for index, row in top_5_results.iterrows():
      print(f"{index+1}. {row['title']} by {row['authors']} (Similarity: {row['similarity']:.4f})")  # Display similarity score

if __name__ == "__main__":
  search()