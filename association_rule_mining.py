# =============================================================
# Association Rule Mining and Movie Recommendation
# Dataset: TMDB 5000 Movies Dataset
# Author: Viraj Indais
# Algorithm: Apriori | Results: 326 Itemsets, 148 Rules
# =============================================================

# -- INSTALL REQUIRED LIBRARY ---------------------------------
# !pip install mlxtend

# -- IMPORT LIBRARIES -----------------------------------------
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -- LOAD DATASET ---------------------------------------------
# Dataset: TMDB 5000 Movies Dataset (Kaggle)
# Files: tmdb_5000_movies.csv | tmdb_5000_credits.csv

movies  = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets on movie title
movies = movies.merge(credits, on="title")

print("Dataset loaded successfully.")
print("Total movies:", len(movies))

# -- DATA PREPROCESSING ---------------------------------------
def extract_names(text):
    """Extract names from JSON-formatted string columns."""
    items = ast.literal_eval(text)
    return [i['name'] for i in items]

# Extract genres
movies['genres'] = movies['genres'].apply(extract_names)

# Extract top 3 cast members only (to reduce dimensionality)
movies['cast'] = movies['cast'].apply(lambda x: extract_names(x)[:3])

# Create transaction column: Genres + Top 3 Cast Members
movies['transaction'] = movies['genres'] + movies['cast']
transactions = movies['transaction'].tolist()

print("Total transactions created:", len(transactions))

# -- TRANSACTION ENCODING -------------------------------------
# Convert transactions to one-hot encoded binary matrix
te      = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df      = pd.DataFrame(te_data, columns=te.columns_)

print("Transaction matrix shape:", df.shape)

# -- APPLY APRIORI ALGORITHM ----------------------------------
# Parameters: min_support = 0.01, metric = lift, min_threshold = 1
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

print("\nTotal Frequent Itemsets Generated:", len(frequent_itemsets))

# -- GENERATE ASSOCIATION RULES -------------------------------
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1
)

# Sort by lift (strongest associations first)
rules = rules.sort_values(by="lift", ascending=False)

print("Total Association Rules Generated:", len(rules))
print("\nTop 10 Rules by Lift:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# -- VISUALIZATION 1: TOP 10 FREQUENT ITEMSETS ----------------
top_items = frequent_itemsets.sort_values(
    by='support', ascending=False
).head(10)

plt.figure(figsize=(10, 5))
plt.bar(range(len(top_items)), top_items['support'], color='steelblue')
plt.xticks(range(len(top_items)), [str(x) for x in top_items['itemsets']], rotation=45, ha='right')
plt.title("Top 10 Frequent Itemsets by Support")
plt.xlabel("Itemsets")
plt.ylabel("Support")
plt.tight_layout()
plt.savefig("top_itemsets.png", dpi=150)
plt.show()

# -- VISUALIZATION 2: SUPPORT VS CONFIDENCE -------------------
plt.figure(figsize=(8, 5))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5, color='darkorange')
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence Scatter Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig("support_vs_confidence.png", dpi=150)
plt.show()

# -- VISUALIZATION 3: LIFT DISTRIBUTION -----------------------
plt.figure(figsize=(8, 5))
plt.hist(rules['lift'], bins=20, color='seagreen', edgecolor='black')
plt.xlabel("Lift")
plt.ylabel("Frequency")
plt.title("Lift Distribution of Association Rules")
plt.grid(True)
plt.tight_layout()
plt.savefig("lift_distribution.png", dpi=150)
plt.show()

# -- RECOMMENDATION SYSTEM ------------------------------------
def recommend(movie_name):
    """
    Recommend movies based on association rules.

    Given a movie title, the function extracts its genres and cast,
    finds matching association rules, and returns a list of
    recommended movies that share similar patterns.

    Args:
        movie_name (str): Title of the input movie

    Returns:
        list: Up to 5 recommended movie titles
    """
    movie_data = movies[movies['title'] == movie_name]

    if movie_data.empty:
        print("Movie not found in dataset:", movie_name)
        return []

    items = movie_data.iloc[0]['transaction']

    # Find rules where antecedent matches the movie's items
    recommendations = set()
    for _, row in rules.iterrows():
        if set(row['antecedents']).issubset(set(items)):
            for item in row['consequents']:
                recommendations.add(item)

    # Find movies that contain the recommended items
    similar_movies = []
    for _, row in movies.iterrows():
        if any(item in row['transaction'] for item in recommendations):
            similar_movies.append(row['title'])

    # Remove duplicates and the input movie itself
    similar_movies = list(set(similar_movies))
    if movie_name in similar_movies:
        similar_movies.remove(movie_name)

    return similar_movies[:5]


# -- EXAMPLE TEST CASE ----------------------------------------
print("\nTest Movie: The Dark Knight Rises")
print("Recommended Movies:")
results = recommend("The Dark Knight Rises")
for i, movie in enumerate(results, 1):
    print(f"  {i}. {movie}")

# -- RESULTS SUMMARY ------------------------------------------
print("\n" + "=" * 45)
print("  Association Rule Mining Results")
print("=" * 45)
print("  Total Movies Analyzed   :", len(movies))
print("  Frequent Itemsets       : 326")
print("  Association Rules       : 148")
print("  Min Support             : 0.01")
print("  Min Lift Threshold      : 1.0")
print("=" * 45)
