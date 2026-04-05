# Association Rule Mining and Movie Recommendation — TMDB Dataset

A machine learning project that applies Association Rule Mining using the Apriori algorithm on the TMDB 5000 Movies dataset to discover hidden relationships between genres and cast members, and builds a rule-based movie recommendation system.

---

## Results

| Metric | Value |
|--------|-------|
| Total Movies Analyzed | 4803 |
| Frequent Itemsets Generated | 326 |
| Association Rules Generated | 148 |
| Minimum Support | 0.01 |
| Minimum Lift Threshold | 1.0 |

---

## About the Project

Each movie is treated as a transaction consisting of its genres and top three cast members. The Apriori algorithm mines frequent itemsets and generates association rules. Based on these rules, a recommendation system is built that suggests similar movies based on genre and cast patterns.

---

## Tech Stack

- Language: Python
- Algorithm: Apriori (mlxtend)
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib
- Encoding: TransactionEncoder (mlxtend)

---

## Dataset

- Source: TMDB 5000 Movies Dataset (Kaggle)
- Files: `tmdb_5000_movies.csv` | `tmdb_5000_credits.csv`
- Attributes used: Movie Title, Genres, Top 3 Cast Members

---

## Methodology

**1. Data Preprocessing**
- Merged movies and credits datasets on movie title
- Extracted genres and top 3 cast members from JSON-formatted columns
- Created transactions: Genres + Top 3 Cast Members per movie

**2. Transaction Encoding**
- Converted transactions to one-hot encoded binary matrix using TransactionEncoder
- Format required by the Apriori algorithm

**3. Apriori Algorithm**
- Min Support: 0.01
- Metric: Lift
- Min Lift Threshold: 1.0
- Generated frequent itemsets and derived association rules
- Rules evaluated using Support, Confidence, and Lift

---

## Key Findings

Strong associations discovered:

| Pattern | Insight |
|---------|---------|
| Animation → Family | Strong bidirectional co-occurrence |
| Adventure → Action | High confidence genre cluster |
| Drama + War → History | Thematic storytelling pattern |
| Action + Adventure → Sci-Fi | Common genre combination |

Most strong rules had lift values greater than 1.5, confirming statistically significant associations.

---

## Recommendation System

The system works as follows:

1. User enters a movie name
2. System extracts its genres and cast members
3. Matching association rules are identified from antecedents
4. Consequents from strong rules are collected
5. Movies containing those consequents are recommended

**Example:**

Input: The Dark Knight Rises

Output:
- The Dark Knight
- Batman Begins
- Inception
- Man of Steel
- Watchmen

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/indaisv/association-rule-mining-tmdb.git
cd association-rule-mining-tmdb

# 2. Install dependencies
pip install pandas numpy matplotlib mlxtend

# 3. Download dataset from Kaggle
# https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

# 4. Run the project
python association_rule_mining.py
```

---

## Visualizations Generated

- Top 10 Frequent Itemsets — Bar chart by support value
- Support vs Confidence — Scatter plot of rule strength
- Lift Distribution — Histogram of lift values across all rules

---

## Author

**Viraj Indais**
- Email: indaisviraj@gmail.com
- LinkedIn: https://linkedin.com/in/viraj-indais-48ba2526a
- GitHub: https://github.com/indaisv
