import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


print("Loading job roles dataset...")
df = pd.read_csv('raw_skills.csv')
print(f"✓ Loaded {len(df)} job roles\n")


print("=" * 60)
print("TECH STACK RECOMMENDER SYSTEM")
print("=" * 60)
print("\nEnter your skills (minimum 3 required)")
print("Example: Python, Machine Learning, SQL\n")

user_input = input("Enter your skills (comma-separated): ").strip()


user_skills = [skill.strip() for skill in user_input.split(',')]
if len(user_skills) < 3:
    print(f"\nError: You entered {len(user_skills)} skills. Minimum 3 required.")
    exit()

user_skills_text = ', '.join(user_skills)
print(f"\n✓ Your skills: {user_skills_text}")

print("\nProcessing TF-IDF vectorization...")
all_skills = df['skills'].tolist() + [user_skills_text]
tfidf_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 3),
    lowercase=True,
    stop_words='english'
)

tfidf_matrix = tfidf_vectorizer.fit_transform(all_skills)

print("TF-IDF vectorization complete")

print("\nCalculating similarity scores.")

user_vector = tfidf_matrix[-1:]
job_vectors = tfidf_matrix[:-1]
similarities = cosine_similarity(user_vector, job_vectors)[0]

print(" Similarity calculation complete\n")

results_df = pd.DataFrame({
    'job_role': df['job_role'],
    'similarity_score': similarities
})
results_df = results_df.sort_values('similarity_score', ascending=False)
top_3 = results_df.head(3)
print("=" * 60)
print("TOP 3 RECOMMENDED JOB ROLES")
print("=" * 60)

for idx, (i, row) in enumerate(top_3.iterrows(), 1):
    print(f"\n{idx}. {row['job_role']}")
    print(f"   Similarity Score: {row['similarity_score']:.4f} ({row['similarity_score']*100:.2f}%)")

print("\n" + "=" * 60)