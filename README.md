# Tech Stack Recommender

A content-based recommendation system that analyzes user skills and recommends the top 3 most relevant tech job roles using TF-IDF vectorization and cosine similarity.

## Project Overview

This project is part of an AI internship initiative to build intelligent recommendation systems. It takes user technical skills as input and matches them against a dataset of job roles to provide personalized job recommendations.

## Features

- **TF-IDF Vectorization**: Converts text-based skills into numerical vectors for comparison
- **Cosine Similarity**: Calculates relevance scores between user skills and job roles
- **Top 3 Recommendations**: Returns the most suitable job roles with similarity percentages
- **Input Validation**: Ensures users provide at least 3 skills
- **User-Friendly Interface**: Clear console output with formatted results

## Tech Stack

- **Language**: Python 3.x
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `scikit-learn` - TF-IDF vectorization and cosine similarity
  - `numpy` - Numerical operations

## Project Structure

```
decodelabs_project_3/
├── raw_skills.csv                  # Dataset of job roles and skills
├── tech_stack_recommender.py        # Main recommendation script
└── README.md                        # This file
```

## Dataset Format

The `raw_skills.csv` file contains job roles and their required skills:

| job_role | skills |
|----------|--------|
| Data Scientist | Python, Machine Learning, Statistics, SQL, ... |
| DevOps Engineer | Docker, Kubernetes, AWS, CI/CD, ... |
| ... | ... |

Currently includes **10 job roles** with diverse technical skills.

## Installation

1. Clone or navigate to the project directory:
```bash
cd /Users/harinik/decodelabs_project_3
```

2. Install required dependencies:
```bash
pip install pandas scikit-learn
```

## Usage

Run the recommendation system:

```bash
python tech_stack_recommender.py
```

### Example Interaction

```
TECH STACK RECOMMENDER SYSTEM

Enter your skills (minimum 3 required)
Example: Python, Machine Learning, SQL

Enter your skills (comma-separated): Python, Machine Learning, SQL

✓ Your skills: Python, Machine Learning, SQL

Processing TF-IDF vectorization...
✓ TF-IDF vectorization complete

Calculating similarity scores...
✓ Similarity calculation complete

============================================================
TOP 3 RECOMMENDED JOB ROLES
============================================================

1. Data Scientist
   Similarity Score: 0.8521 (85.21%)

2. Machine Learning Engineer
   Similarity Score: 0.8234 (82.34%)

3. Backend Developer
   Similarity Score: 0.5643 (56.43%)

============================================================
```

## How It Works

### Step 1: Load Dataset
Reads the CSV file containing job roles and their required skills.

### Step 2: User Input
Accepts minimum 3 comma-separated skills from the user with validation.

### Step 3: TF-IDF Vectorization
- Converts all skills text (job roles + user input) into numerical vectors
- Uses character-level n-grams (2-3 characters) to capture skill similarities
- Builds a shared vocabulary for consistent comparison

### Step 4: Cosine Similarity
- Calculates the cosine similarity between user's skill vector and each job role vector
- Scores range from 0 (no similarity) to 1 (perfect match)

### Step 5: Results
- Sorts recommendations by similarity score (descending)
- Displays top 3 job roles with their similarity percentages

## Algorithm Details

**TF-IDF Parameters:**
- `analyzer='char'` - Character-level analysis
- `ngram_range=(2, 3)` - Uses 2-3 character sequences
- `lowercase=True` - Normalizes text
- `stop_words='english'` - Removes common English words

**Similarity Metric:** Cosine Similarity (0-1 scale)

## Sample Skills for Testing

Try these skill combinations:

1. **Data Science Path**: Python, Machine Learning, Statistics
2. **DevOps Path**: Docker, Kubernetes, AWS
3. **Web Development**: JavaScript, React, SQL
4. **AI Research**: Deep Learning, PyTorch, Mathematics

## Limitations & Future Improvements

**Current Limitations:**
- Fixed dataset of 10 job roles
- No skill level distinction (beginner, intermediate, expert)
- No geographic or salary considerations

**Possible Enhancements:**
- Add more job roles and skills to the dataset
- Implement skill proficiency levels
- Add salary range information
- Build a web interface using Flask/Django
- Add skill gap analysis (what skills to learn next)
- Use advanced NLP models (word embeddings, transformer models)

## Author

Created for AI Internship Project 3 at DecodeLabs

## License

This project is open source and available for educational purposes.

---

**Note:** Ensure `raw_skills.csv` is in the same directory as the Python script for the program to run correctly.