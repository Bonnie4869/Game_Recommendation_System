# Game_Recommender_System
a game recommender system based on steam, involing sentiment analysis on player reviews, content-based and collaborative filtering model, finally come up with a hybrid recommendation system.


# Hybrid Game Recommendation System

An improved hybrid recommendation system for Steam games, combining content-based filtering, collaborative filtering (SVD), and game rating/popularity scoring.

## Features
- **Content-based filtering** – Recommends games similar to those a user has played
- **Collaborative filtering (SVD)** – Uses matrix factorization to find user-item patterns
- **Rating-based scoring** – Incorporates game popularity and comprehensive scores
- **Hybrid combination** – Intelligently merges multiple recommendation approaches with adjustable weights
- **Batch processing** – Supports recommendations for multiple users at once
- **Interactive CLI** – User-friendly command-line interface

## How It Works
1. Loads precomputed game similarities, SVD model, and game ratings
2. Generates recommendations from each component
3. Combines scores using configurable weights
4. Filters out games already played by the user
5. Ranks and returns top recommendations

## environment
Requirements
Python 3.7+

pandas, numpy

Precomputed data files:
Model/
|-game_similarities.csv
|-comprehensive_scores_s0_p1.csv
|-game_svd_model.pkl
|-recommendations_small.csv
|-hybrid_recommender_improved.py


Other file description: (For display only, does not affect the final model operation)
data preparation folder:
Contains all data processing pipeline codes and raw data
(Due to different code paths, it is only for display and archiving)
Includes:
crawler/:
game_info_crawler.py: crawling game information including game description, games tag, genres
game_review_crawler.py: crawling for user rating and review for all the games

datasetset_2022/:
All initial data files

data_cleaning.ipynb:
Reducing dataset size and preliminary data cleaning

module_1/2/3:
Code corresponding to the three recommendation models

## Usage
Run the system interactively:
```bash
python hybrid_recommender_improved.py
input the user_id in "example user_id for test.txt" or column "user_id" in "recommendations_small.csv"

the result will save as .csv file

e.g. successful running sample in running_sample1.png and running_sample2.png

# dataset:
https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam
