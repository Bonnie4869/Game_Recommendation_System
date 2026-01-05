# hybrid_recommender_improved.py
"""
Improved Hybrid Game Recommendation System
Integrates content-based filtering from final(1).py with SVD collaborative filtering
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime

class ContentBasedRecommender:
    """
    Content-based recommender based on game similarities
    Improved version using algorithm from final(1).py
    """
    
    def __init__(self, similarity_data_path: str):
        """
        Initialize content-based recommender
        
        Args:
            similarity_data_path: Path to game_similarities.csv
        """
        self.similarity_df = None
        self.similarity_index = {}
        self.load_similarity_data(similarity_data_path)
    
    def load_similarity_data(self, file_path: str) -> bool:
        """Load game similarity data"""
        try:
            self.similarity_df = pd.read_csv(file_path)
            
            # Standardize app_id as strings
            self.similarity_df['app_id_1'] = self.similarity_df['app_id_1'].apply(lambda x: str(int(float(x))))
            self.similarity_df['app_id_2'] = self.similarity_df['app_id_2'].apply(lambda x: str(int(float(x))))
            
            # Build similarity index for faster lookup
            print(f"  Building similarity index...")
            self._build_similarity_index()
            
            print(f"  Loaded {len(self.similarity_df)} similarity pairs")
            print(f"  Unique games in index: {len(self.similarity_index)}")
            return True
            
        except Exception as e:
            print(f"  Error loading similarity data: {e}")
            return False
    
    def _build_similarity_index(self):
        """Build bidirectional similarity index"""
        self.similarity_index = {}
        
        for _, row in self.similarity_df.iterrows():
            try:
                a = str(row['app_id_1'])
                b = str(row['app_id_2'])
                sim = float(row['combined_similarity'])
                
                if sim > 0:
                    # Add bidirectional entries
                    self.similarity_index.setdefault(a, []).append((b, sim))
                    self.similarity_index.setdefault(b, []).append((a, sim))
            except (ValueError, TypeError) as e:
                continue
        
        # Sort each game's similarity list by similarity score
        for game_id in self.similarity_index:
            self.similarity_index[game_id].sort(key=lambda x: x[1], reverse=True)
    
    def get_user_play_history(self, user_id: str, reviews_data_path: str) -> Dict[str, float]:
        """
        Get user's play history with hours
        
        Args:
            user_id: User ID
            reviews_data_path: Path to reviews/playtime data
        
        Returns:
            Dictionary of {game_id: hours_played}
        """
        try:
            # Try to load the user's play history
            reviews_df = pd.read_csv(reviews_data_path)
            
            # Filter for this user
            user_history = reviews_df[reviews_df['user_id'] == int(user_id)]
            
            if user_history.empty:
                print(f"    User {user_id} not found in reviews data")
                return {}
            
            # Standardize app_id as strings and extract hours
            user_history = user_history.copy()
            user_history['app_id'] = user_history['app_id'].apply(lambda x: str(int(float(x))))
            
            # Create dictionary of {game_id: hours}
            history_dict = dict(zip(
                user_history['app_id'].astype(str),
                user_history['hours'].astype(float)
            ))
            
            print(f"    Found {len(history_dict)} games played by user")
            return history_dict
            
        except Exception as e:
            print(f"    Error loading user history: {e}")
            return {}
    
    def recommend_for_user(self, user_id: str, reviews_data_path: str, 
                          top_k: int = 100, weight_by_hours: bool = True) -> List[Tuple[str, float]]:
        """
        Generate content-based recommendations for a user
        
        Args:
            user_id: User ID
            reviews_data_path: Path to reviews data
            top_k: Number of recommendations to generate
            weight_by_hours: Whether to weight by play hours
        
        Returns:
            List of (game_id, score) tuples
        """
        # Get user's play history
        user_history = self.get_user_play_history(user_id, reviews_data_path)
        
        if not user_history:
            print(f"    No play history found for user {user_id}")
            return []
        
        played_game_ids = set(user_history.keys())
        scores = {}
        
        print(f"    Processing {len(played_game_ids)} played games...")
        
        for game_id, hours in user_history.items():
            if game_id not in self.similarity_index:
                continue
            
            # Weight based on play hours (optional)
            weight = hours if weight_by_hours else 1.0
            
            # Get similar games
            similar_games = self.similarity_index.get(game_id, [])
            
            for candidate, similarity in similar_games:
                # Skip games user has already played
                if candidate in played_game_ids:
                    continue
                
                # Calculate weighted score
                current_score = scores.get(candidate, 0.0)
                scores[candidate] = current_score + (weight * similarity)
        
        if not scores:
            print(f"    No similar games found for user's played games")
            return []
        
        # Sort by score and get top_k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = sorted_scores[:top_k]
        
        print(f"    Generated {len(recommendations)} content recommendations")
        return recommendations
    
    def get_content_scores_df(self, user_id: str, reviews_data_path: str, 
                             top_k: int = 100) -> pd.DataFrame:
        """
        Get content recommendations as DataFrame
        
        Returns:
            DataFrame with columns ['app_id', 'content_score']
        """
        recommendations = self.recommend_for_user(user_id, reviews_data_path, top_k)
        
        if not recommendations:
            return pd.DataFrame(columns=['app_id', 'content_score'])
        
        df = pd.DataFrame(recommendations, columns=['app_id', 'content_score'])
        
        # Normalize scores to 0-1 range
        if len(df) > 0 and df['content_score'].max() > 0:
            df['content_score'] = df['content_score'] / df['content_score'].max()
        
        return df

class SVDRecommender:
    """SVD-based collaborative filtering recommender"""
    
    def __init__(self, n_factors=50):
        self.K = n_factors
        self.user_mapper = None
        self.app_mapper = None
        self.user_inv_mapper = None
        self.app_inv_mapper = None
        self.R = None
        self.all_user_predicted_ratings = None
    
    def load_model_safely(self, model_path: str):
        """Safely load SVD model from pickle file"""
        if not os.path.exists(model_path):
            print(f"  Model file not found: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Verify model has required attributes
            required_attrs = ['user_mapper', 'app_mapper', 'user_inv_mapper', 
                            'app_inv_mapper', 'R', 'all_user_predicted_ratings']
            
            missing = [attr for attr in required_attrs if not hasattr(model, attr)]
            if missing:
                print(f"  Model missing attributes: {missing}")
                return None
            
            return model
            
        except Exception as e:
            print(f"  Error loading SVD model: {e}")
            return None
    
    def get_recommendations(self, user_id: str, top_n: int = 100) -> pd.DataFrame:
        """
        Get SVD recommendations for user
        
        Returns:
            DataFrame with columns ['app_id', 'svd_score']
        """
        if self.all_user_predicted_ratings is None:
            return pd.DataFrame(columns=['app_id', 'svd_score'])
        
        try:
            user_index = self.user_mapper[user_id]
        except (KeyError, TypeError):
            return pd.DataFrame(columns=['app_id', 'svd_score'])
        
        predicted_ratings = self.all_user_predicted_ratings[user_index]
        
        # Get played games
        played_indices = self.R[user_index, :].nonzero()[1] if self.R is not None else []
        
        # Mask played games
        temp_ratings = predicted_ratings.copy()
        temp_ratings[list(played_indices)] = -np.inf
        
        # Get top recommendations
        top_indices = temp_ratings.argsort()[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            if idx in self.app_inv_mapper:
                app_id = self.app_inv_mapper[idx]
                score = predicted_ratings[idx]
                recommendations.append((str(app_id), score))
        
        if not recommendations:
            return pd.DataFrame(columns=['app_id', 'svd_score'])
        
        df = pd.DataFrame(recommendations, columns=['app_id', 'svd_score'])
        
        # Normalize scores
        if df['svd_score'].max() > 0:
            df['svd_score'] = df['svd_score'] / df['svd_score'].max()
        
        return df

class RatingRecommender:
    """Game rating/popularity recommender"""
    
    def __init__(self, ratings_data_path: str):
        self.ratings_df = None
        self.game_title_map = {}
        self.game_rating_dict = {}
        self.load_ratings_data(ratings_data_path)
    
    def load_ratings_data(self, file_path: str) -> bool:
        """Load comprehensive game ratings"""
        try:
            self.ratings_df = pd.read_csv(file_path)
            
            # Create string version of app_id for consistency
            self.ratings_df['app_id_str'] = self.ratings_df['app_id'].astype(str)
            
            # Create game title mapping
            self.game_title_map = dict(zip(
                self.ratings_df['app_id_str'],
                self.ratings_df['title']
            ))
            
            # Create normalized rating dictionary (0-1 range)
            max_score = self.ratings_df['comprehensive_score'].max()
            if max_score > 0:
                self.game_rating_dict = dict(zip(
                    self.ratings_df['app_id_str'],
                    self.ratings_df['comprehensive_score'] / max_score
                ))
            else:
                self.game_rating_dict = dict(zip(
                    self.ratings_df['app_id_str'],
                    self.ratings_df['comprehensive_score']
                ))
            
            print(f"  Loaded ratings for {len(self.game_rating_dict)} games")
            return True
            
        except Exception as e:
            print(f"  Error loading ratings data: {e}")
            return False
    
    def get_ratings_for_games(self, game_ids: List[str]) -> pd.DataFrame:
        """
        Get normalized ratings for specified games
        
        Returns:
            DataFrame with columns ['app_id', 'rating_score']
        """
        scores = []
        
        for game_id in game_ids:
            game_id_str = str(game_id)
            
            # Try to find rating
            rating = self.game_rating_dict.get(game_id_str, 0.0)
            
            if rating > 0:
                scores.append((game_id_str, rating))
        
        if not scores:
            return pd.DataFrame(columns=['app_id', 'rating_score'])
        
        df = pd.DataFrame(scores, columns=['app_id', 'rating_score'])
        
        # Ensure scores are in 0-1 range
        if df['rating_score'].max() > 1.0:
            df['rating_score'] = df['rating_score'] / df['rating_score'].max()
        
        return df
    
    def get_game_title(self, game_id: str) -> str:
        """Get game title by ID"""
        game_id_str = str(game_id)
        return self.game_title_map.get(game_id_str, f"Game {game_id_str}")

class HybridGameRecommender:
    """
    Improved Hybrid Game Recommendation System
    Combines content-based, collaborative filtering, and rating-based recommendations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize hybrid recommender
        
        Args:
            config: Configuration dictionary with paths and weights
        """
        # Default configuration
        self.config = config or {
            'paths': {
                'ratings': 'comprehensive_scores_s0_p1.csv',
                'similarities': 'game_similarities.csv',
                'svd_model': 'game_svd_model.pkl',
                'reviews': 'recommendations_small.csv'  # User play history data
            },
            'weights': {
                'content': 0.3,
                'svd': 0.3,
                'rating': 0.4
            },
            'parameters': {
                'top_k_content': 100,
                'top_k_svd': 100,
                'final_top_n': 20,
                'weight_by_hours': True
            }
        }
        
        # Initialize component recommenders
        self.content_recommender = None
        self.svd_recommender = None
        self.rating_recommender = None
        
        # Data
        self.user_history_cache = {}
    
    def load_data(self) -> bool:
        """Load all required data"""
        print("=" * 60)
        print("Loading Hybrid Recommendation System Data")
        print("=" * 60)
        
        try:
            # Load ratings data
            print("\n1. Loading game ratings...")
            ratings_path = self.config['paths']['ratings']
            self.rating_recommender = RatingRecommender(ratings_path)
            
            # Load similarity data for content-based recommendations
            print("\n2. Loading game similarities...")
            similarities_path = self.config['paths']['similarities']
            self.content_recommender = ContentBasedRecommender(similarities_path)
            
            # Load SVD model
            print("\n3. Loading SVD collaborative filtering model...")
            svd_path = self.config['paths']['svd_model']
            
            # Try multiple possible model files
            possible_paths = [
                svd_path,
                svd_path.replace('.pkl', '_fixed.pkl'),
                svd_path.replace('.pkl', '_recreated.pkl'),
                'new_svd_model.pkl'
            ]
            
            for model_path in possible_paths:
                if os.path.exists(model_path):
                    print(f"  Found model: {model_path}")
                    svd_model = SVDRecommender()
                    loaded_model = svd_model.load_model_safely(model_path)
                    
                    if loaded_model is not None:
                        self.svd_recommender = loaded_model
                        print(f"  ✓ SVD model loaded successfully")
                        print(f"    Users: {len(self.svd_recommender.user_mapper)}")
                        print(f"    Games: {len(self.svd_recommender.app_mapper)}")
                        break
            
            if self.svd_recommender is None:
                print("  ⚠️ Could not load SVD model")
                print("    Will use content-based and rating-based recommendations only")
                # Adjust weights
                total = self.config['weights']['content'] + self.config['weights']['rating']
                if total > 0:
                    self.config['weights']['content'] = self.config['weights']['content'] / total
                    self.config['weights']['svd'] = 0.0
                    self.config['weights']['rating'] = self.config['weights']['rating'] / total
            
            print("\n" + "=" * 60)
            print("Data loading completed successfully!")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_user_played_games(self, user_id: str) -> Set[str]:
        """Get set of games played by user"""
        if self.svd_recommender is None:
            return set()
        
        try:
            if user_id not in self.svd_recommender.user_mapper:
                return set()
            
            user_index = self.svd_recommender.user_mapper[user_id]
            played_indices = self.svd_recommender.R[user_index, :].nonzero()[1]
            
            played_games = set()
            for idx in played_indices:
                if idx in self.svd_recommender.app_inv_mapper:
                    game_id = str(self.svd_recommender.app_inv_mapper[idx])
                    played_games.add(game_id)
            
            return played_games
            
        except Exception:
            return set()
    
    def combine_recommendations(self, user_id: str) -> pd.DataFrame:
        """
        Combine recommendations from all models
        
        Returns:
            DataFrame with combined scores
        """
        params = self.config['parameters']
        weights = self.config['weights']
        
        print(f"\nGenerating recommendations for user {user_id}")
        print("-" * 40)
        
        # Get played games to filter out
        played_games = self.get_user_played_games(user_id)
        print(f"User has played {len(played_games)} games")
        
        # Get content-based recommendations
        print("\n1. Content-based recommendations...")
        reviews_path = self.config['paths']['reviews']
        
        if os.path.exists(reviews_path):
            content_rec = self.content_recommender.get_content_scores_df(
                user_id, 
                reviews_path,
                top_k=params['top_k_content']
            )
            print(f"   Generated {len(content_rec)} content recommendations")
        else:
            print(f"   Reviews file not found: {reviews_path}")
            content_rec = pd.DataFrame(columns=['app_id', 'content_score'])
        
        # Get SVD recommendations
        print("\n2. Collaborative filtering recommendations...")
        if self.svd_recommender is not None:
            svd_rec = self.svd_recommender.get_recommendations(
                user_id, 
                top_n=params['top_k_svd']
            )
            print(f"   Generated {len(svd_rec)} SVD recommendations")
        else:
            svd_rec = pd.DataFrame(columns=['app_id', 'svd_score'])
            print("   SVD model not available")
        
        # Combine candidate games from both models
        all_candidates = set()
        
        if len(content_rec) > 0:
            all_candidates.update(content_rec['app_id'].tolist())
        
        if len(svd_rec) > 0:
            all_candidates.update(svd_rec['app_id'].tolist())
        
        # Remove played games
        all_candidates = all_candidates - played_games
        
        if len(all_candidates) == 0:
            print("❌ No candidate games after filtering")
            return pd.DataFrame()
        
        print(f"\nTotal unique candidates: {len(all_candidates)}")
        
        # Create combined DataFrame
        combined_df = pd.DataFrame({'app_id': list(all_candidates)})
        
        # Merge content scores
        if len(content_rec) > 0:
            combined_df = combined_df.merge(
                content_rec, 
                on='app_id', 
                how='left'
            )
        else:
            combined_df['content_score'] = 0.0
        
        # Merge SVD scores
        if len(svd_rec) > 0:
            combined_df = combined_df.merge(
                svd_rec,
                on='app_id',
                how='left'
            )
        else:
            combined_df['svd_score'] = 0.0
        
        # Get rating scores
        print("\n3. Getting game ratings...")
        rating_scores = self.rating_recommender.get_ratings_for_games(
            combined_df['app_id'].tolist()
        )
        print(f"   Found ratings for {len(rating_scores)} games")
        
        if len(rating_scores) > 0:
            combined_df = combined_df.merge(
                rating_scores,
                on='app_id',
                how='left'
            )
        else:
            combined_df['rating_score'] = 0.0
        
        # Fill NaN values with 0
        combined_df = combined_df.fillna(0.0)
        
        # Adjust weights based on available models
        active_weights = weights.copy()
        
        if combined_df['content_score'].max() == 0:
            active_weights['content'] = 0.0
        
        if combined_df['svd_score'].max() == 0:
            active_weights['svd'] = 0.0
        
        if combined_df['rating_score'].max() == 0:
            active_weights['rating'] = 0.0
        
        # Renormalize weights
        total_weight = sum(active_weights.values())
        if total_weight > 0:
            for key in active_weights:
                active_weights[key] = active_weights[key] / total_weight
        else:
            # Fallback to equal weights
            active_weights = {'content': 0.33, 'svd': 0.34, 'rating': 0.33}
        
        print(f"\nUsing weights: Content={active_weights['content']:.3f}, "
              f"SVD={active_weights['svd']:.3f}, Rating={active_weights['rating']:.3f}")
        
        # Calculate final weighted score
        combined_df['final_score'] = (
            combined_df['content_score'] * active_weights['content'] +
            combined_df['svd_score'] * active_weights['svd'] +
            combined_df['rating_score'] * active_weights['rating']
        )
        
        # Sort by final score
        combined_df = combined_df.sort_values('final_score', ascending=False)
        
        # Add game titles
        combined_df['title'] = combined_df['app_id'].apply(
            lambda x: self.rating_recommender.get_game_title(x)
        )
        
        # Add rank
        combined_df.insert(0, 'rank', range(1, len(combined_df) + 1))
        
        return combined_df
    
    def recommend(self, user_id: str, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Generate final recommendations
        
        Args:
            user_id: User ID
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommendations
        """
        if top_n is None:
            top_n = self.config['parameters']['final_top_n']
        
        recommendations = self.combine_recommendations(user_id)
        
        if recommendations.empty:
            print(f"❌ Could not generate recommendations for user {user_id}")
            return pd.DataFrame(columns=['rank', 'app_id', 'title', 'final_score',
                                        'content_score', 'svd_score', 'rating_score'])
        
        # Get top N recommendations
        final_rec = recommendations.head(top_n).copy()
        
        # Reindex ranks
        final_rec['rank'] = range(1, len(final_rec) + 1)
        
        print(f"\n✅ Successfully generated {len(final_rec)} recommendations")
        
        return final_rec[['rank', 'app_id', 'title', 'final_score',
                         'content_score', 'svd_score', 'rating_score']]
    
    def save_recommendations(self, recommendations: pd.DataFrame, 
                           user_id: str, filename: Optional[str] = None) -> str:
        """Save recommendations to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hybrid_recommendations_{user_id}_{timestamp}.csv"
        
        recommendations.to_csv(filename, index=False, encoding='utf-8')
        print(f"✅ Recommendations saved to: {filename}")
        
        return filename
    
    def batch_recommend(self, user_ids: List[str], output_dir: str = ".") -> Dict:
        """Generate recommendations for multiple users"""
        print(f"\n{'=' * 60}")
        print(f"Batch Recommendation Mode")
        print(f"Processing {len(user_ids)} users")
        print('=' * 60)
        
        all_results = {}
        success_count = 0
        
        for i, user_id in enumerate(user_ids, 1):
            print(f"\n[{i}/{len(user_ids)}] Processing user: {user_id}")
            
            try:
                recommendations = self.recommend(str(user_id))
                
                if not recommendations.empty:
                    all_results[user_id] = recommendations
                    success_count += 1
                    
                    # Save individual file
                    filename = os.path.join(output_dir, 
                                          f"recommendations_{user_id}.csv")
                    recommendations.to_csv(filename, index=False)
                    print(f"  ✓ Saved to {filename}")
                else:
                    print(f"  ✗ No recommendations")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # Save summary file
        if all_results:
            summary_df = pd.concat(all_results.values(), keys=all_results.keys())
            summary_df.reset_index(level=0, inplace=True)
            summary_df.rename(columns={'level_0': 'user_id'}, inplace=True)
            
            summary_file = os.path.join(output_dir, "batch_recommendations_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            
            print(f"\n{'=' * 60}")
            print(f"Batch processing completed!")
            print(f"Successfully processed: {success_count}/{len(user_ids)} users")
            print(f"Summary saved to: {summary_file}")
            print('=' * 60)
        
        return all_results

def main():
    """Main interactive function"""
    print("\n" + "=" * 60)
    print("Improved Hybrid Game Recommendation System")
    print("=" * 60)
    
    # Initialize recommender
    recommender = HybridGameRecommender()
    
    # Load data
    print("\nLoading data...")
    if not recommender.load_data():
        print("❌ Failed to load data")
        return
    
    # Main interaction loop
    while True:
        print("\n" + "=" * 60)
        user_input = input("\nEnter User ID (or 'quit' to exit, 'batch' for batch mode): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'batch':
            # Batch mode
            user_file = input("Enter path to user IDs file (one per line): ").strip()
            
            if os.path.exists(user_file):
                with open(user_file, 'r') as f:
                    user_ids = [line.strip() for line in f if line.strip()]
                
                output_dir = input("Enter output directory (default: current): ").strip()
                if not output_dir:
                    output_dir = "."
                
                recommender.batch_recommend(user_ids, output_dir)
            else:
                print(f"❌ File not found: {user_file}")
            
            continue
        
        if not user_input:
            print("Please enter a valid User ID")
            continue
        
        print(f"\nGenerating recommendations for user: {user_input}")
        
        try:
            # Get recommendations
            recommendations = recommender.recommend(user_input)
            
            if recommendations.empty:
                print("❌ No recommendations generated")
                continue
            
            # Display top recommendations
            print(f"\nTop Recommendations for User {user_input}:")
            print("-" * 100)
            for _, row in recommendations.iterrows():
                title = str(row['title'])[:40]
                print(f"{row['rank']:2d}. {title:<40} | "
                      f"Total: {row['final_score']:.4f} | "
                      f"Content: {row['content_score']:.3f} | "
                      f"SVD: {row['svd_score']:.3f} | "
                      f"Rating: {row['rating_score']:.3f}")
            print("-" * 100)
            
            # Save option
            save_choice = input("\nSave recommendations to CSV? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                recommender.save_recommendations(recommendations, user_input)
            
            # Continue
            continue_choice = input("\nContinue with another user? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("Goodbye!")
                break
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()