"""
Semantic Search Engine for Vector Memory
Provides semantic similarity search capabilities
"""

import asyncio
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
import json
import logging
from datetime import datetime

from utils.observability.logging import get_logger
from .vector_memory import MemoryEntry

logger = get_logger(__name__)


class SemanticSearchEngine:
    """
    Semantic search engine using simple vector similarity
    Falls back to lightweight implementations when advanced libraries unavailable
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.vocabulary: Dict[str, int] = {}
        self.word_index = 0
        
        logger.info(f"Initialized semantic search engine with {embedding_dim}D embeddings")
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using simple bag-of-words approach"""
        try:
            # Tokenize and clean text
            words = text.lower().split()
            words = [word.strip('.,!?;:()[]{}"\'-') for word in words if len(word) > 2]
            
            # Create embedding vector
            embedding = np.zeros(self.embedding_dim)
            
            # Simple bag-of-words with position weighting
            for i, word in enumerate(words):
                # Add word to vocabulary if new
                if word not in self.vocabulary:
                    self.vocabulary[word] = self.word_index
                    self.word_index += 1
                
                # Get word index and add to embedding
                word_idx = self.vocabulary[word] % self.embedding_dim
                
                # Weight by inverse position (earlier words matter more)
                position_weight = 1.0 / (i + 1)
                embedding[word_idx] += position_weight
                
                # Add some randomness for better distribution
                if len(words) > 1:
                    neighbor_idx = (word_idx + 1) % self.embedding_dim
                    embedding[neighbor_idx] += position_weight * 0.3
            
            # Normalize the embedding
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return random normalized vector as fallback
            embedding = np.random.random(self.embedding_dim)
            return embedding / np.linalg.norm(embedding)
    
    async def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure vectors are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure similarity is between 0 and 1
            return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def search_similar(self, query_embedding: np.ndarray, 
                           memory_embeddings: List[Tuple[MemoryEntry, np.ndarray]], 
                           k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        """Find most similar memories to query"""
        try:
            similarities = []
            
            for memory, embedding in memory_embeddings:
                similarity = await self.calculate_similarity(query_embedding, embedding)
                similarities.append((memory, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results
            return similarities[:k]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def cluster_memories(self, memory_embeddings: List[Tuple[MemoryEntry, np.ndarray]], 
                              num_clusters: int = 5) -> Dict[int, List[MemoryEntry]]:
        """Simple clustering of memories based on embeddings"""
        try:
            if not memory_embeddings or num_clusters <= 0:
                return {}
            
            # Extract embeddings
            embeddings = np.array([emb for _, emb in memory_embeddings])
            memories = [mem for mem, _ in memory_embeddings]
            
            if len(embeddings) <= num_clusters:
                # Not enough memories to cluster meaningfully
                clusters = {}
                for i, memory in enumerate(memories):
                    clusters[i] = [memory]
                return clusters
            
            # Simple k-means clustering
            # Initialize centroids randomly
            centroids = embeddings[np.random.choice(len(embeddings), num_clusters, replace=False)]
            
            # Iterate to find stable clusters
            for iteration in range(10):  # Max 10 iterations
                # Assign each point to closest centroid
                clusters = {i: [] for i in range(num_clusters)}
                assignments = []
                
                for i, embedding in enumerate(embeddings):
                    # Find closest centroid
                    distances = [np.linalg.norm(embedding - centroid) for centroid in centroids]
                    closest_cluster = np.argmin(distances)
                    clusters[closest_cluster].append(memories[i])
                    assignments.append(closest_cluster)
                
                # Update centroids
                new_centroids = []
                for cluster_id in range(num_clusters):
                    cluster_embeddings = [embeddings[i] for i, assignment in enumerate(assignments) if assignment == cluster_id]
                    if cluster_embeddings:
                        new_centroid = np.mean(cluster_embeddings, axis=0)
                        new_centroids.append(new_centroid)
                    else:
                        # Keep old centroid if cluster is empty
                        new_centroids.append(centroids[cluster_id])
                
                # Check for convergence
                centroid_movement = np.mean([np.linalg.norm(old - new) for old, new in zip(centroids, new_centroids)])
                centroids = np.array(new_centroids)
                
                if centroid_movement < 0.01:  # Convergence threshold
                    break
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering memories: {e}")
            return {}
    
    async def find_outliers(self, memory_embeddings: List[Tuple[MemoryEntry, np.ndarray]], 
                           threshold: float = 0.3) -> List[MemoryEntry]:
        """Find memories that are outliers (dissimilar to others)"""
        try:
            if len(memory_embeddings) < 3:
                return []
            
            outliers = []
            
            for i, (memory, embedding) in enumerate(memory_embeddings):
                # Calculate average similarity to all other memories
                similarities = []
                
                for j, (_, other_embedding) in enumerate(memory_embeddings):
                    if i != j:
                        similarity = await self.calculate_similarity(embedding, other_embedding)
                        similarities.append(similarity)
                
                avg_similarity = np.mean(similarities) if similarities else 0.0
                
                # If average similarity is below threshold, it's an outlier
                if avg_similarity < threshold:
                    outliers.append(memory)
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error finding outliers: {e}")
            return []
    
    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search engine"""
        try:
            return {
                'embedding_dimension': self.embedding_dim,
                'vocabulary_size': len(self.vocabulary),
                'total_words_indexed': self.word_index,
                'search_engine_type': 'simple_bag_of_words'
            }
        except Exception as e:
            logger.error(f"Error getting search statistics: {e}")
            return {}
    
    async def save_vocabulary(self, filepath: str) -> None:
        """Save vocabulary to file"""
        try:
            vocab_data = {
                'vocabulary': self.vocabulary,
                'word_index': self.word_index,
                'embedding_dim': self.embedding_dim
            }
            
            with open(filepath, 'w') as f:
                json.dump(vocab_data, f, indent=2)
            
            logger.info(f"Saved vocabulary to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving vocabulary: {e}")
    
    async def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary from file"""
        try:
            with open(filepath, 'r') as f:
                vocab_data = json.load(f)
            
            self.vocabulary = vocab_data.get('vocabulary', {})
            self.word_index = vocab_data.get('word_index', 0)
            self.embedding_dim = vocab_data.get('embedding_dim', 384)
            
            logger.info(f"Loaded vocabulary from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")