import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import warnings
from configs.cluster.args import define_main_parser
from os.path import join
from os import makedirs
from os.path import join
import logging
import numpy as np
import torch
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from collections import Counter
import tqdm

warnings.filterwarnings('ignore')

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from dataset.prsa import PRSADataset
from dataset.card import TransactionDataset, TransactionDatasetEmbedded
from models.modules import TabFormerBertLM, TabFormerGPT2
from misc.utils import random_split_dataset, get_save_steps_for_epoch
from dataset.datacollator import TransDataCollatorForFineTuning
from testing.test_fraud_utils import test_fraud_model, compute_metrics_fraud
from models.lstm import build_baseline_model, PreTrainedModelWrapper

@dataclass
class EmbeddingAnalysisConfig:
    """Configuration for embedding analysis"""
    pooling_strategy: str = "average"  # "average" or "last"
    apply_pca: bool = True
    pca_variance_threshold: float = 0.95
    min_clusters: int = 2
    max_clusters: int = 40
    random_state: int = 42
    tsne_perplexity: int = 30
    tsne_n_iter: int = 1000
    figsize: Tuple[int, int] = (15, 10)
    output_dir: str = "."

class EmbeddingAnalyzer:
    """
    Comprehensive embedding analysis class with pooling, PCA, clustering, and visualization
    """
    
    def __init__(self, config: EmbeddingAnalysisConfig = None):
        self.config = config or EmbeddingAnalysisConfig()
        self.embeddings = None
        self.labels = None
        self.pooled_embeddings = None
        self.pca_embeddings = None
        self.pca_model = None
        self.scaler = None
        self.optimal_clusters = None
        self.cluster_labels = None
        self.kmeans_model = None
        self.tsne_embeddings = None
        self.pca_2d = None
        
    def load_embeddings_from_dataset(self, dataset) -> None:
        """
        Load embeddings from TransactionDatasetEmbedded
        
        Args:
            dataset: TransactionDatasetEmbedded instance
        """
        print("Loading embeddings from dataset...")
        
        # Extract embeddings and labels
        embeddings_list = []
        labels_list = []
        
        for i in range(len(dataset)):
            embedding, label = dataset[i]
            embeddings_list.append(embedding)
            labels_list.append(label)
        
        # Convert to numpy arrays
        self.embeddings = torch.stack(embeddings_list).numpy()
        self.labels = torch.stack(labels_list).numpy()
        
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Loaded labels shape: {self.labels.shape}")
        
    def load_embeddings_from_arrays(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """
        Load embeddings from numpy arrays
        
        Args:
            embeddings: numpy array of shape (batch_size, seq_len, embedding_dim)
            labels: numpy array of labels
        """
        self.embeddings = embeddings
        self.labels = labels
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Loaded labels shape: {self.labels.shape}")
    
    def apply_pooling(self) -> np.ndarray:
        """
        Apply pooling strategy to embeddings
        
        Returns:
            Pooled embeddings of shape (batch_size, embedding_dim)
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_embeddings_from_dataset first.")
        
        print(f"Applying {self.config.pooling_strategy} pooling...")
        
        if self.config.pooling_strategy == "average":
            # Average pooling across sequence dimension
            self.pooled_embeddings = np.mean(self.embeddings, axis=1)
        elif self.config.pooling_strategy == "last":
            # Take last token embedding
            self.pooled_embeddings = self.embeddings[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
        
        print(f"Pooled embeddings shape: {self.pooled_embeddings.shape}")
        return self.pooled_embeddings
    
    def apply_pca(self) -> np.ndarray:
        """
        Apply PCA to pooled embeddings
        
        Returns:
            PCA-transformed embeddings
        """
        if self.pooled_embeddings is None:
            raise ValueError("No pooled embeddings. Call apply_pooling first.")
        
        if not self.config.apply_pca:
            print("PCA disabled in config")
            self.pca_embeddings = self.pooled_embeddings
            return self.pca_embeddings
        
        print(f"Applying PCA with {self.config.pca_variance_threshold*100}% variance threshold...")
        
        # Standardize features
        self.scaler = StandardScaler()
        scaled_embeddings = self.scaler.fit_transform(self.pooled_embeddings)
        
        # Apply PCA
        self.pca_model = PCA(n_components=self.config.pca_variance_threshold, random_state=self.config.random_state)
        self.pca_embeddings = self.pca_model.fit_transform(scaled_embeddings)
        
        print(f"PCA reduced dimensions from {self.pooled_embeddings.shape[1]} to {self.pca_embeddings.shape[1]}")
        print(f"Explained variance ratio: {self.pca_model.explained_variance_ratio_.sum():.4f}")
        
        return self.pca_embeddings
    
    def find_optimal_clusters(self) -> Dict[str, Any]:
        """
        Find optimal number of clusters using silhouette score and Calinski-Harabasz index
        
        Returns:
            Dictionary with clustering results
        """
        if self.config.apply_pca and self.pca_embeddings is not None:
            data_for_clustering = self.pca_embeddings
        else:
            data_for_clustering = self.pooled_embeddings
        
        if data_for_clustering is None:
            raise ValueError("No embeddings for clustering. Call apply_pooling and optionally apply_pca first.")
        
        print(f"Finding optimal clusters between {self.config.min_clusters} and {self.config.max_clusters}...")
        
        cluster_range = range(self.config.min_clusters, self.config.max_clusters + 1)
        silhouette_scores = []
        calinski_scores = []
        inertias = []
        
        for n_clusters in tqdm.tqdm(cluster_range, desc="Evaluating clusters", unit="clusters"):
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(data_for_clustering)
            
            # Calculate metrics
            sil_score = silhouette_score(data_for_clustering, cluster_labels)
            cal_score = calinski_harabasz_score(data_for_clustering, cluster_labels)
            
            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
            inertias.append(kmeans.inertia_)
        
        # Find optimal number of clusters (highest silhouette score)
        optimal_idx = np.argmax(silhouette_scores)
        self.optimal_clusters = cluster_range[optimal_idx]
        
        print(f"Optimal number of clusters: {self.optimal_clusters}")
        print(f"Best silhouette score: {silhouette_scores[optimal_idx]:.4f}")
        
        # Fit final model with optimal clusters
        self.kmeans_model = KMeans(n_clusters=self.optimal_clusters, random_state=self.config.random_state, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(data_for_clustering)
        
        return {
            'cluster_range': list(cluster_range),
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'inertias': inertias,
            'optimal_clusters': self.optimal_clusters,
            'cluster_labels': self.cluster_labels
        }
    
    def prepare_visualizations(self) -> None:
        """
        Prepare t-SNE and PCA visualizations
        """
        if self.config.apply_pca and self.pca_embeddings is not None:
            data_for_viz = self.pca_embeddings
        else:
            data_for_viz = self.pooled_embeddings
        
        print("Preparing visualizations...")
        
        # t-SNE
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=self.config.tsne_perplexity, 
               max_iter=self.config.tsne_n_iter, random_state=self.config.random_state)
        self.tsne_embeddings = tsne.fit_transform(data_for_viz)
        
        # PCA 2D (separate from the main PCA for visualization)
        print("Computing PCA 2D...")
        if self.config.apply_pca and self.pca_embeddings is not None:
            # data_for_viz is PCA output (lower dim)
            # Fit a new scaler on this data for visualization only
            scaler_viz = StandardScaler()
            scaled_data = scaler_viz.fit_transform(data_for_viz)
        else:
            # data_for_viz is pooled embeddings (original dim)
            if self.scaler is None:
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(data_for_viz)
            else:
                scaled_data = self.scaler.transform(data_for_viz)
        
        pca_2d = PCA(n_components=2, random_state=self.config.random_state)
        self.pca_2d = pca_2d.fit_transform(scaled_data)
        
        print("Visualizations prepared!")
    
    def plot_clustering_metrics(self, results: Dict[str, Any]) -> None:
        """
        Plot clustering evaluation metrics
        
        Args:
            results: Results from find_optimal_clusters
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize)
        
        # Silhouette scores
        axes[0, 0].plot(results['cluster_range'], results['silhouette_scores'], 'bo-')
        axes[0, 0].axvline(x=self.optimal_clusters, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Silhouette Score vs Number of Clusters')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Calinski-Harabasz scores
        axes[0, 1].plot(results['cluster_range'], results['calinski_scores'], 'go-')
        axes[0, 1].axvline(x=self.optimal_clusters, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Calinski-Harabasz Score')
        axes[0, 1].set_title('Calinski-Harabasz Score vs Number of Clusters')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Inertia (Elbow method)
        axes[1, 0].plot(results['cluster_range'], results['inertias'], 'ro-')
        axes[1, 0].axvline(x=self.optimal_clusters, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Inertia')
        axes[1, 0].set_title('Inertia vs Number of Clusters (Elbow Method)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cluster distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        axes[1, 1].bar(unique, counts)
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Cluster Distribution (k={self.optimal_clusters})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_visualizations(self) -> None:
        """
        Plot t-SNE and PCA visualizations
        """
        if self.tsne_embeddings is None or self.pca_2d is None:
            raise ValueError("Visualizations not prepared. Call prepare_visualizations first.")
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize)
        
        # t-SNE with clusters
        scatter1 = axes[0, 0].scatter(self.tsne_embeddings[:, 0], self.tsne_embeddings[:, 1], 
                                    c=self.cluster_labels, cmap='tab10', alpha=0.6)
        axes[0, 0].set_title(f't-SNE Visualization (Clusters, k={self.optimal_clusters})')
        axes[0, 0].set_xlabel('t-SNE 1')
        axes[0, 0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # PCA with clusters
        scatter2 = axes[0, 1].scatter(self.pca_2d[:, 0], self.pca_2d[:, 1], 
                                    c=self.cluster_labels, cmap='tab10', alpha=0.6)
        axes[0, 1].set_title(f'PCA Visualization (Clusters, k={self.optimal_clusters})')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # t-SNE with true labels (if available)
        if self.labels is not None:
            # Handle multi-label case - take argmax or sum
            if len(self.labels.shape) > 1:
                if self.labels.shape[1] > 1:
                    # Multi-label case - convert to single label (fraud vs non-fraud)
                    label_colors = np.any(self.labels, axis=1).astype(int)
                else:
                    label_colors = self.labels.squeeze()
            else:
                label_colors = self.labels
            
            scatter3 = axes[1, 0].scatter(self.tsne_embeddings[:, 0], self.tsne_embeddings[:, 1], 
                                        c=label_colors, cmap='RdYlBu', alpha=0.6)
            axes[1, 0].set_title('t-SNE Visualization (True Labels)')
            axes[1, 0].set_xlabel('t-SNE 1')
            axes[1, 0].set_ylabel('t-SNE 2')
            plt.colorbar(scatter3, ax=axes[1, 0])
            
            # PCA with true labels
            scatter4 = axes[1, 1].scatter(self.pca_2d[:, 0], self.pca_2d[:, 1], 
                                        c=label_colors, cmap='RdYlBu', alpha=0.6)
            axes[1, 1].set_title('PCA Visualization (True Labels)')
            axes[1, 1].set_xlabel('PC1')
            axes[1, 1].set_ylabel('PC2')
            plt.colorbar(scatter4, ax=axes[1, 1])
        else:
            axes[1, 0].text(0.5, 0.5, 'No true labels available', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 1].text(0.5, 0.5, 'No true labels available', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
        
        plt.tight_layout()
        file_name = "cluster_" + ("priorPCA_" if self.config.apply_pca else "") + self.config.pooling_strategy + ".png"
        plt.savefig(join(self.config.output_dir, file_name), dpi=300)  # You can change filename and dpi as needed
        plt.show()
    
    def analyze_cluster_label_correspondence(self) -> pd.DataFrame:
        """
        Analyze correspondence between clusters and true labels
        
        Returns:
            DataFrame with cluster-label correspondence
        """
        if self.labels is None or self.cluster_labels is None:
            print("No labels or cluster labels available")
            return None
        
        # Handle multi-label case
        if len(self.labels.shape) > 1:
            if self.labels.shape[1] > 1:
                # Multi-label case - convert to single label (fraud vs non-fraud)
                true_labels = np.any(self.labels, axis=1).astype(int)
            else:
                true_labels = self.labels.squeeze()
        else:
            true_labels = self.labels
        
        # Create correspondence matrix
        correspondence = pd.crosstab(self.cluster_labels, true_labels, 
                                   rownames=['Cluster'], colnames=['True Label'])
        
        print("Cluster-Label Correspondence:")
        print(correspondence)
        
        # Calculate purity for each cluster
        cluster_purity = []
        for cluster in range(self.optimal_clusters):
            cluster_mask = self.cluster_labels == cluster
            cluster_true_labels = true_labels[cluster_mask]
            if len(cluster_true_labels) > 0:
                purity = np.max(np.bincount(cluster_true_labels)) / len(cluster_true_labels)
                cluster_purity.append(purity)
            else:
                cluster_purity.append(0)
        
        purity_df = pd.DataFrame({
            'Cluster': range(self.optimal_clusters),
            'Purity': cluster_purity,
            'Size': [np.sum(self.cluster_labels == i) for i in range(self.optimal_clusters)]
        })
        
        print("\nCluster Purity:")
        print(purity_df)
        
        return correspondence, purity_df
    
    def run_full_analysis(self, dataset=None, embeddings=None, labels=None) -> Dict[str, Any]:
        """
        Run complete analysis pipeline
        
        Args:
            dataset: TransactionDatasetEmbedded instance (optional)
            embeddings: numpy array of embeddings (optional)
            labels: numpy array of labels (optional)
            
        Returns:
            Dictionary with all analysis results
        """
        print("=" * 60)
        print("STARTING FULL EMBEDDING ANALYSIS")
        print("=" * 60)
        
        # Load data
        if dataset is not None:
            self.load_embeddings_from_dataset(dataset)
        elif embeddings is not None and labels is not None:
            self.load_embeddings_from_arrays(embeddings, labels)
        else:
            raise ValueError("Either dataset or embeddings+labels must be provided")
        
        # Apply pooling
        self.apply_pooling()
        
        # Apply PCA (if enabled)
        if self.config.apply_pca:
            self.apply_pca()
        
        # Find optimal clusters
        clustering_results = self.find_optimal_clusters()
        
        # Prepare visualizations
        self.prepare_visualizations()
        
        # Plot results
        print("\nPlotting clustering metrics...")
        self.plot_clustering_metrics(clustering_results)
        
        print("\nPlotting visualizations...")
        self.plot_visualizations()
        
        # Analyze cluster-label correspondence
        if self.labels is not None:
            correspondence, purity_df = self.analyze_cluster_label_correspondence()
            clustering_results['correspondence'] = correspondence
            clustering_results['purity'] = purity_df
        
        print("=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        return clustering_results
class EnhancedEmbeddingAnalyzer(EmbeddingAnalyzer):
    """
    Enhanced embedding analyzer that can retrieve raw transaction data for clustered samples
    """
    
    def __init__(self, config: EmbeddingAnalysisConfig = None):
        super().__init__(config)
        self.raw_dataset = None
        self.sample_indices = None
        self.cluster_to_samples = None
        self.sample_to_raw_data = None
        self.raw_trans_table = None
        
    def load_embeddings_from_dataset_with_raw_data(self, dataset, raw_dataset=None) -> None:
        """
        Load embeddings from TransactionDatasetEmbedded and keep track of raw data
        
        Args:
            dataset: TransactionDatasetEmbedded instance
            raw_dataset: Original TransactionDataset instance (optional, will try to get from dataset)
        """
        print("Loading embeddings from dataset with raw data tracking...")
        
        # Try to get raw dataset from the embedded dataset
        if raw_dataset is None:
            if hasattr(dataset, 'raw_dataset'):
                raw_dataset = dataset.raw_dataset
            else:
                raise ValueError("raw_dataset must be provided or available in dataset.raw_dataset")
        
        self.raw_dataset = raw_dataset
        self.raw_trans_table = raw_dataset.trans_table  # Access the original transaction table
        
        # Extract embeddings, labels, and sample indices
        embeddings_list = []
        labels_list = []
        sample_indices = []
        
        for i in range(len(dataset)):
            embedding, label = dataset[i]
            embeddings_list.append(embedding)
            labels_list.append(label)
            sample_indices.append(i)
        
        # Convert to numpy arrays
        self.embeddings = torch.stack(embeddings_list).numpy()
        self.labels = torch.stack(labels_list).numpy()
        self.sample_indices = np.array(sample_indices)
        
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Loaded labels shape: {self.labels.shape}")
        print(f"Loaded sample indices: {len(self.sample_indices)}")
        
        # Build mapping from sample index to raw data
        self._build_sample_to_raw_data_mapping()
        
    def _build_sample_to_raw_data_mapping(self) -> None:
        """
        Build mapping from sample indices to raw transaction data
        """
        print("Building sample to raw data mapping...")
        
        self.sample_to_raw_data = {}
        
        # Get the raw transaction data
        trans_data, trans_labels, column_names = self.raw_dataset.user_level_data()
        
        # Create a mapping from sample index to actual transactions
        sample_idx = 0
        for user_idx in range(len(trans_data)):
            user_row = trans_data[user_idx]
            user_labels = trans_labels[user_idx]
            
            # Convert user_row back to transaction format
            # user_row is flattened, so we need to reshape it
            num_fields = len(column_names)
            user_transactions = []
            
            for trans_idx in range(0, len(user_row), num_fields):
                transaction = user_row[trans_idx:trans_idx + num_fields]
                if len(transaction) == num_fields:
                    user_transactions.append(transaction)
            
            # Now create sliding windows just like in prepare_samples
            for jdx in range(0, len(user_transactions) - self.raw_dataset.seq_len + 1, self.raw_dataset.trans_stride):
                window_transactions = user_transactions[jdx:(jdx + self.raw_dataset.seq_len)]
                window_labels = user_labels[jdx:(jdx + self.raw_dataset.seq_len)]
                
                # Create readable transaction data
                readable_transactions = []
                for trans in window_transactions:
                    readable_trans = {}
                    for field_idx, field_name in enumerate(column_names):
                        readable_trans[field_name] = trans[field_idx]
                    readable_transactions.append(readable_trans)
                
                self.sample_to_raw_data[sample_idx] = {
                    'user_idx': user_idx,
                    'window_start': jdx,
                    'transactions': readable_transactions,
                    'labels': window_labels,
                    'column_names': column_names
                }
                sample_idx += 1
                
        print(f"Built mapping for {len(self.sample_to_raw_data)} samples")
    
    def _get_readable_transaction_from_raw_table(self, user_id: int, trans_indices: List[int]) -> List[Dict]:
        """
        Get readable transaction data from the raw transaction table
        
        Args:
            user_id: User ID
            trans_indices: List of transaction indices for this user
            
        Returns:
            List of dictionaries representing readable transactions
        """
        user_data = self.raw_trans_table[self.raw_trans_table['User'] == user_id]
        readable_transactions = []
        
        for trans_idx in trans_indices:
            if trans_idx < len(user_data):
                transaction = user_data.iloc[trans_idx].to_dict()
                readable_transactions.append(transaction)
        
        return readable_transactions
    
    def build_cluster_to_samples_mapping(self) -> None:
        """
        Build mapping from cluster IDs to sample indices
        """
        if self.cluster_labels is None:
            raise ValueError("No cluster labels available. Run clustering first.")
        
        print("Building cluster to samples mapping...")
        
        self.cluster_to_samples = {}
        
        for cluster_id in range(self.optimal_clusters):
            # Get sample indices for this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_sample_indices = self.sample_indices[cluster_mask]
            self.cluster_to_samples[cluster_id] = cluster_sample_indices
            
        print(f"Built mapping for {len(self.cluster_to_samples)} clusters")
        
        # Print cluster sizes
        for cluster_id, sample_indices in self.cluster_to_samples.items():
            print(f"Cluster {cluster_id}: {len(sample_indices)} samples")
    
    def get_raw_data_for_cluster(self, cluster_id: int) -> Dict[str, Any]:
        """
        Get raw transaction data for all samples in a specific cluster
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Dictionary containing raw data for all samples in the cluster
        """
        if self.cluster_to_samples is None:
            self.build_cluster_to_samples_mapping()
        
        if cluster_id not in self.cluster_to_samples:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        cluster_samples = self.cluster_to_samples[cluster_id]
        cluster_raw_data = {}
        
        for sample_idx in cluster_samples:
            if sample_idx in self.sample_to_raw_data:
                cluster_raw_data[sample_idx] = self.sample_to_raw_data[sample_idx]
        
        return cluster_raw_data
    
    def get_readable_transactions_for_cluster(self, cluster_id: int) -> Dict[int, List[Dict]]:
        """
        Get readable transaction data for all samples in a specific cluster
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Dictionary mapping sample_idx to list of readable transaction dictionaries
        """
        cluster_raw_data = self.get_raw_data_for_cluster(cluster_id)
        
        readable_data = {}
        for sample_idx, raw_data in cluster_raw_data.items():
            readable_data[sample_idx] = raw_data['transactions']
        
        return readable_data
    
    def print_cluster_samples(self, cluster_id: int, max_samples: int = 5) -> None:
        """
        Print readable transaction data for samples in a cluster
        
        Args:
            cluster_id: ID of the cluster
            max_samples: Maximum number of samples to print
        """
        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster_id} SAMPLE DATA")
        print(f"{'='*60}")
        
        cluster_raw_data = self.get_raw_data_for_cluster(cluster_id)
        
        sample_count = 0
        for sample_idx, raw_data in cluster_raw_data.items():
            if sample_count >= max_samples:
                print(f"... and {len(cluster_raw_data) - max_samples} more samples")
                break
                
            print(f"\nSample {sample_idx} (User {raw_data['user_idx']}):")
            print(f"Window starts at transaction {raw_data['window_start']}")
            print(f"Labels: {raw_data['labels']}")
            print("Transactions:")
            
            for trans_idx, transaction in enumerate(raw_data['transactions']):
                print(f"  Transaction {trans_idx + 1}:")
                for field, value in transaction.items():
                    print(f"    {field}: {value}")
                print()
            
            sample_count += 1
    
    def analyze_cluster_characteristics(self, cluster_id: int) -> Dict[str, Any]:
        """
        Analyze characteristics of transactions in a specific cluster
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Dictionary with cluster characteristics
        """
        cluster_raw_data = self.get_raw_data_for_cluster(cluster_id)
        
        characteristics = {
            'cluster_id': cluster_id,
            'num_samples': len(cluster_raw_data),
            'sample_indices': list(cluster_raw_data.keys()),
            'field_distributions': {},
            'fraud_stats': {
                'total_transactions': 0,
                'fraud_transactions': 0,
                'fraud_rate': 0.0
            }
        }
        
        # Analyze field distributions and fraud stats
        all_transactions = []
        fraud_count = 0
        total_transactions = 0
        
        for sample_idx, raw_data in cluster_raw_data.items():
            all_transactions.extend(raw_data['transactions'])
            fraud_count += sum(raw_data['labels'])
            total_transactions += len(raw_data['labels'])
        
        characteristics['fraud_stats']['total_transactions'] = total_transactions
        characteristics['fraud_stats']['fraud_transactions'] = fraud_count
        characteristics['fraud_stats']['fraud_rate'] = fraud_count / total_transactions if total_transactions > 0 else 0
        
        # Analyze field distributions
        if all_transactions:
            for field_name in all_transactions[0].keys():
                field_values = [trans[field_name] for trans in all_transactions]
                field_counts = Counter(field_values)
                characteristics['field_distributions'][field_name] = dict(field_counts.most_common(5))
        
        return characteristics
    
    def compare_clusters(self, cluster_ids: List[int]) -> None:
        """
        Compare characteristics between multiple clusters
        
        Args:
            cluster_ids: List of cluster IDs to compare
        """
        print(f"\n{'='*60}")
        print(f"CLUSTER COMPARISON")
        print(f"{'='*60}")
        
        for cluster_id in cluster_ids:
            characteristics = self.analyze_cluster_characteristics(cluster_id)
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Samples: {characteristics['num_samples']}")
            print(f"  Total transactions: {characteristics['fraud_stats']['total_transactions']}")
            print(f"  Fraud rate: {characteristics['fraud_stats']['fraud_rate']:.2%}")
            
            # Show most common values for key fields
            key_fields = ['Use Chip', 'Merchant State', 'MCC', 'Errors?']
            for field in key_fields:
                if field in characteristics['field_distributions']:
                    top_values = list(characteristics['field_distributions'][field].items())[:2]
                    print(f"  Top {field}: {top_values}")
    
    def export_cluster_data(self, cluster_id: int, output_path: str) -> None:
        """
        Export raw data for a cluster to a file
        
        Args:
            cluster_id: ID of the cluster
            output_path: Path to save the data
        """
        cluster_raw_data = self.get_raw_data_for_cluster(cluster_id)
        
        # Convert to a more serializable format
        export_data = {
            'cluster_id': int(cluster_id),
            'num_samples': len(cluster_raw_data),
            'samples': {}
        }
        
        for sample_idx, raw_data in cluster_raw_data.items():
            # Convert numpy types to native Python types
            sample_key = str(int(sample_idx))  # Convert numpy int64 to string
            export_data['samples'][sample_key] = {
                'user_idx': int(raw_data['user_idx']),
                'window_start': int(raw_data['window_start']),
                'transactions': self._convert_transactions_to_json_serializable(raw_data['transactions']),
                'labels': [int(label) for label in raw_data['labels']],  # Convert numpy array to list
                'column_names': raw_data['column_names']
            }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported cluster {cluster_id} data to {output_path}")
    
    def _convert_transactions_to_json_serializable(self, transactions: List[Dict]) -> List[Dict]:
        """
        Convert transactions to JSON serializable format
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            JSON serializable list of transaction dictionaries
        """
        serializable_transactions = []
        for trans in transactions:
            serializable_trans = {}
            for key, value in trans.items():
                # Convert numpy types to native Python types
                if isinstance(value, np.integer):
                    serializable_trans[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_trans[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_trans[key] = value.tolist()
                else:
                    serializable_trans[key] = value
            serializable_transactions.append(serializable_trans)
        return serializable_transactions
    
    def get_cluster_dataframe(self, cluster_id: int) -> pd.DataFrame:
        """
        Get cluster data as a pandas DataFrame for easier analysis
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            DataFrame with all transactions from the cluster
        """
        cluster_raw_data = self.get_raw_data_for_cluster(cluster_id)
        
        all_rows = []
        for sample_idx, raw_data in cluster_raw_data.items():
            for trans_idx, transaction in enumerate(raw_data['transactions']):
                row = transaction.copy()
                row['sample_idx'] = sample_idx
                row['user_idx'] = raw_data['user_idx']
                row['transaction_idx'] = trans_idx
                row['is_fraud'] = raw_data['labels'][trans_idx]
                all_rows.append(row)
        
        return pd.DataFrame(all_rows)
    
    def run_full_analysis_with_raw_data(self, dataset, raw_dataset=None) -> Dict[str, Any]:
        """
        Run complete analysis pipeline with raw data tracking
        
        Args:
            dataset: TransactionDatasetEmbedded instance
            raw_dataset: Original TransactionDataset instance (optional)
            
        Returns:
            Dictionary with all analysis results including raw data mappings
        """
        print("=" * 60)
        print("STARTING FULL EMBEDDING ANALYSIS WITH RAW DATA")
        print("=" * 60)
        
        # Load data with raw data tracking
        self.load_embeddings_from_dataset_with_raw_data(dataset, raw_dataset)
        
        # Apply pooling
        self.apply_pooling()
        
        # Apply PCA (if enabled)
        if self.config.apply_pca:
            self.apply_pca()
        
        # Find optimal clusters
        clustering_results = self.find_optimal_clusters()
        
        # Build cluster to samples mapping
        self.build_cluster_to_samples_mapping()
        
        # Prepare visualizations
        self.prepare_visualizations()
        
        # Plot results
        print("\nPlotting clustering metrics...")
        self.plot_clustering_metrics(clustering_results)
        
        print("\nPlotting visualizations...")
        self.plot_visualizations()
        
        # Analyze cluster-label correspondence
        if self.labels is not None:
            correspondence, purity_df = self.analyze_cluster_label_correspondence()
            clustering_results['correspondence'] = correspondence
            clustering_results['purity'] = purity_df
        
        # Add raw data mappings to results
        clustering_results['cluster_to_samples'] = self.cluster_to_samples
        clustering_results['sample_to_raw_data'] = self.sample_to_raw_data
        
        # Print sample data for each cluster
        print("\nSample data for each cluster:")
        for cluster_id in range(self.optimal_clusters):
            self.print_cluster_samples(cluster_id, max_samples=2)
        
        # Compare clusters
        self.compare_clusters(list(range(min(3, self.optimal_clusters))))
        
        print("=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        return clustering_results
# Usage example function
def main(args):
    """
    Example of how to use the enhanced analyzer
    """
    # Your existing setup code...
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create original dataset
    raw_dataset = TransactionDataset(
        root=args.data_root,
        fname=args.data_fname,
        seq_len=args.seq_len,
        fextension=args.data_extension,
        vocab_dir=args.output_dir,
        nrows=args.nrows,
        user_ids=args.user_ids,
        mlm=args.mlm,
        cached=args.cached,
        stride=args.stride,
        flatten=args.flatten,
        return_labels=True,
        skip_user=args.skip_user
    )

    # Create pretrained model
    vocab = raw_dataset.vocab
    custom_special_tokens = vocab.get_special_tokens()
    
    pretrained_model = TabFormerBertLM(
        custom_special_tokens,
        vocab=vocab,
        field_ce=args.field_ce,
        flatten=args.flatten,
        ncols=raw_dataset.ncols,
        field_hidden_size=args.field_hs, 
        return_embeddings=True
    )

    # Load model checkpoint
    model_path = join(args.path_to_checkpoint, f'checkpoint-{args.checkpoint}')
    pretrained_model.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))

    # Create embedded dataset
    embedded_dataset = TransactionDatasetEmbedded(
        pretrained_model=pretrained_model,
        raw_dataset=raw_dataset,
        batch_size=args.batch_size, 
        force_recompute=False
    )

    # Configuration
    config = EmbeddingAnalysisConfig(
        pooling_strategy=args.pooling_strategy,
        apply_pca=True,
        pca_variance_threshold=0.95,
        min_clusters=2,
        max_clusters=4,
        random_state=42,
        output_dir=args.output_dir
    )

    # Create enhanced analyzer
    analyzer = EnhancedEmbeddingAnalyzer(config)
    
    # Run analysis with raw data tracking
    results = analyzer.run_full_analysis_with_raw_data(embedded_dataset, raw_dataset)
    
    # Example of accessing raw data for specific clusters
    print("\nExample: Getting raw data for cluster 0")
    cluster_0_data = analyzer.get_raw_data_for_cluster(0)
    print(f"Cluster 0 has {len(cluster_0_data)} samples")
    
    # Get as DataFrame for easier analysis
    cluster_0_df = analyzer.get_cluster_dataframe(0)
    print(f"Cluster 0 DataFrame shape: {cluster_0_df.shape}")
    print(f"Fraud rate in cluster 0: {cluster_0_df['is_fraud'].mean():.2%}")
    
    # Export cluster data
    analyzer.export_cluster_data(0, join(args.output_dir, "cluster_0_data.json"))
    
    return results
    
if __name__ == "__main__":
    parser = define_main_parser()
    opts = parser.parse_args()

    opts.log_dir = join(opts.output_dir, "logs")
    makedirs(opts.output_dir, exist_ok=True)
    makedirs(opts.log_dir, exist_ok=True)

    main(opts)


    