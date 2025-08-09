"""
t-SNE Analysis for TKI Ligand Datasets

This script performs t-distributed Stochastic Neighbor Embedding (t-SNE) analysis
to visualize high-dimensional molecular data in 2D space, revealing cluster patterns
and relationships between different molecular datasets.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import sdf_tools as st

sns.set_theme()


class TSNEAnalyzer:
    """t-SNE analysis for molecular datasets."""
    
    def __init__(self, 
                 pca_components: int = 20,
                 tsne_components: int = 2,
                 perplexity: float = 30.0,
                 n_iter: int = 1000,
                 standardize: bool = True):
        """
        Initialize t-SNE analyzer.
        
        Args:
            pca_components: Number of PCA components for dimensionality reduction
            tsne_components: Number of t-SNE components (usually 2 for visualization)
            perplexity: t-SNE perplexity parameter
            n_iter: Number of t-SNE iterations
            standardize: Whether to standardize features before analysis
        """
        self.pca_components = pca_components
        self.tsne_components = tsne_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.standardize = standardize
        
        self.pca = PCA(n_components=pca_components)
        self.scaler = StandardScaler() if standardize else None
        
        self.datasets = {}
        self.pca_results = {}
        self.tsne_results = {}
    
    def load_dataset(self, name: str, filepath: str, dataset_type: str = 'sdf') -> None:
        """
        Load a molecular dataset.
        
        Args:
            name: Name identifier for the dataset
            filepath: Path to the dataset file/directory
            dataset_type: Type of dataset ('sdf' or 'mol2')
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        try:
            if dataset_type == 'sdf':
                molecules = st.read_sdf_mol(filepath)
            elif dataset_type == 'mol2':
                molecules = st.read_mol2_mol(filepath)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
            fingerprints = st.molsfeaturizer(molecules)
            self.datasets[name] = {
                'molecules': molecules,
                'fingerprints': fingerprints,
                'filepath': filepath,
                'type': dataset_type
            }
            print(f"Loaded {len(molecules)} molecules from {name}")
            
        except Exception as e:
            print(f"Error loading {name}: {e}")
            raise
    
    def evaluate_pca_components(self, dataset_name: str, max_components: int = 50) -> List[Tuple[int, float]]:
        """
        Evaluate explained variance for different numbers of PCA components.
        
        Args:
            dataset_name: Name of the dataset to evaluate
            max_components: Maximum number of components to test
            
        Returns:
            List of (n_components, explained_variance) tuples
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        fingerprints = self.datasets[dataset_name]['fingerprints']
        if self.standardize:
            fingerprints = self.scaler.fit_transform(fingerprints)
        
        results = []
        for n_comp in tqdm(range(2, min(max_components + 1, fingerprints.shape[1])),
                          desc="Evaluating PCA components"):
            pca = PCA(n_components=n_comp)
            pca.fit(fingerprints)
            var_explained = np.sum(pca.explained_variance_ratio_)
            results.append((n_comp, var_explained))
        
        return results
    
    def fit_pca(self, fit_dataset: str) -> None:
        """
        Fit PCA model on a specific dataset.
        
        Args:
            fit_dataset: Name of the dataset to fit PCA on
        """
        if fit_dataset not in self.datasets:
            raise ValueError(f"Dataset {fit_dataset} not found")
        
        fingerprints = self.datasets[fit_dataset]['fingerprints']
        
        if self.standardize:
            fingerprints = self.scaler.fit_transform(fingerprints)
        
        self.pca.fit(fingerprints)
        print(f"PCA fitted on {fit_dataset}")
        print(f"Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.3f}")
    
    def transform_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply PCA and t-SNE transformations to a dataset.
        
        Args:
            dataset_name: Name of the dataset to transform
            
        Returns:
            Tuple of (PCA-transformed data, t-SNE-transformed data)
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        fingerprints = self.datasets[dataset_name]['fingerprints']
        
        # Apply PCA transformation
        if self.standardize:
            fingerprints = self.scaler.transform(fingerprints)
        
        pca_data = self.pca.transform(fingerprints)
        self.pca_results[dataset_name] = pca_data
        
        # Apply t-SNE transformation
        tsne = TSNE(n_components=self.tsne_components,
                   perplexity=self.perplexity,
                   n_iter=self.n_iter,
                   random_state=42,
                   verbose=1)
        
        tsne_data = tsne.fit_transform(pca_data)
        self.tsne_results[dataset_name] = tsne_data
        
        return pca_data, tsne_data
    
    def transform_ensemble(self, dataset_names: List[str]) -> np.ndarray:
        """
        Create ensemble t-SNE embedding from multiple datasets.
        
        Args:
            dataset_names: List of dataset names to include in ensemble
            
        Returns:
            Ensemble t-SNE embedding
        """
        # Combine PCA results
        pca_ensemble = []
        dataset_indices = {}
        current_idx = 0
        
        for name in dataset_names:
            if name not in self.pca_results:
                raise ValueError(f"PCA results not found for {name}. Run transform_dataset first.")
            
            pca_data = self.pca_results[name]
            pca_ensemble.append(pca_data)
            dataset_indices[name] = (current_idx, current_idx + len(pca_data))
            current_idx += len(pca_data)
        
        pca_combined = np.vstack(pca_ensemble)
        
        # Apply t-SNE to ensemble
        tsne = TSNE(n_components=self.tsne_components,
                   perplexity=self.perplexity,
                   n_iter=self.n_iter,
                   random_state=42,
                   verbose=1)
        
        tsne_ensemble = tsne.fit_transform(pca_combined)
        
        # Store ensemble results
        self.tsne_results['ensemble'] = tsne_ensemble
        self.dataset_indices = dataset_indices
        
        return tsne_ensemble
    
    def plot_tsne_comparison(self, 
                            datasets_config: Dict[str, Dict],
                            use_ensemble: bool = True,
                            output_file: str = "tsne_analysis.png",
                            figsize: Tuple[int, int] = (12, 8),
                            dpi: int = 300) -> None:
        """
        Create t-SNE comparison plot.
        
        Args:
            datasets_config: Configuration for datasets with colors and labels
            use_ensemble: Whether to use ensemble embedding or individual embeddings
            output_file: Output filename for the plot
            figsize: Figure size
            dpi: Image resolution
        """
        plt.figure(figsize=figsize)
        
        if use_ensemble and 'ensemble' in self.tsne_results:
            tsne_ensemble = self.tsne_results['ensemble']
            
            for dataset_name, config in datasets_config.items():
                if dataset_name not in self.dataset_indices:
                    print(f"Warning: {dataset_name} not found in ensemble")
                    continue
                
                start_idx, end_idx = self.dataset_indices[dataset_name]
                data_subset = tsne_ensemble[start_idx:end_idx]
                
                plt.scatter(data_subset[:, 0], data_subset[:, 1],
                           color=config.get('color', 'blue'),
                           alpha=config.get('alpha', 0.7),
                           label=config.get('label', dataset_name),
                           s=config.get('size', 50),
                           edgecolors=config.get('edgecolor', 'none'),
                           linewidth=config.get('linewidth', 0.5))
        else:
            # Use individual embeddings
            for dataset_name, config in datasets_config.items():
                if dataset_name not in self.tsne_results:
                    print(f"Warning: {dataset_name} not found in t-SNE results")
                    continue
                
                tsne_data = self.tsne_results[dataset_name]
                plt.scatter(tsne_data[:, 0], tsne_data[:, 1],
                           color=config.get('color', 'blue'),
                           alpha=config.get('alpha', 0.7),
                           label=config.get('label', dataset_name),
                           s=config.get('size', 50),
                           edgecolors=config.get('edgecolor', 'none'),
                           linewidth=config.get('linewidth', 0.5))
        
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE Analysis of Molecular Datasets')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, transparent=True, bbox_inches='tight')
        plt.close()
        print(f"t-SNE plot saved to {output_file}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='t-SNE Analysis of TKI Ligand Datasets')
    parser.add_argument('--active', default='sdfs/active_less_than10.sdf',
                       help='Path to active compounds SDF file')
    parser.add_argument('--inactive', default='sdfs/inactive.sdf',
                       help='Path to inactive compounds SDF file')
    parser.add_argument('--allo-all', default='ASD_Release_201909_3D',
                       help='Path to all allosteric sites directory')
    parser.add_argument('--drugs-all', default='sdfs/all_drugs.sdf',
                       help='Path to all drugs SDF file')
    parser.add_argument('--output', default='tsne_analysis.png',
                       help='Output filename for t-SNE plot')
    parser.add_argument('--pca-components', type=int, default=20,
                       help='Number of PCA components for preprocessing')
    parser.add_argument('--perplexity', type=float, default=30.0,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--n-iter', type=int, default=1000,
                       help='Number of t-SNE iterations')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble t-SNE embedding')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TSNEAnalyzer(
        pca_components=args.pca_components,
        perplexity=args.perplexity,
        n_iter=args.n_iter
    )
    
    # Load datasets
    datasets_to_load = {
        'active': (args.active, 'sdf'),
        'allo_all': (args.allo_all, 'mol2'),
        'drugs_all': (args.drugs_all, 'sdf')
    }
    
    for name, (filepath, dtype) in datasets_to_load.items():
        try:
            analyzer.load_dataset(name, filepath, dtype)
        except Exception as e:
            print(f"Warning: Could not load {name}: {e}")
    
    # Fit PCA on largest dataset
    try:
        analyzer.fit_pca('allo_all')
    except Exception as e:
        print(f"Error fitting PCA: {e}")
        return
    
    # Transform datasets
    loaded_datasets = list(analyzer.datasets.keys())
    for dataset_name in loaded_datasets:
        try:
            analyzer.transform_dataset(dataset_name)
        except Exception as e:
            print(f"Error transforming {dataset_name}: {e}")
    
    # Create ensemble if requested
    if args.ensemble:
        try:
            analyzer.transform_ensemble(loaded_datasets)
        except Exception as e:
            print(f"Error creating ensemble: {e}")
    
    # Plot configuration
    datasets_config = {
        'active': {'color': 'red', 'alpha': 0.8, 'label': 'Active Compounds', 'size': 60},
        'allo_all': {'color': 'purple', 'alpha': 0.3, 'label': 'All Allosteric Sites', 'size': 30},
        'drugs_all': {'color': 'grey', 'alpha': 0.3, 'label': 'All Drugs', 'size': 30}
    }
    
    # Filter config for loaded datasets
    datasets_config = {k: v for k, v in datasets_config.items() if k in loaded_datasets}
    
    # Create plot
    analyzer.plot_tsne_comparison(datasets_config, args.ensemble, args.output)


if __name__ == '__main__':
    main()