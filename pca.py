"""
PCA Analysis for TKI Ligand Datasets

This script performs Principal Component Analysis (PCA) on different molecular datasets
to visualize chemical space and identify clustering patterns among active/inactive compounds
and allosteric modulators.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sdf_tools as st

sns.set_theme()


class PCAAnalyzer:
    """Principal Component Analysis for molecular datasets."""
    
    def __init__(self, n_components: int = 2, standardize: bool = True):
        """
        Initialize PCA analyzer.
        
        Args:
            n_components: Number of PCA components
            standardize: Whether to standardize features before PCA
        """
        self.n_components = n_components
        self.standardize = standardize
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler() if standardize else None
        self.datasets = {}
        self.pca_results = {}
    
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
    
    def fit_transform_dataset(self, dataset_name: str, fit_on: Optional[str] = None) -> np.ndarray:
        """
        Apply PCA transformation to a dataset.
        
        Args:
            dataset_name: Name of the dataset to transform
            fit_on: Dataset name to fit PCA on (if None, fit on current dataset)
            
        Returns:
            PCA-transformed data
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        fingerprints = self.datasets[dataset_name]['fingerprints']
        
        if fit_on is None or fit_on == dataset_name:
            # Fit and transform
            if self.standardize:
                fingerprints = self.scaler.fit_transform(fingerprints)
            pca_result = self.pca.fit_transform(fingerprints)
        else:
            # Transform using existing fit
            if self.standardize:
                fingerprints = self.scaler.transform(fingerprints)
            pca_result = self.pca.transform(fingerprints)
        
        self.pca_results[dataset_name] = pca_result
        return pca_result
    
    def plot_pca_comparison(self, 
                           datasets_config: Dict[str, Dict],
                           output_file: str = "pca_analysis.png",
                           figsize: Tuple[int, int] = (12, 8),
                           dpi: int = 300) -> None:
        """
        Create PCA comparison plot.
        
        Args:
            datasets_config: Configuration for datasets with colors and labels
            output_file: Output filename for the plot
            figsize: Figure size
            dpi: Image resolution
        """
        plt.figure(figsize=figsize)
        
        for dataset_name, config in datasets_config.items():
            if dataset_name not in self.pca_results:
                print(f"Warning: {dataset_name} not found in PCA results")
                continue
            
            pca_data = self.pca_results[dataset_name]
            plt.scatter(pca_data[:, 0], pca_data[:, 1],
                       color=config.get('color', 'blue'),
                       alpha=config.get('alpha', 0.7),
                       label=config.get('label', dataset_name),
                       s=config.get('size', 50),
                       edgecolors=config.get('edgecolor', 'none'),
                       linewidth=config.get('linewidth', 0.5))
        
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA Analysis of Molecular Datasets')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, transparent=True, bbox_inches='tight')
        plt.close()
        print(f"PCA plot saved to {output_file}")
    
    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance ratios for each component."""
        return self.pca.explained_variance_ratio_


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='PCA Analysis of TKI Ligand Datasets')
    parser.add_argument('--active', default='sdfs/active_less_than10.sdf',
                       help='Path to active compounds SDF file')
    parser.add_argument('--inactive', default='sdfs/inactive.sdf',
                       help='Path to inactive compounds SDF file')
    parser.add_argument('--allo-druggable', default='ASD_druggable',
                       help='Path to druggable allosteric sites directory')
    parser.add_argument('--allo-all', default='ASD_Release_201909_3D',
                       help='Path to all allosteric sites directory')
    parser.add_argument('--output', default='pca_analysis.png',
                       help='Output filename for PCA plot')
    parser.add_argument('--components', type=int, default=2,
                       help='Number of PCA components')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PCAAnalyzer(n_components=args.components)
    
    # Load datasets
    try:
        analyzer.load_dataset('active', args.active, 'sdf')
        analyzer.load_dataset('inactive', args.inactive, 'sdf')
        analyzer.load_dataset('allo_druggable', args.allo_druggable, 'mol2')
        analyzer.load_dataset('allo_all', args.allo_all, 'mol2')
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Perform PCA (fit on the largest dataset)
    fit_dataset = 'allo_all'  # Largest dataset
    try:
        analyzer.fit_transform_dataset(fit_dataset)
        analyzer.fit_transform_dataset('active', fit_on=fit_dataset)
        analyzer.fit_transform_dataset('inactive', fit_on=fit_dataset)
        analyzer.fit_transform_dataset('allo_druggable', fit_on=fit_dataset)
    except Exception as e:
        print(f"Error performing PCA: {e}")
        return
    
    # Plot configuration
    datasets_config = {
        'allo_all': {'color': 'grey', 'alpha': 0.3, 'label': 'All Allosteric Sites', 'size': 30},
        'active': {'color': 'red', 'alpha': 0.8, 'label': 'Active Compounds', 'size': 60},
        'inactive': {'color': 'green', 'alpha': 0.8, 'label': 'Inactive Compounds', 'size': 60},
        'allo_druggable': {'color': 'blue', 'alpha': 0.8, 'label': 'Druggable Allosteric Sites', 'size': 50}
    }
    
    # Create plot
    analyzer.plot_pca_comparison(datasets_config, args.output)
    
    # Print explained variance
    variance_ratios = analyzer.get_explained_variance()
    print(f"Explained variance ratios: {variance_ratios}")
    print(f"Total variance explained: {np.sum(variance_ratios):.1%}")


if __name__ == '__main__':
    main()