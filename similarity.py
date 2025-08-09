"""
Molecular Similarity Analysis for TKI Ligand Datasets

This script calculates and visualizes similarity matrices between different molecular datasets
using Tanimoto and cosine similarity metrics based on molecular fingerprints.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

import sdf_tools as st

sns.set_theme()


class SimilarityAnalyzer:
    """Molecular similarity analysis for drug discovery datasets."""
    
    def __init__(self, fingerprint_type: str = 'Morgan'):
        """
        Initialize similarity analyzer.
        
        Args:
            fingerprint_type: Type of molecular fingerprint to use
        """
        self.fingerprint_type = fingerprint_type
        self.datasets = {}
        self.similarity_matrices = {}
    
    def load_dataset(self, name: str, filepath: str) -> None:
        """
        Load a molecular dataset.
        
        Args:
            name: Name identifier for the dataset
            filepath: Path to the SDF file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        try:
            molecules = st.read_sdf_mol(filepath)
            fingerprints = st.molsfeaturizer(molecules, fingerprint_type=self.fingerprint_type)
            
            self.datasets[name] = {
                'molecules': molecules,
                'fingerprints': fingerprints,
                'filepath': filepath
            }
            print(f"Loaded {len(molecules)} molecules from {name}")
            
        except Exception as e:
            print(f"Error loading {name}: {e}")
            raise
    
    def calculate_tanimoto_similarity(self, dataset1: str, dataset2: str) -> np.ndarray:
        """
        Calculate Tanimoto similarity matrix between two datasets.
        
        Args:
            dataset1: Name of first dataset
            dataset2: Name of second dataset
            
        Returns:
            Tanimoto similarity matrix
        """
        if dataset1 not in self.datasets or dataset2 not in self.datasets:
            raise ValueError("Both datasets must be loaded first")
        
        fps1 = self.datasets[dataset1]['fingerprints']
        fps2 = self.datasets[dataset2]['fingerprints']
        
        similarity_matrix = np.zeros((len(fps1), len(fps2)))
        
        for i, fp1 in enumerate(fps1):
            for j, fp2 in enumerate(fps2):
                intersection = np.sum(np.minimum(fp1, fp2))
                union = np.sum(np.maximum(fp1, fp2))
                similarity_matrix[i, j] = intersection / union if union > 0 else 0
        
        matrix_name = f"{dataset1}_vs_{dataset2}_tanimoto"
        self.similarity_matrices[matrix_name] = similarity_matrix
        
        return similarity_matrix
    
    def calculate_cosine_similarity(self, dataset1: str, dataset2: str) -> np.ndarray:
        """
        Calculate cosine similarity matrix between two datasets.
        
        Args:
            dataset1: Name of first dataset
            dataset2: Name of second dataset
            
        Returns:
            Cosine similarity matrix
        """
        if dataset1 not in self.datasets or dataset2 not in self.datasets:
            raise ValueError("Both datasets must be loaded first")
        
        fps1 = self.datasets[dataset1]['fingerprints']
        fps2 = self.datasets[dataset2]['fingerprints']
        
        similarity_matrix = cosine_similarity(fps1, fps2)
        
        matrix_name = f"{dataset1}_vs_{dataset2}_cosine"
        self.similarity_matrices[matrix_name] = similarity_matrix
        
        return similarity_matrix
    
    def plot_similarity_heatmaps(self, 
                                matrices_to_plot: List[str],
                                titles: Optional[List[str]] = None,
                                output_file: str = "similarity_analysis.png",
                                figsize: Tuple[int, int] = (20, 6),
                                dpi: int = 300) -> None:
        """
        Create similarity heatmap plots.
        
        Args:
            matrices_to_plot: List of similarity matrix names to plot
            titles: Optional list of titles for each subplot
            output_file: Output filename for the plot
            figsize: Figure size
            dpi: Image resolution
        """
        n_plots = len(matrices_to_plot)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharex=False, sharey=False)
        
        if n_plots == 1:
            axes = [axes]
        
        for i, matrix_name in enumerate(matrices_to_plot):
            if matrix_name not in self.similarity_matrices:
                print(f"Warning: {matrix_name} not found in similarity matrices")
                continue
            
            matrix = self.similarity_matrices[matrix_name]
            title = titles[i] if titles and i < len(titles) else matrix_name
            
            sns.heatmap(matrix, 
                       ax=axes[i], 
                       cmap="YlGnBu", 
                       vmin=0, 
                       vmax=1,
                       cbar=True,
                       square=True)
            axes[i].set_title(title)
            axes[i].set_xlabel('Molecules')
            axes[i].set_ylabel('Molecules')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, transparent=True, bbox_inches='tight')
        plt.close()
        print(f"Similarity heatmaps saved to {output_file}")
    
    def get_similarity_statistics(self, matrix_name: str) -> Dict[str, float]:
        """
        Calculate statistics for a similarity matrix.
        
        Args:
            matrix_name: Name of the similarity matrix
            
        Returns:
            Dictionary with similarity statistics
        """
        if matrix_name not in self.similarity_matrices:
            raise ValueError(f"Matrix {matrix_name} not found")
        
        matrix = self.similarity_matrices[matrix_name]
        
        return {
            'mean': np.mean(matrix),
            'median': np.median(matrix),
            'std': np.std(matrix),
            'min': np.min(matrix),
            'max': np.max(matrix),
            'q25': np.percentile(matrix, 25),
            'q75': np.percentile(matrix, 75)
        }


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Molecular Similarity Analysis')
    parser.add_argument('--active', default='sdfs/active_less_than10.sdf',
                       help='Path to active compounds SDF file')
    parser.add_argument('--inactive', default='sdfs/inactive.sdf',
                       help='Path to inactive compounds SDF file')
    parser.add_argument('--fingerprint', default='Morgan',
                       choices=['Morgan', 'MACCS', 'RDKit'],
                       help='Type of molecular fingerprint')
    parser.add_argument('--output', default='similarity_analysis.png',
                       help='Output filename for similarity plots')
    parser.add_argument('--similarity-type', default='tanimoto',
                       choices=['tanimoto', 'cosine', 'both'],
                       help='Type of similarity metric to calculate')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SimilarityAnalyzer(fingerprint_type=args.fingerprint)
    
    # Load datasets
    try:
        analyzer.load_dataset('active', args.active)
        analyzer.load_dataset('inactive', args.inactive)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Calculate similarities
    matrices_to_plot = []
    titles = []
    
    try:
        if args.similarity_type in ['tanimoto', 'both']:
            # Active vs Inactive
            analyzer.calculate_tanimoto_similarity('active', 'inactive')
            matrices_to_plot.append('active_vs_inactive_tanimoto')
            titles.append('Active vs Inactive\n(Tanimoto)')
            
            # Active vs Active
            analyzer.calculate_tanimoto_similarity('active', 'active')
            matrices_to_plot.append('active_vs_active_tanimoto')
            titles.append('Active vs Active\n(Tanimoto)')
            
            # Inactive vs Inactive
            analyzer.calculate_tanimoto_similarity('inactive', 'inactive')
            matrices_to_plot.append('inactive_vs_inactive_tanimoto')
            titles.append('Inactive vs Inactive\n(Tanimoto)')
        
        if args.similarity_type in ['cosine', 'both']:
            # Active vs Inactive
            analyzer.calculate_cosine_similarity('active', 'inactive')
            matrices_to_plot.append('active_vs_inactive_cosine')
            titles.append('Active vs Inactive\n(Cosine)')
            
            # Active vs Active
            analyzer.calculate_cosine_similarity('active', 'active')
            matrices_to_plot.append('active_vs_active_cosine')
            titles.append('Active vs Active\n(Cosine)')
            
            # Inactive vs Inactive
            analyzer.calculate_cosine_similarity('inactive', 'inactive')
            matrices_to_plot.append('inactive_vs_inactive_cosine')
            titles.append('Inactive vs Inactive\n(Cosine)')
    
    except Exception as e:
        print(f"Error calculating similarities: {e}")
        return
    
    # Create plots
    analyzer.plot_similarity_heatmaps(matrices_to_plot, titles, args.output)
    
    # Print statistics
    print("\nSimilarity Matrix Statistics:")
    print("=" * 50)
    for matrix_name in matrices_to_plot:
        stats = analyzer.get_similarity_statistics(matrix_name)
        print(f"\n{matrix_name}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.3f}")


if __name__ == '__main__':
    main()