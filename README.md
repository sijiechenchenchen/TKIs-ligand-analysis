# TKI Ligand Analysis

A comprehensive toolkit for analyzing Tyrosine Kinase Inhibitor (TKI) ligands using molecular fingerprints, dimensionality reduction techniques, and similarity analysis for drug discovery research.

## 🔬 Overview

This project provides tools for analyzing molecular datasets containing TKI compounds, allosteric modulators, and drug libraries. It focuses on:

- **Molecular Fingerprint Generation**: Converting molecular structures into numerical features
- **Dimensionality Reduction**: Using PCA and t-SNE to visualize chemical space
- **Similarity Analysis**: Computing Tanimoto and cosine similarity between compounds
- **Clustering Analysis**: Identifying patterns in active/inactive compounds

## 📁 Project Structure

```
TKIs-ligand-analysis/
├── README.md                    # This file
├── sdf_tools.py                # Core molecular analysis utilities
├── pca.py                      # Principal Component Analysis
├── similarity.py               # Molecular similarity analysis
├── tsne.py                     # t-SNE visualization
├── ASD_Release_201909_3D/      # Allosteric database (MOL2 files)
├── sdfs/                       # SDF datasets
│   ├── active_less_than10.sdf  # Active TKI compounds
│   ├── inactive.sdf            # Inactive compounds
│   ├── all_drugs.sdf           # Complete drug library
│   └── mol2/                   # Additional MOL2 structures
└── ASD_Release_201909_3D/      # Allosteric site database
    └── get_filelist.py         # Utility for file listing
```

## 🚀 Features

### Core Utilities (`sdf_tools.py`)
- **Multi-format Support**: Read SDF and MOL2 molecular files
- **Fingerprint Generation**: Morgan, MACCS, and RDKit fingerprints
- **Similarity Metrics**: Tanimoto and cosine similarity calculations
- **Error Handling**: Robust file processing with informative error messages

### Analysis Modules

#### 1. Principal Component Analysis (`pca.py`)
- Dimensionality reduction for molecular datasets
- Standardization and scaling options
- Configurable number of components
- Interactive visualization with explained variance
- Command-line interface for batch processing

#### 2. Similarity Analysis (`similarity.py`)
- Tanimoto and cosine similarity matrices
- Heatmap visualization
- Statistical analysis of similarity distributions
- Support for different fingerprint types
- Batch processing capabilities

#### 3. t-SNE Analysis (`tsne.py`)
- Non-linear dimensionality reduction
- PCA preprocessing for high-dimensional data
- Ensemble embedding for multiple datasets
- Configurable perplexity and iteration parameters
- Component evaluation utilities

## 📋 Requirements

### Core Dependencies
```bash
pip install rdkit
pip install scikit-learn
pip install numpy
pip install matplotlib
pip install seaborn
pip install tqdm
```

### System Requirements
- Python 3.7+
- RDKit (for molecular structure handling)
- 4GB+ RAM (depending on dataset size)

## ⚡ Quick Start

### 1. Basic PCA Analysis
```bash
python pca.py --active sdfs/active_less_than10.sdf \
              --inactive sdfs/inactive.sdf \
              --output pca_results.png
```

### 2. Similarity Analysis
```bash
python similarity.py --active sdfs/active_less_than10.sdf \
                     --inactive sdfs/inactive.sdf \
                     --fingerprint Morgan \
                     --similarity-type tanimoto
```

### 3. t-SNE Visualization
```bash
python tsne.py --active sdfs/active_less_than10.sdf \
               --allo-all ASD_Release_201909_3D \
               --ensemble \
               --perplexity 50
```

## 📊 Usage Examples

### Loading and Processing Molecules

```python
import sdf_tools as st

# Load molecules from SDF file
molecules = st.read_sdf_mol('sdfs/active_less_than10.sdf')

# Generate Morgan fingerprints
fingerprints = st.molsfeaturizer(molecules, fingerprint_type='Morgan')

# Calculate similarity
similarity_matrix = st.get_tanimoto_similarity('active.sdf', 'inactive.sdf')
```

### PCA Analysis

```python
from pca import PCAAnalyzer

# Initialize analyzer
analyzer = PCAAnalyzer(n_components=2, standardize=True)

# Load datasets
analyzer.load_dataset('active', 'sdfs/active_less_than10.sdf')
analyzer.load_dataset('inactive', 'sdfs/inactive.sdf')

# Perform analysis
analyzer.fit_transform_dataset('active')
analyzer.plot_pca_comparison(datasets_config, output_file='results.png')
```

### Similarity Analysis

```python
from similarity import SimilarityAnalyzer

# Initialize analyzer
analyzer = SimilarityAnalyzer(fingerprint_type='Morgan')

# Load and analyze
analyzer.load_dataset('active', 'sdfs/active_less_than10.sdf')
analyzer.calculate_tanimoto_similarity('active', 'active')
analyzer.plot_similarity_heatmaps(['active_vs_active_tanimoto'])
```

## 🎛️ Configuration Options

### Fingerprint Types
- **Morgan**: Circular fingerprints based on atomic environments
- **MACCS**: 166-bit structural keys for drug-like molecules
- **RDKit**: Path-based fingerprints

### PCA Parameters
- `n_components`: Number of principal components (default: 2)
- `standardize`: Feature standardization (default: True)

### t-SNE Parameters
- `perplexity`: Balance between local and global structure (default: 30.0)
- `n_iter`: Number of optimization iterations (default: 1000)
- `pca_components`: PCA preprocessing components (default: 20)

### Similarity Metrics
- **Tanimoto**: Intersection over union for binary fingerprints
- **Cosine**: Angular similarity for continuous features

## 📈 Output Files

### Generated Visualizations
- `pca_analysis.png`: PCA scatter plot with explained variance
- `similarity_analysis.png`: Similarity heatmaps
- `tsne_analysis.png`: t-SNE embedding visualization

### Data Formats
- **Input**: SDF, MOL2 molecular structure files
- **Output**: PNG visualizations, similarity matrices

## 🔧 Advanced Usage

### Custom Fingerprint Parameters
```python
# Custom Morgan fingerprints
fingerprints = st.molsfeaturizer(
    molecules, 
    fingerprint_type='Morgan',
    radius=3,          # Extended radius
    nbits=4096         # Higher resolution
)
```

### Batch Processing
```bash
# Process multiple datasets
for dataset in active inactive druggable; do
    python pca.py --active sdfs/${dataset}.sdf --output pca_${dataset}.png
done
```

### Performance Optimization
- Use PCA preprocessing for large t-SNE analyses
- Standardize features for better clustering
- Adjust perplexity based on dataset size (typically 5-50)

## 🐛 Troubleshooting

### Common Issues

#### 1. RDKit Import Error
```bash
# Install via conda (recommended)
conda install -c conda-forge rdkit

# Or via pip (may require additional setup)
pip install rdkit-pypi
```

#### 2. Memory Issues with Large Datasets
- Reduce `pca_components` parameter
- Use batch processing for similarity calculations
- Consider downsampling large datasets

#### 3. t-SNE Convergence Issues
- Increase `n_iter` parameter
- Adjust `perplexity` (try 5-50 range)
- Ensure PCA preprocessing is used

### Error Messages
- `FileNotFoundError`: Check file paths and permissions
- `ValueError: No valid molecules`: Verify SDF/MOL2 file integrity
- `Memory Error`: Reduce dataset size or increase system RAM

## 🧪 Datasets

### Included Datasets
- **Active Compounds**: TKI compounds with IC50 < 10μM
- **Inactive Compounds**: Compounds with low TKI activity
- **Allosteric Sites**: Database of allosteric modulator binding sites
- **Drug Library**: Comprehensive pharmaceutical compound collection

### Data Sources
- **ASD**: Allosteric Database (allosteric sites)
- **ChEMBL**: Bioactivity data for TKI compounds
- **ZINC**: Chemical database for drug-like molecules

## 📚 Scientific Background

### Molecular Fingerprints
Molecular fingerprints encode structural information as bit vectors or feature vectors, enabling computational analysis of chemical similarity and biological activity relationships.

### Dimensionality Reduction
- **PCA**: Linear technique preserving global variance structure
- **t-SNE**: Non-linear method emphasizing local neighborhood relationships

### Applications
- **Drug Discovery**: Identify novel TKI scaffolds
- **SAR Analysis**: Structure-activity relationship studies
- **Chemical Space**: Visualization of molecular diversity

## 🤝 Contributing

### Development Setup
```bash
git clone <repository-url>
cd TKIs-ligand-analysis
pip install -r requirements.txt
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function parameters
- Document functions with docstrings
- Include error handling for file operations

### Testing
```bash
python -m pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 References

1. Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754.

2. van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.

3. Bajusz, D., Rácz, A., & Héberger, K. (2015). Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations? *Journal of Cheminformatics*, 7(1), 20.

4. Huang, Z., et al. (2019). ASD v3.0: Unraveling allosteric regulation with structural mechanisms and biological networks. *Nucleic Acids Research*, 47(D1), D308-D314.

## 📬 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub or contact the development team.

---

**Note**: This toolkit is designed for research purposes. Ensure proper validation of results for any commercial or clinical applications.