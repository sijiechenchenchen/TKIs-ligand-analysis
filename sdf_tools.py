"""
SDF Tools for molecular analysis and featurization.

This module provides utilities for reading molecular data from SDF and MOL2 files,
generating molecular fingerprints, and calculating similarity metrics.
"""

import os
from typing import List, Union, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from sklearn.metrics.pairwise import cosine_similarity


def read_sdf_mol(sdf_file: str) -> List[Chem.Mol]:
    """
    Read molecules from an SDF file.
    
    Args:
        sdf_file (str): Path to the SDF file
        
    Returns:
        List[Chem.Mol]: List of RDKit molecule objects
        
    Raises:
        FileNotFoundError: If the SDF file doesn't exist
        ValueError: If no valid molecules found in the file
    """
    if not os.path.exists(sdf_file):
        raise FileNotFoundError(f"SDF file not found: {sdf_file}")
    
    molecules = []
    supplier = Chem.SDMolSupplier(sdf_file)
    
    for mol in supplier:
        if mol is not None:
            molecules.append(mol)
    
    if not molecules:
        raise ValueError(f"No valid molecules found in {sdf_file}")
    
    return molecules


def read_mol2_mol(mol2_directory: str) -> List[Chem.Mol]:
    """
    Read molecules from MOL2 files in a directory.
    
    Args:
        mol2_directory (str): Path to directory containing MOL2 files
        
    Returns:
        List[Chem.Mol]: List of RDKit molecule objects
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        ValueError: If no valid molecules found
    """
    if not os.path.exists(mol2_directory):
        raise FileNotFoundError(f"MOL2 directory not found: {mol2_directory}")
    
    molecules = []
    
    for filename in os.listdir(mol2_directory):
        if filename.endswith('.mol2'):
            filepath = os.path.join(mol2_directory, filename)
            try:
                mol = Chem.MolFromMol2File(filepath)
                if mol is not None:
                    molecules.append(mol)
            except Exception as e:
                print(f"Warning: Could not read {filepath}: {e}")
    
    if not molecules:
        raise ValueError(f"No valid molecules found in {mol2_directory}")
    
    return molecules


def molsfeaturizer(molecules: List[Chem.Mol], 
                   fingerprint_type: str = 'Morgan',
                   radius: int = 2, 
                   nbits: int = 2048) -> np.ndarray:
    """
    Generate molecular fingerprints from a list of molecules.
    
    Args:
        molecules (List[Chem.Mol]): List of RDKit molecule objects
        fingerprint_type (str): Type of fingerprint ('Morgan', 'MACCS', 'RDKit')
        radius (int): Radius for Morgan fingerprints
        nbits (int): Number of bits for fingerprints
        
    Returns:
        np.ndarray: Array of molecular fingerprints
        
    Raises:
        ValueError: If fingerprint_type is not supported
    """
    if fingerprint_type not in ['Morgan', 'MACCS', 'RDKit']:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")
    
    fingerprints = []
    
    for mol in molecules:
        if mol is None:
            continue
            
        if fingerprint_type == 'Morgan':
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        elif fingerprint_type == 'MACCS':
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        elif fingerprint_type == 'RDKit':
            fp = Chem.RDKFingerprint(mol, fpSize=nbits)
        
        # Convert to numpy array
        arr = np.zeros((nbits if fingerprint_type != 'MACCS' else 167,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
    
    return np.array(fingerprints)


def get_tanimoto_similarity(sdf_file1: str, 
                           sdf_file2: str, 
                           feature: str = 'Morgan') -> np.ndarray:
    """
    Calculate Tanimoto similarity matrix between molecules from two SDF files.
    
    Args:
        sdf_file1 (str): Path to first SDF file
        sdf_file2 (str): Path to second SDF file
        feature (str): Fingerprint type to use
        
    Returns:
        np.ndarray: Tanimoto similarity matrix
    """
    mols1 = read_sdf_mol(sdf_file1)
    mols2 = read_sdf_mol(sdf_file2)
    
    fps1 = molsfeaturizer(mols1, fingerprint_type=feature)
    fps2 = molsfeaturizer(mols2, fingerprint_type=feature)
    
    # Calculate Tanimoto similarity
    similarity_matrix = np.zeros((len(fps1), len(fps2)))
    
    for i, fp1 in enumerate(fps1):
        for j, fp2 in enumerate(fps2):
            # Tanimoto similarity = intersection / union
            intersection = np.sum(np.minimum(fp1, fp2))
            union = np.sum(np.maximum(fp1, fp2))
            similarity_matrix[i, j] = intersection / union if union > 0 else 0
    
    return similarity_matrix


def get_cosine_similarity(sdf_file1: str, 
                         sdf_file2: str, 
                         feature: str = 'Morgan') -> np.ndarray:
    """
    Calculate cosine similarity matrix between molecules from two SDF files.
    
    Args:
        sdf_file1 (str): Path to first SDF file
        sdf_file2 (str): Path to second SDF file
        feature (str): Fingerprint type to use
        
    Returns:
        np.ndarray: Cosine similarity matrix
    """
    mols1 = read_sdf_mol(sdf_file1)
    mols2 = read_sdf_mol(sdf_file2)
    
    fps1 = molsfeaturizer(mols1, fingerprint_type=feature)
    fps2 = molsfeaturizer(mols2, fingerprint_type=feature)
    
    return cosine_similarity(fps1, fps2)