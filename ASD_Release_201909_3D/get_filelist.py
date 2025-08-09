"""
File List Utility for ASD Release

This utility processes the filelist file and generates a clean list of MOL2 files
in the ASD (Allosteric Database) release directory.
"""

import os
from pathlib import Path
from typing import List


def process_filelist(input_file: str = "filelist", output_file: str = "files") -> List[str]:
    """
    Process the filelist file and create a clean file list.
    
    Args:
        input_file: Path to the input filelist file
        output_file: Path to the output files list
        
    Returns:
        List of processed filenames
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return []
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"Warning: {input_file} is empty")
            return []
        
        # Split the first line by spaces to get individual files
        files = lines[0].strip().split(' ')
        
        # Filter out empty strings and ensure .mol2 extension
        processed_files = []
        for file in files:
            file = file.strip()
            if file and not file.endswith('.mol2'):
                file += '.mol2'
            if file:
                processed_files.append(file)
        
        # Write processed files to output
        with open(output_file, 'w') as f_write:
            for file in processed_files:
                f_write.write(file + '\n')
        
        print(f"Processed {len(processed_files)} files from {input_file}")
        print(f"Output written to {output_file}")
        
        return processed_files
    
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return []


def main():
    """Main function to process filelist."""
    processed_files = process_filelist()
    
    # Additional validation - check if files exist
    if processed_files:
        existing_files = []
        for filename in processed_files:
            if os.path.exists(filename):
                existing_files.append(filename)
        
        print(f"Found {len(existing_files)} existing files out of {len(processed_files)} total")
        
        if len(existing_files) != len(processed_files):
            missing = len(processed_files) - len(existing_files)
            print(f"Warning: {missing} files are missing from the directory")


if __name__ == '__main__':
    main()