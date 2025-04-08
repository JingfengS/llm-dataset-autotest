#!/usr/bin/env python3
"""
Script to convert JSON Lines (JSONL) files to properly formatted JSON arrays.

This script reads JSONL files where each line is a separate JSON object,
combines them into a single JSON array, and saves the result to new files
with proper formatting and indentation.
"""

import json
import os
import sys
from typing import List, Dict, Any


def convert_jsonl_to_json(input_file: str, output_file: str) -> bool:
    """
    Convert a JSONL file to a properly formatted JSON array file.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSON file
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    print(f"Processing file: {input_file}")
    
    # Data container for JSON objects
    json_objects: List[Dict[str, Any]] = []
    
    try:
        # Read the JSONL file line by line
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # Parse each line as a JSON object
                    json_obj = json.loads(line)
                    json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num} of {input_file}: {e}")
                    print(f"Problematic line: {line[:100]}...")
                    continue  # Continue with the next line
        
        # Write the collected objects as a properly formatted JSON array
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_objects, f, indent=4)
        
        print(f"Successfully converted {input_file} to {output_file}")
        print(f"Processed {len(json_objects)} JSON objects")
        return True
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return False
    except PermissionError:
        print(f"Error: Permission denied when trying to access or write to files")
        return False
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False


def main():
    """
    Main function to convert JSONL files to JSON arrays.
    """
    # List of files to process
    files_to_process = [
        "coco_pope_adversarial.json",
        "coco_pope_popular.json",
        "coco_pope_random.json"
    ]
    
    success_count = 0
    
    # Process each file
    for file_name in files_to_process:
        if not os.path.exists(file_name):
            print(f"Warning: File '{file_name}' does not exist, skipping.")
            continue
        
        # Create output file name with 'formatted_' prefix
        output_file = f"formatted_{file_name}"
        
        # Convert the file
        if convert_jsonl_to_json(file_name, output_file):
            success_count += 1
    
    # Print summary
    print(f"\nSummary: Successfully processed {success_count} out of {len(files_to_process)} files.")
    
    return 0 if success_count == len(files_to_process) else 1


if __name__ == "__main__":
    sys.exit(main())

