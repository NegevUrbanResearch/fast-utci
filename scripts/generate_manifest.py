#!/usr/bin/env python3
"""
Generate analysis manifest dynamically from data folder

This script scans the data/analyses folder for analysis files and generates
a manifest.json file that the viewer can use to list available analyses.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import re


def parse_analysis_filename(filename):
    """
    Parse analysis filename to extract metadata.
    
    Expected format: YYYYMMDD_grid_XXm_single_hHH.json or YYYYMMDD_grid_XXm_fullday.json
    
    Args:
        filename: Analysis filename (without .json extension)
        
    Returns:
        dict with parsed metadata or None if parsing fails
    """
    # Pattern for single hour: 20250815_grid_10m_single_h13
    single_pattern = r'(\d{8})_grid_(\d+)m_single_h(\d+)'
    single_match = re.match(single_pattern, filename)
    
    if single_match:
        date_str, grid_size, hour = single_match.groups()
        return {
            'type': 'single_hour',
            'date': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            'grid_size': int(grid_size),
            'hour': int(hour),
            'title': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} - {hour}:00 ({grid_size}m grid)",
            'description': f"Single hour analysis for {date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} at {hour}:00 with {grid_size}m grid spacing."
        }
    
    # Pattern for full day: 20250815_grid_10m_fullday
    fullday_pattern = r'(\d{8})_grid_(\d+)m_fullday'
    fullday_match = re.match(fullday_pattern, filename)
    
    if fullday_match:
        date_str, grid_size = fullday_match.groups()
        return {
            'type': 'full_day',
            'date': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            'grid_size': int(grid_size),
            'title': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} - Full Day ({grid_size}m grid)",
            'description': f"Full day analysis for {date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} with {grid_size}m grid spacing."
        }
    
    return None


def load_analysis_metadata(analysis_path):
    """
    Load additional metadata from analysis JSON file.
    
    Args:
        analysis_path: Path to analysis JSON file
        
    Returns:
        dict with additional metadata or empty dict if loading fails
    """
    try:
        with open(analysis_path, 'r') as f:
            metadata = json.load(f)
        
        num_positions = metadata.get('num_positions', metadata.get('positions', 0))

        return {
            'analysis_id': metadata.get('analysis_id', ''),
            'analysis_type': metadata.get('analysis_type', ''),
            'date': metadata.get('date', ''),
            'grid_size': metadata.get('grid_size', 0),
            'num_positions': num_positions,
            'positions': num_positions,
            'runtime_seconds': metadata.get('runtime_seconds', 0),
            'utci_min': metadata.get('utci_min', 0),
            'utci_max': metadata.get('utci_max', 0),
            'utci_mean': metadata.get('utci_mean', 0),
            'model_file': metadata.get('model_file', ''),
            'epw_file': metadata.get('epw_file', ''),
            'analysis_period': metadata.get('analysis_period', ''),
            'target_hours': metadata.get('target_hours', [])
        }
    except Exception as e:
        print(f"[WARN] Could not load metadata from {analysis_path}: {e}")
        return {}


def generate_manifest(analyses_dir='data/analyses', output_path='data/analyses/manifest.json'):
    """
    Generate manifest.json from analysis files in the analyses directory.
    
    Args:
        analyses_dir: Directory containing analysis files
        output_path: Path to output manifest.json file
    """
    analyses_dir = Path(analyses_dir)
    output_path = Path(output_path)
    
    if not analyses_dir.exists():
        print(f"[ERROR] Analyses directory not found: {analyses_dir}")
        return False
    
    # Find all analysis JSON files in root and project subdirectories
    analysis_files = []
    
    # Root level files (legacy)
    root_files = [f for f in analyses_dir.glob('*.json') if f.name != 'manifest.json']
    analysis_files.extend([(f, None, None) for f in root_files])
    
    # Project-level files and categories
    for project_dir in analyses_dir.iterdir():
        if not project_dir.is_dir():
            continue
        project = project_dir.name
        
        # Project root files
        for json_file in project_dir.glob('*.json'):
            analysis_files.append((json_file, project, None))
        
        # Category subdirectories
        for category_dir in project_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                for json_file in category_dir.glob('*.json'):
                    analysis_files.append((json_file, project, category))
    
    if not analysis_files:
        print(f"[WARN] No analysis files found in {analyses_dir}")
        # Create empty manifest
        manifest = {
            "analyses": [],
            "last_updated": datetime.now().isoformat()
        }
    else:
        analyses = []
        
        for analysis_file, project, category in sorted(analysis_files):
            filename = analysis_file.stem  # Remove .json extension
            rel_id = analysis_file.relative_to(analyses_dir).with_suffix('').as_posix()
            
            # Parse filename to get basic metadata
            parsed = parse_analysis_filename(filename)
            
            # Load additional metadata from JSON file
            additional_metadata = load_analysis_metadata(analysis_file)
            
            if parsed:
                title = parsed['title']
                analysis_type = parsed['type']
                date = parsed['date']
                grid_size = parsed['grid_size']
                description = parsed['description']
            else:
                title = additional_metadata.get('analysis_id') or filename
                analysis_type = additional_metadata.get('analysis_type', 'unknown')
                date = additional_metadata.get('date', '')
                grid_size = additional_metadata.get('grid_size', 0)
                description = f"Analysis {title}"
            
            analysis_entry = {
                'id': rel_id,
                'title': title,
                'type': analysis_type,
                'date': date,
                'grid_size': grid_size,
                'description': description,
                **additional_metadata
            }
            
            if project:
                analysis_entry['project'] = project
                analysis_entry['path'] = rel_id
            if category:
                analysis_entry['category'] = category
            
            if parsed and parsed['type'] == 'single_hour':
                analysis_entry['hour'] = parsed['hour']
            
            analyses.append(analysis_entry)
        
        manifest = {
            "analyses": analyses,
            "last_updated": datetime.now().isoformat()
        }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write manifest
    try:
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"[OK] Generated manifest with {len(manifest['analyses'])} analyses")
        print(f"[OK] Manifest saved to: {output_path}")
        
        # Print summary
        for analysis in manifest['analyses']:
            print(f"  - {analysis['title']} ({analysis['type']})")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to write manifest: {e}")
        return False


def main():
    """Main function to generate manifest."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate analysis manifest from data folder')
    parser.add_argument('--analyses-dir', default='data/analyses', 
                       help='Directory containing analysis files (default: data/analyses)')
    parser.add_argument('--output', default='data/analyses/manifest.json',
                       help='Output manifest file (default: data/analyses/manifest.json)')
    
    args = parser.parse_args()
    
    success = generate_manifest(args.analyses_dir, args.output)
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
