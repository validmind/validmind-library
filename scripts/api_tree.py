#!/usr/bin/env python3
import json
import sys
from typing import Any, Dict

# Define which kinds of members to show
SHOW_KINDS = {'module', 'class', 'function'}  # Base kinds to show

# Modules to skip
SKIP_MODULES = {'utils'}  # Add any other modules you want to skip

def resolve_alias(member: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an alias to its target member."""
    if member.get('kind') == 'alias' and member.get('target_path'):
        path_parts = member['target_path'].split('.')
        current = data
        for part in path_parts[1:]:  # Skip the first part (validmind)
            if part in current.get('members', {}):
                current = current['members'][part]
            else:
                return member
        return current
    return member

def print_tree(data: Dict[str, Any], prefix: str = "", is_last: bool = True, is_root: bool = False, full_data: Dict[str, Any] = None) -> None:
    """Print a tree view of the API structure."""
    members = data.get('members', {})
    
    # Filter items to only show desired kinds
    items = [(name, member) for name, member in sorted(members.items())
            if member and (member.get('kind') in SHOW_KINDS or member.get('kind') == 'alias')]
    
    for i, (name, member) in enumerate(items):
        if name == '__all__':
            continue
            
        # Skip private members and utils module
        if name.startswith('_') or name in SKIP_MODULES:
            continue

        is_last_item = i == len(items) - 1
        
        # Resolve alias before getting kind and members
        resolved_member = resolve_alias(member, full_data)
        kind = resolved_member.get('kind') if resolved_member else 'unknown'
        
        # Create the branch symbol
        branch = "└── " if is_last_item else "├── "
        
        # Print the current item
        print(f"{prefix}{branch}{name} ({kind})")
        
        # Recursively print children for modules and classes
        if (kind == 'module' or kind == 'class') and 'members' in resolved_member:
            new_prefix = prefix + ("    " if is_last_item else "│   ")
            print_tree(resolved_member, new_prefix, is_last_item, is_root=False, full_data=full_data)

def main():
    """Main function to process the JSON file."""
    if len(sys.argv) != 2:
        print("Usage: python api_tree.py <path_to_json>")
        sys.exit(1)

    json_path = sys.argv[1]
    
    try:
        with open(json_path) as f:
            data = json.load(f)
        print("\nValidMind Python API:")
        # Start with the 'validmind' module
        if 'validmind' in data:
            print_tree(data['validmind'], is_root=True, full_data=data)
        else:
            print("Error: No 'validmind' module found in JSON")
        
    except FileNotFoundError:
        print(f"Error: File {json_path} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: {json_path} is not a valid JSON file")
        sys.exit(1)

if __name__ == "__main__":
    main()