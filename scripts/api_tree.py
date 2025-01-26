#!/usr/bin/env python3
import json
import sys
from typing import Any, Dict

# Define which kinds of members to show
SHOW_KINDS = {'module', 'class', 'function', 'method'}

# Modules to skip
SKIP_MODULES = {'utils'}

def resolve_alias(member: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an alias to its target member."""
    if member.get('kind') == 'alias' and member.get('target_path'):
        path_parts = member['target_path'].split('.')
        
        # Skip resolution if it's not in our codebase
        if path_parts[0] != 'validmind':
            return member
            
        current = data[path_parts[0]]  # Start at validmind
        for part in path_parts[1:]:
            if part in current.get('members', {}):
                current = current['members'][part]
            else:
                return member
        return current
    return member

def get_all_members(members: Dict[str, Any]) -> set:
    """Extract the __all__ list from a module's members if present."""
    if '__all__' in members:
        all_elements = members['__all__'].get('value', {}).get('elements', [])
        return {elem.strip("'") for elem in all_elements}
    return set()

def has_docstring(member: Dict[str, Any]) -> bool:
    """Check if a member has a docstring."""
    docstring = member.get('docstring', {})
    if isinstance(docstring, dict):
        return bool(docstring.get('value'))
    return bool(docstring)

def get_sort_key(item):
    """Get sort key for an item tuple (name, member)."""
    name, member = item
    
    # Special case: __version__ should be first
    if name == "__version__":
        return (0, name)
    
    kind = member.get('kind', '')
    
    # Special case: validation modules in tests should be last
    if kind == 'module' and name.endswith('_validation'):
        return (4, name)
    
    # Then:
    # 1. Aliases (top-level exports)
    # 2. Functions
    # 3. Modules and other items
    if kind == 'alias':
        return (1, name)
    elif kind == 'function':
        return (2, name)
    elif kind == 'module':
        return (3, name)
    else:
        return (3, name)

def resolve_alias_target(data: Dict[str, Any], target_path: str, visited_paths=None) -> Dict[str, Any]:
    """Resolve an alias target to its actual member definition."""
    if not target_path or not target_path.startswith('validmind.'):
        return None
        
    # Prevent infinite recursion with visited paths set
    if visited_paths is None:
        visited_paths = set()
    if target_path in visited_paths:
        return None
    visited_paths.add(target_path)
    
    # Split path into parts and remove 'validmind' prefix
    parts = target_path.split('.')
    if parts[0] != 'validmind':
        return None
    parts = parts[1:]  # Remove 'validmind' prefix
        
    # Start at root
    current = data
    for i, part in enumerate(parts):
        if part not in current.get('members', {}):
            return None
        current = current['members'][part]
    
    # If we found another alias, recursively resolve it
    if current.get('kind') == 'alias':
        next_target = current.get('target_path')
        if next_target:
            return resolve_alias_target(data, next_target, visited_paths)
    
    return current

def find_class_def(members: dict, class_name: str) -> dict:
    """Recursively search for a class definition in the members dictionary."""
    if not isinstance(members, dict):
        return None
        
    # Check if this is the class we're looking for
    if members.get('kind') == 'class' and members.get('name') == class_name:
        return members
        
    # Search in members
    if 'members' in members:
        for member in members['members'].values():
            result = find_class_def(member, class_name)
            if result:
                return result
    return None

def print_tree(data: Dict[str, Any], prefix: str = "", is_last: bool = True, is_root: bool = False, full_data: Dict[str, Any] = None) -> None:
    """Print a tree view of the API structure."""
    members = data.get('members', {})
    name = data.get('name', '')
    
    # Special handling for vm_models
    if name == 'vm_models':
        docstring = '*' if data.get('docstring') else ''
        print(f"{prefix}{'└── ' if is_last else '├── '}{name} (module){docstring}")
        
        # Get vm_models members
        vm_members = full_data['validmind']['members']['vm_models']['members']
        
        # Get the __all__ list
        if '__all__' in vm_members:
            all_value = vm_members['__all__'].get('value', {})
            if 'elements' in all_value:
                all_list = [e.strip("'") for e in all_value['elements']]
                
                new_prefix = prefix + ('    ' if is_last else '│   ')
                # Print items in __all__ order
                for i, item in enumerate(all_list):
                    is_last_item = i == len(all_list) - 1  # Check if this is the last item
                    if item in vm_members:
                        member = vm_members[item]
                        docstring = '*' if member.get('docstring') else ''
                        if member.get('kind') == 'alias':
                            target = member.get('target_path', '')
                            branch = '└── ' if is_last_item else '├── '  # Use correct branch for last item
                            print(f"{new_prefix}{branch}{item} (alias) -> {target}{docstring}")
                            
                            # Get the class name from the target path
                            class_name = target.split('.')[-1]
                            
                            # Navigate through the JSON structure
                            current = vm_members
                            for part in target.split('.')[2:]:  # Skip 'validmind' and 'vm_models'
                                if part in current:
                                    current = current[part]
                                elif 'members' in current and part in current['members']:
                                    current = current['members'][part]
                                else:
                                    break
                            
                            # If we found a class definition, print its methods
                            if current.get('kind') == 'class':
                                class_prefix = new_prefix + ('    ' if is_last_item else '│   ')  # Adjust prefix based on last item
                                methods = [(name, method) for name, method in current.get('members', {}).items()
                                          if method.get('kind') == 'function' and not name.startswith('_')]
                                
                                # Sort methods
                                methods.sort()
                                
                                # Print methods with proper last item handling
                                for j, (method_name, method) in enumerate(methods):
                                    is_last_method = j == len(methods) - 1
                                    method_docstring = '*' if method.get('docstring') else ''
                                    branch = '└── ' if is_last_method else '├── '
                                    print(f"{class_prefix}{branch}{method_name} (function){method_docstring}")
        return
    
    # Get root __all__ if we're at root level
    root_all = set()
    if is_root and 'validmind' in full_data:
        root_module = full_data['validmind']
        root_all = set(get_all_members(root_module.get('members', {})))
    
    # Get module-level __all__ members
    all_members = set(get_all_members(members))
    
    # Filter and sort items
    items = []
    for name, member in sorted(members.items()):
        if not member or member.get('kind') is None:
            continue
            
        # Skip private members and utils module
        if name.startswith('_') and name not in {'__init__', '__post_init__'}:
            continue
            
        if name in SKIP_MODULES:
            continue
            
        # At root level, only show items from __all__
        if is_root:
            if name not in root_all:
                continue
            items.append((name, member))
            continue
            
        # Handle aliases
        if member.get('kind') == 'alias':
            # Keep aliases only if they're in __all__
            if name not in all_members:
                continue
            # Skip external library imports
            target_path = member.get('target_path', '')
            if not target_path.startswith('validmind.'):
                continue
            
        # Show only modules, classes, functions in __all__ (if present)
        kind = member.get('kind', 'unknown')
        if kind == 'module' or kind == 'class' or (
            kind in SHOW_KINDS and (name in all_members or not all_members)
        ):
            items.append((name, member))
    
    # Sort items using custom sort key
    items.sort(key=get_sort_key)
    
    # Print items
    for i, (name, member) in enumerate(items):
        is_last_item = i == len(items) - 1
        kind = member.get('kind', 'unknown')
        
        # Create branch symbol
        branch = "└── " if is_last_item else "├── "
        
        # For aliases, check both alias and target docstrings
        docstring_marker = ""
        if kind == 'alias' and full_data:
            # Check if either the alias or its target has a docstring
            has_alias_docstring = has_docstring(member)
            target = member.get('target_path', '')
            target = resolve_alias_target(full_data['validmind'], target)
            has_target_docstring = target is not None and has_docstring(target)
            
            if has_alias_docstring or has_target_docstring:
                docstring_marker = "*"
        else:
            docstring_marker = "*" if has_docstring(member) else ""
        
        # For aliases, show the target path
        if kind == 'alias':
            target = f" -> {member.get('target_path', 'unknown')}"
        else:
            target = ""
        
        # Print the item
        print(f"{prefix}{branch}{name} ({kind}){target}{docstring_marker}")
        
        # Recursively print children for modules and classes
        if kind in {'module', 'class'} and 'members' in member:
            new_prefix = prefix + ("    " if is_last_item else "│   ")
            print_tree(member, new_prefix, is_last_item, is_root=False, full_data=full_data)

def main():
    """Main function to process the JSON file."""
    if len(sys.argv) != 2:
        print("Usage: python api_tree.py <path_to_json>")
        sys.exit(1)

    json_path = sys.argv[1]
    
    try:
        with open(json_path) as f:
            data = json.load(f)
        print("validmind (* = docstring)")
        # Pass the full data to print_tree for resolving aliases
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