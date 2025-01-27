#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Any, Dict, Set
from jinja2 import Environment, FileSystemLoader

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

def get_all_members(members: Dict[str, Any]) -> Set[str]:
    """Extract the __all__ list from a module's members if present."""
    if '__all__' in members:
        all_elements = members['__all__'].get('value', {}).get('elements', [])
        return {elem.strip("'") for elem in all_elements}
    return set()

def sort_members(members):
    """Sort members by kind and name."""
    if isinstance(members, dict):
        members = members.values()
    
    def get_sort_key(member):
        name = str(member.get('name', ''))  # Ensure string for sorting
        kind = member.get('kind', '')
        
        # Special case: __version__ should be first
        if name == "__version__":
            return (0, name)
        
        # Order: aliases, functions, modules, others
        if kind == 'alias':
            return (1, name.lower())  # Case-insensitive natural sort
        elif kind == 'function':
            return (2, name.lower())
        elif kind == 'module':
            return (3, name.lower())
        else:
            return (4, name.lower())
    
    return sorted(members, key=get_sort_key)

def is_public(member: Dict[str, Any], module: Dict[str, Any], full_data: Dict[str, Any], is_root: bool = False) -> bool:
    """Check if a member should be included in public documentation."""
    name = member.get('name', '')
    
    # Skip private members except __init__ and __post_init__
    if name.startswith('_') and name not in {'__init__', '__post_init__'}:
        return False
    
    # At root level, only show items from __all__
    if is_root:
        root_all = get_all_members(full_data['validmind'].get('members', {}))
        return name in root_all
    
    # If module has __all__, only include members listed there
    if module and '__all__' in module.get('members', {}):
        module_all = get_all_members(module.get('members', {}))
        return name in module_all
    
    return True

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def process_module(module: Dict[str, Any], path: list, env: Environment, data: Dict[str, Any]):
    """Process a module and its members."""
    module_dir = os.path.join('docs', *path[:-1])
    ensure_dir(module_dir)
    
    # Get module template
    module_template = env.get_template('module.qmd.jinja2')
    
    # Generate module documentation
    output = module_template.render(
        module=module,
        full_data=data,
        is_root=(len(path) <= 1),
        resolve_alias=resolve_alias
    )
    
    # Write output
    filename = f"{path[-1]}.qmd"
    output_path = os.path.join(module_dir, filename)
    with open(output_path, 'w') as f:
        f.write(output)
    
    # Process submodules
    members = module.get('members', {})
    for name, member in members.items():
        if member.get('kind') == 'module':
            if is_public(member, module, data, is_root=len(path) <= 1):
                process_module(member, path + [name], env, data)

def generate_docs(json_path: str, template_dir: str, output_dir: str):
    """Generate documentation from JSON data using templates."""
    # Load JSON data
    with open(json_path) as f:
        data = json.load(f)
    
    # Set up Jinja environment
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    
    # Add custom filters and globals
    env.filters['sort_members'] = sort_members
    env.globals['is_public'] = is_public
    env.globals['resolve_alias'] = resolve_alias
    env.globals['get_all_members'] = get_all_members
    
    # Start processing from root module
    if 'validmind' in data:
        process_module(data['validmind'], ['validmind'], env, data)
    else:
        print("Error: No 'validmind' module found in JSON")

if __name__ == '__main__':
    generate_docs(
        json_path='docs/validmind.json',
        template_dir='docs/templates',
        output_dir='docs'
    )