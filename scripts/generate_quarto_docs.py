#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Any, Dict, Set, List, Optional
from jinja2 import Environment, FileSystemLoader
import mdformat
from docstring_parser import parse, Style
from glob import glob
import subprocess
import re
import inspect

# Add at module level
_alias_cache = {}  # Cache for resolved aliases

def resolve_alias(member: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an alias to its target member."""
    if member.get('kind') == 'alias' and member.get('target_path'):
        target_path = member['target_path']
        
        # Check cache first
        if target_path in _alias_cache:
            return _alias_cache[target_path]
            
        path_parts = target_path.split('.')
        # Skip resolution if it's not in our codebase
        if path_parts[0] != 'validmind':
            return member
        
        # Skip known modules that aren't in the documentation
        if len(path_parts) > 1 and path_parts[1] in ['ai', 'internal']:
            # Silently return the member without warning for expected missing paths
            return member
                        
        current = data[path_parts[0]]  # Start at validmind
        for part in path_parts[1:]:
            if part in current.get('members', {}):
                current = current['members'][part]
            else:
                # If we can't find the direct path, try alternative approaches
                # For test suites, specially handle class aliases
                if 'test_suites' in path_parts and current.get('name') == 'test_suites':
                    # If we're looking for a class in test_suites but can't find it directly,
                    # check if it exists anywhere else in the codebase
                    class_name = path_parts[-1]
                    found_class = find_class_in_all_modules(class_name, data)
                    if found_class:
                        # Cache the result if found
                        _alias_cache[target_path] = found_class
                        return found_class
                
                print(f"Warning: Could not resolve alias path {target_path}, part '{part}' not found")
                return member
                

        # Cache the result
        _alias_cache[target_path] = current
        return current
    return member

def get_all_members(members: Dict[str, Any]) -> Set[str]:
    """Extract the __all__ list from a module's members if present."""
    if '__all__' in members:
        all_elements = members['__all__'].get('value', {}).get('elements', [])
        return {elem.strip("'") for elem in all_elements}
    return set()

def get_all_list(members: Dict[str, Any]) -> List[str]:
    """Extract the __all__ list from a module's members if present, preserving order."""
    if '__all__' in members:
        all_elements = members['__all__'].get('value', {}).get('elements', [])
        return [elem.strip("'") for elem in all_elements]
    return []

def sort_members(members, is_errors_module=False):
    """Sort members by kind and name."""
    if isinstance(members, dict):
        members = members.values()
    
    def get_sort_key(member):
        name = str(member.get('name', ''))
        kind = member.get('kind', '')
        
        if is_errors_module and kind == 'class':
            # Base errors first
            if name == 'BaseError':
                return ('0', '0', name)  # Use strings for consistent comparison
            elif name == 'APIRequestError':
                return ('0', '1', name)
            # Then group by category
            elif name.startswith('API') or name.endswith('APIError'):
                return ('1', '0', name)
            elif 'Model' in name:
                return ('2', '0', name)
            elif 'Test' in name:
                return ('3', '0', name)
            elif name.startswith('Invalid') or name.startswith('Missing'):
                return ('4', '0', name)
            elif name.startswith('Unsupported'):
                return ('5', '0', name)
            else:
                return ('6', '0', name)
        else:
            # Default sorting for non-error modules
            if kind == 'class':
                return ('0', name.lower())
            elif kind == 'function':
                return ('1', name.lower())
            else:
                return ('2', name.lower())
    
    return sorted(members, key=get_sort_key)

def is_public(member: Dict[str, Any], module: Dict[str, Any], full_data: Dict[str, Any], is_root: bool = False) -> bool:
    """Check if a member should be included in public documentation."""
    name = member.get('name', '')
    path = member.get('path', '')
    
    # Skip private members except __init__ and __post_init__
    if name.startswith('_') and name not in {'__init__', '__post_init__'}:
        return False
    
    # Specifically exclude SkipTestError and logger/get_logger from test modules
    if name in {'SkipTestError', 'logger'} and 'tests' in path:
        return False
    
    if name == 'get_logger' and path.startswith('validmind.tests'):
        return False
    
    # Check if the member is an alias that's imported from another module
    if member.get('kind') == 'alias' and member.get('target_path'):
        # If the module has __all__, only include aliases listed there
        if module and '__all__' in module.get('members', {}):
            module_all = get_all_members(module.get('members', {}))
            return name in module_all
        
        # Otherwise, skip aliases (imported functions) unless at root level
        if not is_root:
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

def clean_anchor_text(heading: str) -> str:
    """Safely clean heading text for anchor generation.
    
    Handles:
    - <span class="muted">()</span>
    - <span class='muted'>class</span>
    - Other HTML formatting
    """
    # First check if this is a class heading
    if '<span class="muted">class</span>' in heading or '<span class=\'muted\'>class</span>' in heading:
        # Remove the HTML span for class
        class_name = re.sub(r'<span class=["\']muted["\']>class</span>\s*', '', heading)
        return 'class-' + class_name.strip().lower()
    
    # For other headings, remove any HTML spans
    cleaned = re.sub(r'<span class=["\']muted["\']>\(\)</span>', '', heading)
    cleaned = re.sub(r'<span class=["\']muted["\']>[^<]*</span>', '', cleaned)
    return cleaned.strip().lower()

def collect_documented_items(module: Dict[str, Any], path: List[str], full_data: Dict[str, Any], is_root: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """Collect all documented items from a module and its submodules."""
    result = {}
    
    # Skip if no members
    if not module.get('members'):
        return result
    
    # Determine if this is the root module
    is_root = module.get('name') == 'validmind' or is_root
    
    # Build the current file path
    file_path = '/'.join(path)
    module_name = module.get('name', 'root')
    
    # For root module, parse validmind.qmd to get headings
    if is_root:
        module_items = []
        qmd_filename = f"{path[-1]}.qmd"
        qmd_path = written_qmd_files.get(qmd_filename)
        
        if qmd_path and os.path.exists(qmd_path):
            with open(qmd_path, 'r') as f:
                content = f.read()
            
            # Track current class for nesting methods
            current_class = None
            
            # Parse headings - only update the heading level checks
            for line in content.split('\n'):
                if line.startswith('## '):  # Main function/class level
                    heading = line[3:].strip()
                    anchor = clean_anchor_text(heading)
                    item = {
                        'text': heading,
                        'file': f"validmind/validmind.qmd#{anchor}"
                    }
                    
                    # Detect class by presence of class span or prefix span
                    is_class = '<span class="muted">class</span>' in heading or '<span class=\'muted\'>class</span>' in heading
                    prefix_class = '<span class="prefix"></span>' in heading
                    
                    if is_class or prefix_class:
                        item['contents'] = []
                        current_class = item
                    module_items.append(item)
                elif line.startswith('### ') and current_class:  # Method level
                    heading = line[4:].strip()
                    anchor = clean_anchor_text(heading)
                    method_item = {
                        'text': heading,
                        'file': f"validmind/validmind.qmd#{anchor}"
                    }
                    current_class['contents'].append(method_item)
            
            # Clean up empty contents lists
            for item in module_items:
                if 'contents' in item and not item['contents']:
                    del item['contents']
            
            if module_items:
                result['root'] = module_items
    
    # Process submodules
    for member in sort_members(module['members'], module.get('name') == 'errors'):
        if member['kind'] == 'module' and is_public(member, module, full_data, is_root):
            submodule_path = path + [member['name']]
            submodule_items = collect_documented_items(member, submodule_path, full_data, False)
            result.update(submodule_items)
            
            # Also check for nested modules in the submodule
            if member.get('members'):
                for submember in sort_members(member['members'], member.get('name') == 'errors'):
                    if submember['kind'] == 'module' and is_public(submember, member, full_data, False):
                        subsubmodule_path = submodule_path + [submember['name']]
                        subsubmodule_items = collect_documented_items(submember, subsubmodule_path, full_data, False)
                        result.update(subsubmodule_items)
    
    return result

# Add at module level
written_qmd_files = {}

def find_class_in_all_modules(class_name: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Recursively search for a class in all modules of the data structure."""
    if not isinstance(data, dict):
        return None
        
    # Check if this is the class we're looking for
    if data.get('kind') == 'class' and data.get('name') == class_name:
        return data
        
    # Special handling for common test suite classes
    if class_name.endswith(('Suite', 'Performance', 'Metrics', 'Diagnosis', 'Validation', 'Description')):
        # These are likely test suite classes, check specifically in test_suites module if available
        if 'validmind' in data and 'test_suites' in data['validmind'].get('members', {}):
            test_suites = data['validmind']['members']['test_suites']
            if class_name in test_suites.get('members', {}):
                return test_suites['members'][class_name]
    
    # Check members if this is a module
    if 'members' in data:
        for member_name, member in data['members'].items():
            # Direct match in members
            if member_name == class_name and member.get('kind') == 'class':
                return member
                
            # Recursive search in this member
            result = find_class_in_all_modules(class_name, member)
            if result:
                return result
                
    return None

def process_module(module: Dict[str, Any], path: List[str], env: Environment, full_data: Dict[str, Any]):
    """Process a module and its submodules."""
    # Parse docstrings first
    parse_docstrings(module)
    
    module_dir = os.path.join('docs', *path[:-1])
    ensure_dir(module_dir)
    
    # Extract __all__ list if present (preserving order)
    if module.get('members') and '__all__' in module.get('members', {}):
        module['all_list'] = get_all_list(module['members'])
    
    # Special handling for test_suites module
    is_test_suites = path and path[-1] == "test_suites"
    if is_test_suites:
        # Ensure all class aliases are properly resolved
        for member_name, member in module.get('members', {}).items():
            if member.get('kind') == 'alias' and member.get('target_path'):
                # Try to resolve and cache the target now
                resolve_alias(member, full_data)
    
    # Enhanced debugging for vm_models
    if path and path[-1] == 'vm_models':
        # Handle special case for vm_models module
        # Look for result module and copy necessary classes
        result_module = None
        for name, member in module.get('members', {}).items():
            if name == 'result' and member.get('kind') == 'module':
                result_module = member
                
                # Copy ResultTable and TestResult to vm_models members if needed
                if 'ResultTable' in member.get('members', {}):
                    module['members']['ResultTable'] = member['members']['ResultTable']
                
                if 'TestResult' in member.get('members', {}):
                    module['members']['TestResult'] = member['members']['TestResult']
                break
        
        if not result_module:
            # Fallback: try to find the classes directly in the full data structure
            result_table = find_class_in_all_modules('ResultTable', full_data)
            if result_table:
                module['members']['ResultTable'] = result_table
                
            test_result = find_class_in_all_modules('TestResult', full_data)
            if test_result:
                module['members']['TestResult'] = test_result
    
    # Check if this is a test module
    is_test_module = 'tests' in path
    
    # Get appropriate template based on module name
    if path[-1] == 'errors':
        # Use the specialized errors template for the errors module
        template = env.get_template('errors.qmd.jinja2')
        
        # Render with the errors template
        output = template.render(
            module=module,
            members=module.get('members', {}),  # Pass members directly
            full_data=full_data,
            is_errors_module=True
        )
    else:
        # Use the standard module template for all other modules
        template = env.get_template('module.qmd.jinja2')
        
        # Generate module documentation
        output = template.render(
            module=module,
            full_data=full_data,
            is_root=(len(path) <= 1),
            resolve_alias=resolve_alias,
            is_test_module=is_test_module  # Pass this flag to template
        )
    
    # Write output
    filename = f"{path[-1]}.qmd"
    output_path = os.path.join(module_dir, filename)
    with open(output_path, 'w') as f:
        f.write(output)
    
    # Track with full relative path as key
    rel_path = os.path.join(*path[1:], filename) if len(path) > 1 else filename
    full_path = os.path.join("docs", os.path.relpath(output_path, "docs"))
    written_qmd_files[rel_path] = full_path
    
    # Generate version.qmd for root module
    if module.get('name') == 'validmind' and module.get('members', {}).get('__version__'):
        version_template = env.get_template('version.qmd.jinja2')
        version_output = version_template.render(
            module=module,
            full_data=full_data
        )
        # Removed the underscores from the filename as Quarto treats files with underscores differently
        version_path = os.path.join('docs/validmind', 'version.qmd')
        with open(version_path, 'w') as f:
            f.write(version_output)
        written_qmd_files['version.qmd'] = version_path
    
    # Process submodules
    members = module.get('members', {})
    for name, member in members.items():
        if member.get('kind') == 'module':
            if is_public(member, module, full_data, is_root=len(path) <= 1):
                process_module(member, path + [name], env, full_data)

def lint_markdown_files(output_dir: str):
    """Clean up whitespace and formatting in all generated markdown files."""
    for path in Path(output_dir).rglob('*.qmd'):
        with open(path) as f:
            content = f.read()
        
        # Split content into front matter and body
        parts = content.split('---', 2)
        if len(parts) >= 3:
            # Preserve front matter and format the rest
            front_matter = parts[1]
            body = parts[2]
            formatted_body = mdformat.text(body, options={
                "wrap": "no",
                "number": False,
                "normalize_whitespace": True
            })
            formatted = f"---{front_matter}---\n\n{formatted_body}"
        else:
            # No front matter, format everything
            formatted = mdformat.text(content, options={
                "wrap": "no",
                "number": False,
                "normalize_whitespace": True
            })
            
        with open(path, 'w') as f:
            f.write(formatted)

def parse_docstrings(data: Dict[str, Any]):
    """Recursively parse all docstrings in the data structure."""
    if isinstance(data, dict):
        if 'docstring' in data:
            if isinstance(data['docstring'], dict):
                original = data['docstring'].get('value', '')
            elif isinstance(data['docstring'], str):
                original = data['docstring']
            else:
                original = str(data['docstring'])
            
            try:
                # Pre-process all docstrings to normalize newlines
                sections = original.split('\n\n')
                # Join lines in the first section (description) with spaces
                if sections:
                    sections[0] = ' '.join(sections[0].split('\n'))
                # Keep other sections as-is
                original = '\n\n'.join(sections)
                
                parsed = parse(original, style=Style.GOOGLE)
                
                data['docstring'] = {
                    'value': original,
                    'parsed': parsed
                }
            except Exception as e:
                print(f"\nParsing failed for {data.get('name', 'unknown')}:")
                print(f"Error: {str(e)}")
                print(f"Original:\n{original}")
        
        if 'members' in data:
            for member in data['members'].values():
                parse_docstrings(member)

def get_inherited_members(base: Dict[str, Any], full_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get all inherited members from a base class."""
    # Handle case where a class object is passed instead of a base name
    if isinstance(base, dict) and 'bases' in base:
        all_members = []
        for base_item in base['bases']:
            if isinstance(base_item, dict) and 'name' in base_item:
                base_members = get_inherited_members(base_item['name'], full_data)
                all_members.extend(base_members)
        return all_members
    
    # Get the base class name
    base_name = base if isinstance(base, str) else base.get('name', '')
    if not base_name:
        return []
    
    # Handle built-in exceptions
    if base_name == 'Exception' or base_name.startswith('builtins.'):
        return [
            {'name': 'with_traceback', 'kind': 'builtin', 'base': 'builtins.BaseException'},
            {'name': 'add_note', 'kind': 'builtin', 'base': 'builtins.BaseException'}
        ]
    
    # Look for the base class in the errors module
    errors_module = full_data.get('validmind', {}).get('members', {}).get('errors', {}).get('members', {})
    base_class = errors_module.get(base_name)
    
    if not base_class:
        return []
    
    # Return the base class and its description method if it exists
    members = [{'name': base_name, 'kind': 'class', 'base': base_name}]
    
    # Add all public methods
    for name, member in base_class.get('members', {}).items():
        # Skip private methods (including __init__)
        if name.startswith('_'):
            continue
        
        if member['kind'] in ('function', 'method', 'property'):
            # Add the method to the list of inherited members
            method_info = {
                'name': name,
                'kind': 'method',
                'base': base_name,
                'parameters': member.get('parameters', []),  # Include parameters
                'returns': member.get('returns', None),       # Include return type
                'docstring': member.get('docstring', {}).get('value', ''),
            }
            
            members.append(method_info)
    
    # Add built-in methods from Exception
    members.extend([
        {'name': 'with_traceback', 'kind': 'builtin', 'base': 'builtins.BaseException'},
        {'name': 'add_note', 'kind': 'builtin', 'base': 'builtins.BaseException'}
    ])
    
    return members

def get_child_files(files_dict: Dict[str, str], module_name: str) -> List[Dict[str, Any]]:
    """Get all child QMD files for a given module."""
    prefix = f'docs/validmind/{module_name}/'
    directory_structure = {}
    
    # First pass: organize files by directory
    for filename, path in files_dict.items():
        if path.startswith(prefix) and path != f'docs/validmind/{module_name}.qmd':
            # Remove the prefix to get the relative path
            rel_path = path.replace('docs/', '')
            parts = Path(rel_path).parts[2:]  # Skip 'validmind' and module_name
            
            # Handle directory-level QMD and its children
            if len(parts) == 1:  # Direct child
                dir_name = Path(parts[0]).stem
                if dir_name not in directory_structure:
                    directory_structure[dir_name] = {
                        'text': dir_name,
                        'file': f'validmind/{rel_path}'  # Add validmind/ prefix
                    }
            else:  # Nested file
                dir_name = parts[0]
                if dir_name not in directory_structure:
                    directory_structure[dir_name] = {
                        'text': dir_name,
                        'file': f'validmind/validmind/{module_name}/{dir_name}.qmd'  # Add validmind/ prefix
                    }
                
                # Add to contents if it's a child file
                if 'contents' not in directory_structure[dir_name]:
                    directory_structure[dir_name]['contents'] = []
                
                directory_structure[dir_name]['contents'].append({
                    'text': Path(parts[-1]).stem,
                    'file': f'validmind/{rel_path}'  # Add validmind/ prefix
                })
    
    # Sort children within each directory
    for dir_info in directory_structure.values():
        if 'contents' in dir_info:
            dir_info['contents'].sort(key=lambda x: x['text'])
    
    # Return sorted list of directories
    return sorted(directory_structure.values(), key=lambda x: x['text'])

def has_subfiles(files_dict, module_name):
    """Check if a module has child QMD files."""
    prefix = f'docs/validmind/{module_name}/'
    return any(path.startswith(prefix) for path in files_dict.values())

def find_qmd_files(base_path: str) -> Dict[str, str]:
    """Find all .qmd files and their associated paths."""
    # Convert the written_qmd_files paths to be relative to docs/
    relative_paths = {}
    for filename, path in written_qmd_files.items():
        if path.startswith('docs/'):
            relative_paths[filename] = path
        else:
            relative_paths[filename] = f'docs/{path}'
    return relative_paths

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
    env.filters['has_subfiles'] = has_subfiles
    env.filters['get_child_files'] = get_child_files
    env.globals['is_public'] = is_public
    env.globals['resolve_alias'] = resolve_alias
    env.globals['get_all_members'] = get_all_members
    env.globals['get_all_list'] = get_all_list
    env.globals['get_inherited_members'] = get_inherited_members
    
    # Start processing from root module
    if 'validmind' in data:
        # First pass: Generate module documentation
        process_module(data['validmind'], ['validmind'], env, data)
        
        qmd_files = find_qmd_files(output_dir)
        
        # Add to template context
        env.globals['qmd_files'] = qmd_files
        
        # Second pass: Collect all documented items
        documented_items = collect_documented_items(
            module=data['validmind'],
            path=['validmind'],
            full_data=data,
            is_root=True
        )
        
        # Generate sidebar with collected items
        sidebar_template = env.get_template('sidebar.qmd.jinja2')
        sidebar_output = sidebar_template.render(
            module=data['validmind'],
            full_data=data,
            is_root=True,
            resolve_alias=resolve_alias,
            documented_items=documented_items
        )
        
        # Write sidebar
        sidebar_path = os.path.join(output_dir, '_sidebar.yml')
        with open(sidebar_path, 'w') as f:
            f.write(sidebar_output)
            
        # Clean up markdown formatting
        lint_markdown_files(output_dir)
    else:
        print("Error: No 'validmind' module found in JSON")

def parse_docstring(docstring):
    """Parse a docstring into its components."""
    if not docstring:
        return None
    try:
        # Pre-process docstring to reconstruct original format
        lines = docstring.split('\n')
        processed_lines = []
        in_args = False
        current_param = []
        
        for line in lines:
            line = line.strip()
            # Check if we're in the Args section
            if line.startswith('Args:'):
                in_args = True
                processed_lines.append(line)
                continue
                
            if in_args and line:
                # Fix mangled parameter lines like "optional): The test suite name..."
                if line.startswith('optional)'):
                    # Extract the actual parameter name from the description
                    desc_parts = line.split(':', 1)[1].strip().split('(')
                    if len(desc_parts) > 1:
                        param_name = desc_parts[1].split(',')[0].strip()
                        desc = desc_parts[0].strip()
                        line = f"    {param_name} (str, optional): {desc}"
                processed_lines.append(line)
            else:
                processed_lines.append(line)
                
        processed_docstring = '\n'.join(processed_lines)
        return parse(processed_docstring, style=Style.GOOGLE)
    except Exception as e:
        # Fallback to just returning the raw docstring
        return {'value': docstring}

if __name__ == '__main__':
    generate_docs(
        json_path='docs/validmind.json',
        template_dir='docs/templates',
        output_dir='docs'
    )