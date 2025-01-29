#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Any, Dict, Set, List
from jinja2 import Environment, FileSystemLoader
import mdformat
from docstring_parser import parse, Style
from glob import glob
import subprocess

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
    
    # Add debug logging
    # print(f"\nChecking visibility for: {name}")
    # print(f"Is root: {is_root}")
    
    # Skip private members except __init__ and __post_init__
    if name.startswith('_') and name not in {'__init__', '__post_init__'}:
        # print(f"- Skipping private member: {name}")
        return False
    
    # At root level, only show items from __all__
    if is_root:
        root_all = get_all_members(full_data['validmind'].get('members', {}))
        # print(f"- Root __all__: {root_all}")
        return name in root_all
    
    # If module has __all__, only include members listed there
    if module and '__all__' in module.get('members', {}):
        module_all = get_all_members(module.get('members', {}))
        # print(f"- Module __all__: {module_all}")
        # print(f"- Is {name} in module __all__? {name in module_all}")
        return name in module_all
    
    # print(f"- No __all__ found, including {name}")
    return True

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def collect_documented_items(module: Dict[str, Any], path: List[str], full_data: Dict[str, Any], is_root: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """Collect all documented items from a module and its submodules."""
    result = {}
    
    # Skip if no members
    if not module.get('members'):
        return result
    
    # Build the current file path
    file_path = '/'.join(path)
    module_name = module.get('name', 'root')
    
    # Collect items from this module
    module_items = []
    for member in sort_members(module['members'], module.get('name') == 'errors'):
        if not is_public(member, module, full_data, is_root):
            continue
            
        if member['kind'] in ('function', 'class'):
            # For root module items, always use reference/validmind.qmd path
            file_prefix = 'reference/validmind' if is_root else file_path
            module_items.append({
                'text': f"{member['name']}()" if member['kind'] == 'function' else member['name'],
                'file': f"{file_prefix}.qmd#{member['name']}"
            })
        elif member['kind'] == 'alias':
            target = resolve_alias(member, full_data)
            if target and target.get('docstring'):
                file_prefix = 'reference/validmind' if is_root else file_path
                module_items.append({
                    'text': f"{member['name']}()",
                    'file': f"{file_prefix}.qmd#{member['name']}"
                })
    
    if module_items:
        result['root' if is_root else module_name] = module_items
    
    # Recursively collect from submodules
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

def process_module(module: Dict[str, Any], path: List[str], env: Environment, full_data: Dict[str, Any]):
    """Process a module and its submodules."""
    # if module.get('name') == 'tests':
    #     print("\nProcessing tests module members:")
    #     for name, member in module.get('members', {}).items():
    #         if is_public(member, module, full_data):
    #             print(f"\n{name}:")
    #             print(f"  Original: kind={member.get('kind')}")
    #             if member.get('kind') == 'alias':
    #                 resolved = resolve_alias(member, full_data)
    #                 print(f"  Resolved: kind={resolved.get('kind')}")
    #                 print(f"  Target path: {member.get('target_path')}")
    #                 print(f"  In __all__: {name in get_all_members(module.get('members', {}))}")
    
    # Parse docstrings first
    parse_docstrings_recursively(module)
    
    module_dir = os.path.join('docs', *path[:-1])
    ensure_dir(module_dir)
    
    # Get module template
    module_template = env.get_template('module.qmd.jinja2')
    
    # Generate module documentation
    output = module_template.render(
        module=module,
        full_data=full_data,
        is_root=(len(path) <= 1),
        resolve_alias=resolve_alias
    )
    
    # Write output
    filename = f"{path[-1]}.qmd"
    output_path = os.path.join(module_dir, filename)
    with open(output_path, 'w') as f:
        f.write(output)
    
    # Track with full path and print full path
    full_path = os.path.join("docs", os.path.relpath(output_path, "docs"))
    print(f"Wrote file: {full_path}")
    written_qmd_files[filename] = full_path
    
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
            formatted_body = mdformat.text(body, options={"wrap": "no"})
            formatted = f"---{front_matter}---\n\n{formatted_body}"
        else:
            # No front matter, format everything
            formatted = mdformat.text(content, options={"wrap": "no"})
            
        with open(path, 'w') as f:
            f.write(formatted)

def format_google_docstring(docstring: str) -> str:
    """Format a Google-style docstring from JSON format back to proper structure."""
    lines = []
    sections = docstring.split('\n\n')
    
    for section in sections:
        if section.startswith('Args:'):
            lines.append('Args:')
            args_text = section.replace('Args:', '').strip()
            
            current_param = None
            param_desc = []
            
            parts = args_text.split()
            i = 0
            while i < len(parts):
                part = parts[i]
                
                if ':' in part and not part.startswith('(default'):
                    # If we have a previous parameter, add it
                    if current_param:
                        lines.append(f"    {current_param}: {' '.join(param_desc)}")
                    current_param = part.split(':', 1)[0]
                    param_desc = [part.split(':', 1)[1]]
                
                elif part.startswith('(default'):
                    # Look ahead for the default value
                    try:
                        if ':' in part:
                            default_value = part.split(':', 1)[1].rstrip(')')
                            param_desc.append(f"(default: {default_value})")
                        else:
                            param_desc.append(part)
                    except IndexError:
                        param_desc.append(part)
                
                else:
                    param_desc.append(part)
                
                i += 1
            
            # Add the last parameter
            if current_param:
                lines.append(f"    {current_param}: {' '.join(param_desc)}")
        
        elif section.startswith('Returns:'):
            lines.append('\nReturns:')
            returns_text = section.replace('Returns:', '').strip()
            lines.append(f"    {returns_text}")
        
        else:
            lines.append(section.strip())
    
    return '\n'.join(lines)

def format_rst_docstring(docstring: str) -> str:
    """Format an RST-style docstring from JSON format back to proper structure."""
    
    # Split on ":param" and ":return:" to separate sections
    parts = []
    current = []
    
    for part in docstring.split():
        if part.startswith(':param') or part.startswith(':return:'):
            if current:
                parts.append(' '.join(current))
            current = [part]
        else:
            current.append(part)
    if current:
        parts.append(' '.join(current))
    
    # Join with newlines
    result = '\n'.join(parts)
    return result

def try_parse_docstring(docstring: str) -> Any:
    """Try to parse a docstring in multiple styles, defaulting to Google."""
    # Convert escaped newlines to actual newlines
    docstring = docstring.replace('\\n', '\n')
    
    # Try Google style first
    try:
        return parse(docstring, style=Style.GOOGLE)
    except Exception:
        # Fallback to RST style
        try:
            return parse(docstring, style=Style.REST)
        except Exception:
            return None

def parse_docstrings_recursively(data: Dict[str, Any]):
    """Recursively parse all docstrings in the data structure."""
    if isinstance(data, dict):
        if 'docstring' in data:
            if isinstance(data['docstring'], dict):
                original = data['docstring'].get('value', '')
            elif isinstance(data['docstring'], str):
                original = data['docstring']
            else:
                original = str(data['docstring'])
            
            # Parse docstring once and store both original and parsed versions
            parsed = try_parse_docstring(original)
            data['docstring'] = {
                'value': original,
                'parsed': parsed
            } if parsed else {'value': original}
        
        if 'members' in data:
            for member in data['members'].values():
                parse_docstrings_recursively(member)

def get_inherited_members(base: Dict[str, Any], full_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get all inherited members from a base class."""
    # Get the base class name
    base_name = base.get('name', '')
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
        # Include __init__ and __str__, skip other private methods
        if name.startswith('_') and name not in {'__init__', '__str__'}:
            continue
        
        if member['kind'] in ('function', 'method', 'property'):
            members.append({
                'name': name,
                'kind': 'method',
                'base': base_name,
                'docstring': member.get('docstring', {}).get('value', '')
            })
    
    # Add built-in methods from Exception
    members.extend([
        {'name': 'with_traceback', 'kind': 'builtin', 'base': 'builtins.BaseException'},
        {'name': 'add_note', 'kind': 'builtin', 'base': 'builtins.BaseException'}
    ])
    
    return members

def generate_module_doc(module, full_data, env, output_dir):
    """Generate documentation for a module."""
    is_errors = module.get('name') == 'errors'
    
    # Choose template based on module name
    template = env.get_template('errors.qmd.jinja2' if is_errors else 'module.qmd.jinja2')
    
    # For errors module, filter to only include classes and skip module-level functions
    if is_errors:
        filtered_members = {
            name: member for name, member in module.get('members', {}).items()
            if member.get('kind') == 'class'
        }
    else:
        filtered_members = module.get('members', {})
    
    # Generate documentation
    output = template.render(
        module=module,
        members=filtered_members,
        full_data=full_data,
        doc=doc_utils,
        is_errors_module=is_errors
    )
    
    # Write output
    output_path = os.path.join(output_dir, f"{module['name']}.qmd")
    with open(output_path, 'w') as f:
        f.write(output)

def find_qmd_files(base_path: str) -> Dict[str, List[str]]:
    """Find all .qmd files and their associated folders in docs/validmind."""
    print("\nEntering find_qmd_files()")
    print(f"\nFiles written during documentation generation (count: {len(written_qmd_files)}):")
    for filename, path in written_qmd_files.items():
        print(f"  {filename}: {path}")
    
    return written_qmd_files

def generate_docs(json_path: str, template_dir: str, output_dir: str):
    """Generate documentation from JSON data using templates."""
    print("\nEntering generate_docs()")
    print(f"output_dir: {output_dir}")
    
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
    env.globals['get_inherited_members'] = get_inherited_members
    
    # Start processing from root module
    if 'validmind' in data:
        # First pass: Generate module documentation
        process_module(data['validmind'], ['validmind'], env, data)
        
        print("\nAbout to call find_qmd_files()")
        qmd_files = find_qmd_files(output_dir)
        print(f"Found QMD files: {qmd_files}")
        
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

def debug_docstring_state(stage: str, docstring: Any):
    """Debug helper to track docstring transformations."""
    print(f"\n=== {stage} ===")
    if isinstance(docstring, dict):
        if 'value' in docstring:
            print("Value:", docstring['value'][:200] + "..." if len(docstring['value']) > 200 else docstring['value'])
        if 'parsed' in docstring and docstring['parsed']:
            print("\nParsed params:")
            for param in docstring['parsed'].params:
                print(f"- arg_name: {param.arg_name}")
                print(f"  type_name: {param.type_name}")
                print(f"  description: {param.description}")
    else:
        print(docstring[:200] + "..." if len(str(docstring)) > 200 else docstring)

if __name__ == '__main__':
    generate_docs(
        json_path='docs/validmind.json',
        template_dir='docs/templates',
        output_dir='docs'
    )