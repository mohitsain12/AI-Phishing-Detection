"""Script utility for automatically inserting docstring templates into Python modules.

Uses Python AST parsing to inspect functions and module definitions.
"""

import ast
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
updated_files_list = []

for file_path in sorted(ROOT_DIRECTORY.glob('**/*.py')):
    if file_path.match('**/__pycache__/**') or file_path.name == 'add_docstrings.py':
        continue

    source_code = file_path.read_text(encoding='utf-8')
    syntax_tree = ast.parse(source_code)
    module_docstring = ast.get_docstring(syntax_tree)
    source_lines = source_code.splitlines(keepends=True)
    docstring_inserts = []

    if not module_docstring:
        module_title = file_path.stem.replace('_', ' ').capitalize()
        default_docstring = (
            f'"""{module_title} module for AI Phishing Detection.\n\n'
            f'Provides utilities and application logic for the project.\n"""\n\n'
        )
        docstring_inserts.append((0, default_docstring))

    for ast_node in ast.walk(syntax_tree):
        if isinstance(ast_node, ast.FunctionDef):
            if ast.get_docstring(ast_node) or not ast_node.body:
                continue

            line_start = ast_node.body[0].lineno - 1
            indentation_space = ' ' * (ast_node.col_offset + 4)
            parameters_list = [
                arg.arg for arg in ast_node.args.args if arg.arg not in ('self', 'cls')
            ]
            docstring_lines = [f'"""{ast_node.name.replace("_", " ").capitalize()}.']

            if parameters_list:
                docstring_lines.append('')
                docstring_lines.append('Parameters')
                docstring_lines.append('----------')
                for parameter_name in parameters_list:
                    docstring_lines.append(f'{parameter_name} : TYPE')
                    docstring_lines.append(f'    Description of {parameter_name}.')

            has_return_statement = any(
                isinstance(child, ast.Return) for child in ast.walk(ast_node)
            )
            if has_return_statement and ast_node.name not in ('__init__',):
                docstring_lines.append('')
                docstring_lines.append('Returns')
                docstring_lines.append('-------')
                docstring_lines.append('TYPE')
                docstring_lines.append('    Description of return value.')

            if any(isinstance(child, ast.Raise) for child in ast.walk(ast_node)):
                docstring_lines.append('')
                docstring_lines.append('Raises')
                docstring_lines.append('------')
                docstring_lines.append('Exception')
                docstring_lines.append('    If an error occurs during execution.')

            docstring_lines.append('"""')
            formatted_docstring = (
                '\n'.join(
                    f'{indentation_space}{line}' if i != 0 else f'{indentation_space}{line}'
                    for i, line in enumerate(docstring_lines)
                ) + '\n'
            )
            docstring_inserts.append((line_start, formatted_docstring))

    if not docstring_inserts:
        continue

    docstring_inserts.sort(reverse=True)
    for line_number, text_content in docstring_inserts:
        source_lines.insert(line_number, text_content)

    file_path.write_text(''.join(source_lines), encoding='utf-8')
    updated_files_list.append(str(file_path.relative_to(ROOT_DIRECTORY)))

print('Updated files:')
for updated_path in updated_files_list:
    print(updated_path)
