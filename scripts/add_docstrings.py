import ast
from pathlib import Path

root = Path(__file__).resolve().parent.parent
updated_files = []

for path in sorted(root.glob('**/*.py')):
    if path.match('**/__pycache__/**') or path.name == 'add_docstrings.py':
        continue

    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    module_doc = ast.get_docstring(tree)
    lines = source.splitlines(keepends=True)
    inserts = []

    if not module_doc:
        module_title = path.stem.replace('_', ' ').capitalize()
        module_docstring = f'"""{module_title} module for AI Phishing Detection.\n\nProvides utilities and application logic for the project.\n"""\n\n'
        inserts.append((0, module_docstring))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if ast.get_docstring(node):
                continue
            if not node.body:
                continue

            start = node.body[0].lineno - 1
            indent = ' ' * (node.col_offset + 4)
            params = [arg.arg for arg in node.args.args if arg.arg not in ('self', 'cls')]
            lines_doc = [f'"""{node.name.replace("_", " ").capitalize()}.']

            if params:
                lines_doc.append('')
                lines_doc.append('Parameters')
                lines_doc.append('----------')
                for param in params:
                    lines_doc.append(f'{param} : TYPE')
                    lines_doc.append(f'    Description of {param}.')

            has_return = any(isinstance(child, ast.Return) for child in ast.walk(node))
            if has_return and node.name not in ('__init__',):
                lines_doc.append('')
                lines_doc.append('Returns')
                lines_doc.append('-------')
                lines_doc.append('TYPE')
                lines_doc.append('    Description of return value.')

            if any(isinstance(child, ast.Raise) for child in ast.walk(node)):
                lines_doc.append('')
                lines_doc.append('Raises')
                lines_doc.append('------')
                lines_doc.append('Exception')
                lines_doc.append('    If an error occurs during execution.')

            lines_doc.append('"""')
            docstring = '\n'.join(f'{indent}{line}' if i != 0 else f'{indent}{line}' for i, line in enumerate(lines_doc)) + '\n'
            inserts.append((start, docstring))

    if not inserts:
        continue

    inserts.sort(reverse=True)
    for lineno, text in inserts:
        lines.insert(lineno, text)

    path.write_text(''.join(lines), encoding='utf-8')
    updated_files.append(str(path.relative_to(root)))

print('Updated files:')
for fp in updated_files:
    print(fp)
