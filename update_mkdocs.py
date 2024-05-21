import os
import yaml

# Paths
mkdocs_path = os.path.join(os.path.dirname(__file__), 'mkdocs.yml')
docs_path = os.path.join(os.path.dirname(__file__), 'docs')
PDT_path = os.path.join(os.path.dirname(__file__), 'tripsender')

def update_mkdocs():
    nav_updates = {}

    # Ensure the API directory exists and is empty
    api_path = os.path.join(docs_path, 'api')
    if os.path.exists(api_path):
        for file in os.listdir(api_path):
            os.remove(os.path.join(api_path, file))
    else:
        os.makedirs(api_path)

    # Create md files for each .py file in PDT_path (excluding __init__.py) and update nav
    for root, dirs, files in os.walk(PDT_path):
        for file in files:
            if file.endswith('.py') and file not in ['__init__.py']:  # Skip __init__.py files
                file_name = file[:-3]  # Remove '.py' extension
                file_path = os.path.join(docs_path, 'api', file_name + '.md')

                with open(file_path, 'w') as f:
                    f.write('::: tripsender.' + file_name)

                nav_updates[file_name] = 'api/' + file_name + '.md'

    # Load and parse mkdocs.yml
    try:
        with open(mkdocs_path, 'r') as f:
            mkdocs = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return

    # Update the nav structure
    if 'nav' in mkdocs:
        for section in mkdocs['nav']:
            if 'API Documentation' in section:
                pdt_section = section['API Documentation']
                for pdt_key in pdt_section:
                    if 'Tripsender' in pdt_key:
                        pdt_key['Tripsender'] = [{k: v} for k, v in nav_updates.items()]

    # Write updated mkdocs.yml back
    with open(mkdocs_path, 'w') as f:
        yaml.safe_dump(mkdocs, f, default_flow_style=False, sort_keys=False)

    print('mkdocs.yml file updated')

if __name__ == '__main__':
    update_mkdocs()
