import zipfile
import os

FILES_TO_ZIP = ['gcn.py', 'result.json']
ZIP_NAME = 'result.zip'

with zipfile.ZipFile(ZIP_NAME, 'w') as zipf:
    for file in FILES_TO_ZIP:
        if os.path.exists(file):
            zipf.write(file)
            print(f'Added {file} to {ZIP_NAME}')
        else:
            print(f'Warning: {file} not found')
