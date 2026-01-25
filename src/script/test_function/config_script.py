import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')

lib_pano_path = os.path.join(SRC_ROOT, 'lib_pano')
assert os.path.isdir(lib_pano_path), f"could not found lib_pano directory: {lib_pano_path}"

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

print(f"project root directory: {PROJECT_ROOT}")
print(f"source path '{SRC_ROOT}' has been added to sys.path")