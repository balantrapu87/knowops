"""Add the project root to sys.path so `import knowops` works from any cwd."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
