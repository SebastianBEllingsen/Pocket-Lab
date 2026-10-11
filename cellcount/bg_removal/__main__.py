"""Entry point for ``python -m cellcount.bg_removal`` (see cli.py for the flags)."""
import sys

from .cli import main

sys.exit(main())
