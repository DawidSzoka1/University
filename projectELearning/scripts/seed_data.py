"""Dane testowe platformy — skrót do komendy ``seed_demo``.

    python scripts/seed_data.py            # = python manage.py seed_demo
    python scripts/seed_data.py --reset    # = python manage.py seed_demo --reset
    docker compose exec -T web python scripts/seed_data.py --reset

Dane do edycji: apps/core/demo/data.py. Logika: apps/core/management/commands/seed_demo.py.
"""

from __future__ import annotations

import os
import sys

import django

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.core.management import call_command  # noqa: E402

if __name__ == "__main__":
    call_command("seed_demo", *[arg for arg in sys.argv[1:] if arg.startswith("--")])
else:
    # Uruchomione przez `python manage.py shell < scripts/seed_data.py`
    call_command("seed_demo")
