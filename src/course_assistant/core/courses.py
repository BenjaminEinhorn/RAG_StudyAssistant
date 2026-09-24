"""Courses: separate workspaces so one install serves several classes.

Each course has its own folder holding its uploaded files, slide images,
vector indexes, chunk registry and quiz topics, so documents, answers and
quizzes never mix between classes. The list of courses lives in
``<data>/courses.json``. The first course uses the data folder itself, so data
ingested before courses existed stays where it is and keeps working.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

DEFAULT_COURSE = os.environ.get("COURSE_NAME", "MBAX 6418")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class CourseRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "courses.json"
        try:
            self._courses: list[dict] = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError):
            self._courses = []
        if not self._courses:
            self._courses = [{"name": DEFAULT_COURSE, "dir": "."}]
            self._save()

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._courses, indent=1))

    def names(self) -> list[str]:
        return [c["name"] for c in self._courses]

    def dir_of(self, name: str) -> Path:
        for c in self._courses:
            if c["name"] == name:
                return (self.root / c["dir"]).resolve()
        raise KeyError(f"No course named '{name}'.")

    def create(self, name: str) -> str:
        """Register a new, empty course. Returns its cleaned name."""
        name = " ".join(name.split())
        slug = slugify(name)
        if not slug:
            raise ValueError("Give the course a name with letters or numbers.")
        if any(c["name"].lower() == name.lower() or slugify(c["name"]) == slug
               for c in self._courses):
            raise ValueError(f"A course named '{name}' already exists.")
        self._courses.append({"name": name, "dir": f"courses/{slug}"})
        self._save()
        return name
