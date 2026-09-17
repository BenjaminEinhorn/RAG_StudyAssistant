import os
os.environ.setdefault("COURSE_BUILD_MODE", os.environ.get("COURSE_BUILD_MODE", "local"))
from course_assistant.config.settings import Settings
from course_assistant.factory import build_app_state

app = build_app_state(Settings())
for f in sorted(os.listdir("data/materials")):
    if f.lower().endswith((".pdf", ".pptx", ".txt", ".md")):
        r = app.catalog.add_file(f"data/materials/{f}")
        print(r["message"], "| chunks:", r.get("chunks"), "| slides:", r.get("slides"))
print("TOTAL DOCS:", len(app.catalog.list_documents()))
