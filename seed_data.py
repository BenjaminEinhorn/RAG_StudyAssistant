import os

# Mode comes from the environment or .env (Settings). Do not set a default here:
# os.environ.setdefault would run before .env is loaded and silently force local.
from course_assistant.config.settings import Settings
from course_assistant.factory import build_app_state

settings = Settings()
print("mode:", settings.build_mode)
app = build_app_state(settings)
for f in sorted(os.listdir("data/materials")):
    if f.lower().endswith((".pdf", ".pptx", ".txt", ".md")):
        r = app.catalog.add_file(f"data/materials/{f}")
        print(r["message"], "| chunks:", r.get("chunks"), "| slides:", r.get("slides"))
print("TOTAL DOCS:", len(app.catalog.list_documents()))
