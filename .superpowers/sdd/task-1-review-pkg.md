# Review package
BASE: 92185ab3c0c053b4f6ad3bafadce26aa4b5fddb0
HEAD: f7223cf109432113f13eccca37c0a934edff4ecf

## Commits
f7223cf chore: scaffold personal pipeline data paths

## Stat
 .gitignore                        | 20 ++++++++++++--------
 config/personal_sources.env       |  3 +++
 data/personal/README.md           |  6 ++++++
 data/personal/candidates/.gitkeep |  0
 4 files changed, 21 insertions(+), 8 deletions(-)

## Diff
diff --git a/.gitignore b/.gitignore
index 754052b..e794bd9 100644
--- a/.gitignore
+++ b/.gitignore
@@ -1,8 +1,12 @@
-.venv/
-__pycache__/
-*.pyc
-.pytest_cache/
-output/
-data/generated/*
-!data/generated/.gitkeep
-data/train.jsonl
+.venv/
+__pycache__/
+*.pyc
+.pytest_cache/
+output/
+data/generated/*
+!data/generated/.gitkeep
+data/train.jsonl
+
+# Personal pipeline datasets (may contain private chat text)
+data/personal/**/*.jsonl
+!data/personal/candidates/.gitkeep
diff --git a/config/personal_sources.env b/config/personal_sources.env
new file mode 100644
index 0000000..999bde9
--- /dev/null
+++ b/config/personal_sources.env
@@ -0,0 +1,3 @@
+# Absolute or ~ paths OK. Override with env vars of the same name.
+AGENT_TRANSCRIPTS_DIR=~/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
+MARKDOWN_GLOBS=docs/superpowers/specs/*.md,docs/superpowers/plans/*.md,README.md
diff --git a/data/personal/README.md b/data/personal/README.md
new file mode 100644
index 0000000..3a12971
--- /dev/null
+++ b/data/personal/README.md
@@ -0,0 +1,6 @@
+# Personal tech datasets
+
+1. Run `python scripts/extract_personal_candidates.py`
+2. Review/edit files under `candidates/`
+3. Promote: `python scripts/promote_personal_data.py --reviewed`
+4. Train each adapter with `scripts/train_lora.py --data ... --output ...`
diff --git a/data/personal/candidates/.gitkeep b/data/personal/candidates/.gitkeep
new file mode 100644
index 0000000..e69de29
