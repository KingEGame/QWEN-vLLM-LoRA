BASE 95cf6afbb527f03e5276c861d649022de059aa34
HEAD ed39be0
ed39be0 docs: link learning guides from README
 README.md | 7 +++++++
 1 file changed, 7 insertions(+)
diff --git a/README.md b/README.md
index f47154e..7492f24 100644
--- a/README.md
+++ b/README.md
@@ -6,10 +6,15 @@ own docs. LoRA fine-tuning of the 27B checkpoint is not in this path yet.
 
 Authored to run on **Linux or WSL2** with an NVIDIA GPU. Native Windows cannot
 run the GPU stack; Windows teammates use the thin setup wrappers below, which
 forward into WSL.
 
+## Guides
+
+- [Architecture learning](docs/guides/architecture-learning.md) — what we built, vLLM vs LoRA vs Qwen, limits
+- [Operator cheatsheet](docs/guides/operator-cheatsheet.md) — commands by scenario
+
 ## Onboarding (setup only)
 
 One command installs the Python venv, dependencies, and verifies CUDA/vLLM.
 Starting the server and sending a test request are **manual** next steps.
 
@@ -158,10 +163,12 @@ text — keep it local unless you explicitly version sanitized data.
 
 ## Troubleshooting
 
 Driver, CUDA, and out-of-memory issues are documented in:
 
+- [Architecture learning](docs/guides/architecture-learning.md)
+- [Operator cheatsheet](docs/guides/operator-cheatsheet.md)
 - [Design: Qwen + vLLM + LoRA setup](docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md)
 - [Design: easy onboard setup scripts](docs/superpowers/specs/2026-08-06-easy-onboard-setup-scripts-design.md)
 - [Design: Qwen3.6-27B AWQ serve](docs/superpowers/specs/2026-08-08-qwen36-27b-awq-serve-design.md)
 - [Design: Qwen3.6-27B LoRA train](docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md)
 - [Design: personal tech pipeline](docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md)
