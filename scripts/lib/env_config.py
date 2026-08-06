"""Parse simple KEY=VALUE .env-style config files shared with the bash scripts."""
from pathlib import Path


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE config file into a dict.

    Supports plain `KEY=VALUE`, `KEY="VALUE"`, and `KEY='VALUE'` lines,
    plus blank lines and full-line '#' comments. Does NOT support bash
    features like `export KEY=VALUE` or trailing inline '# comment'
    text after a value -- config/*.env files in this project should
    stick to the plain KEY=VALUE form so bash `source` and this loader
    stay in agreement.
    """
    result: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        result[key] = value
    return result
