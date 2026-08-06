from pathlib import Path

from scripts.lib.env_config import load_env_file


def test_load_env_file_parses_key_value_pairs(tmp_path: Path):
    env_file = tmp_path / "model.env"
    env_file.write_text("MODEL=Qwen/Qwen3-4B-Instruct-2507\nPORT=8000\n")

    result = load_env_file(env_file)

    assert result == {"MODEL": "Qwen/Qwen3-4B-Instruct-2507", "PORT": "8000"}


def test_load_env_file_skips_comments_and_blank_lines(tmp_path: Path):
    env_file = tmp_path / "model.env"
    env_file.write_text("# a comment\n\nPORT=8000\n")

    result = load_env_file(env_file)

    assert result == {"PORT": "8000"}


def test_load_env_file_strips_quotes(tmp_path: Path):
    env_file = tmp_path / "model.env"
    env_file.write_text('MODEL="Qwen/Qwen3-4B-Instruct-2507"\nNAME=\'support-adapter\'\n')

    result = load_env_file(env_file)

    assert result == {"MODEL": "Qwen/Qwen3-4B-Instruct-2507", "NAME": "support-adapter"}
