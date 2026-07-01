"""Tests for the Milestone 1 repository hygiene scanner.

CPU-only and offline. Detection cases run against temporary directories (the
scanner's filesystem-walk fallback) so they exercise the rules without touching
the real repo. A separate guard scans the *actual* repository and asserts it is
clean, which doubles as a regression alarm if junk or a secret ever gets staged.

Secret-shaped fixtures are assembled by string concatenation so that this test
file's own (version-controlled) source never contains a contiguous credential —
otherwise the repo-clean guard would flag this very file.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

from code_agent_runtime import cli, hygiene

REPO_ROOT = Path(__file__).resolve().parents[1]


def _categories(report: hygiene.HygieneReport) -> set[str]:
    return {f.category for f in report.findings}


def test_clean_directory_has_no_findings(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# ok\n", encoding="utf-8")
    report = hygiene.scan_repo(tmp_path)
    assert report.mode == "filesystem-walk"
    assert report.findings == []
    assert report.ok() is True


def test_detects_each_junk_category(tmp_path: Path) -> None:
    # Directory-level junk.
    (tmp_path / "node_modules" / "leftpad").mkdir(parents=True)
    (tmp_path / "node_modules" / "leftpad" / "index.js").write_text("module.exports=1\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"\x00pyc")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = /usr\n")

    # File-level markers and artifacts.
    (tmp_path / "toolenv").mkdir()
    (tmp_path / "toolenv" / "pyvenv.cfg").write_text("home = /usr\n")
    (tmp_path / "model.safetensors").write_bytes(b"\x00\x00weights")
    (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04zip")
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.local.json").write_text("{}\n")
    results_raw = tmp_path / "results" / "raw"
    results_raw.mkdir(parents=True)
    (results_raw / "dump.dat").write_bytes(b"\x00" * 64)

    report = hygiene.scan_repo(tmp_path)
    cats = _categories(report)
    assert {
        "node_modules",
        "cache",
        "virtualenv",
        "model_weights",
        "result_blob",
        "local_claude_settings",
    } <= cats

    # Directory-junk is reported once, not once per contained file.
    venv_findings = [f for f in report.findings if f.category == "virtualenv"]
    venv_paths = {f.path for f in venv_findings}
    assert ".venv" in venv_paths  # the directory, deduped
    assert "toolenv/pyvenv.cfg" in venv_paths  # the standalone marker
    # Errors (venv, weights, local settings) make the repo "not ok".
    assert report.ok() is False


def test_large_file_threshold(tmp_path: Path) -> None:
    (tmp_path / "big.dat").write_bytes(b"x" * 4096)
    clean = hygiene.scan_repo(tmp_path, max_bytes=10_000)
    assert "large_file" not in _categories(clean)
    flagged = hygiene.scan_repo(tmp_path, max_bytes=1_000)
    assert "large_file" in _categories(flagged)
    # A large file alone is a warning, not an error.
    assert flagged.ok() is True
    assert flagged.ok(strict=True) is False


def test_detects_secret_patterns(tmp_path: Path) -> None:
    aws_body = "AKIA" + "IOSFODNN7EXAMPLE"  # contiguous only at runtime
    (tmp_path / "creds.txt").write_text(f"aws_id={aws_body}\n", encoding="utf-8")

    pem = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIB...\n"
    (tmp_path / "id_rsa").write_text(pem, encoding="utf-8")

    cred_value = "a1B2c3D4e5" + "F6g7H8j9K0"
    (tmp_path / "config.py").write_text(f'api_key = "{cred_value}"\n', encoding="utf-8")

    report = hygiene.scan_repo(tmp_path)
    secret_findings = [f for f in report.findings if f.category == "secret"]
    secret_paths = {f.path for f in secret_findings}
    assert {"creds.txt", "id_rsa", "config.py"} <= secret_paths
    assert all(f.severity == "error" for f in secret_findings)
    assert report.ok() is False


def test_detects_jwt(tmp_path: Path) -> None:
    # A JWT contains exactly two dots, so the generic key=value rule skips it;
    # the dedicated jwt pattern must still catch it. Assemble the token at
    # runtime so this test's own source holds no contiguous credential.
    header = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"  # hygiene: ignore
    payload = "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0"  # hygiene: ignore
    signature = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV" + "_adQssw5c"  # hygiene: ignore
    token = ".".join([header, payload, signature])
    (tmp_path / "app.py").write_text(f'auth_token = "{token}"\n', encoding="utf-8")

    report = hygiene.scan_repo(tmp_path)
    secret_findings = [f for f in report.findings if f.category == "secret"]
    # Exactly the jwt rule fires — not the generic rule (which drops 2-dot values).
    assert any("jwt" in f.message for f in secret_findings)
    assert all(f.severity == "error" for f in secret_findings)
    assert report.ok() is False


def test_detects_high_entropy_base64_blob(tmp_path: Path) -> None:
    # Built at runtime so no high-entropy literal lands in this file's source.
    # Assigned to a NON-credential variable to prove the backstop is not scoped
    # to credential-shaped key names.
    blob = (
        base64.b64encode(hashlib.sha256(b"seed-a").digest() + hashlib.sha256(b"seed-b").digest())
        .decode()
        .rstrip("=")
    )
    (tmp_path / "app.py").write_text(f'session = "{blob}"\n', encoding="utf-8")
    report = hygiene.scan_repo(tmp_path)
    hits = [f for f in report.findings if f.category == "high_entropy_string"]
    assert hits and hits[0].severity == "error"
    assert report.ok() is False


def test_detects_high_entropy_hex_blob(tmp_path: Path) -> None:
    # A pure-hex secret sits near prose entropy (~3.8), so shape does the work.
    blob = hashlib.sha256(b"seed-hex").hexdigest()  # 64 hex chars
    (tmp_path / "token.txt").write_text(f"{blob}\n", encoding="utf-8")
    assert "high_entropy_string" in _categories(hygiene.scan_repo(tmp_path))


def test_entropy_backstop_can_be_disabled(tmp_path: Path) -> None:
    blob = base64.b64encode(hashlib.sha256(b"x").digest() * 2).decode().rstrip("=")
    (tmp_path / "a.py").write_text(f'v = "{blob}"\n', encoding="utf-8")
    assert "high_entropy_string" in _categories(hygiene.scan_repo(tmp_path))
    assert "high_entropy_string" not in _categories(hygiene.scan_repo(tmp_path, entropy=False))


def test_low_entropy_long_strings_not_flagged(tmp_path: Path) -> None:
    # Long identifiers, slash-joined paths, and degenerate runs are not secrets.
    (tmp_path / "notes.txt").write_text(
        "path = /usr/local/share/app/config/settings/deep/enough/here\n"
        "ident = RawDescriptionHelpFormatterAbstractionLayer\n"
        "repeat = " + "a" * 60 + "\n"
        "zeros = " + "0" * 64 + "\n",  # long, hex-shaped, but zero entropy
        encoding="utf-8",
    )
    assert "high_entropy_string" not in _categories(hygiene.scan_repo(tmp_path))


def test_entropy_finding_respects_allow_marker(tmp_path: Path) -> None:
    blob = base64.b64encode(hashlib.sha256(b"marked").digest() * 2).decode().rstrip("=")
    (tmp_path / "a.py").write_text(f'v = "{blob}"  # {hygiene.ALLOW_MARKER}\n', encoding="utf-8")
    assert "high_entropy_string" not in _categories(hygiene.scan_repo(tmp_path))


def test_placeholders_are_not_flagged_as_secrets(tmp_path: Path) -> None:
    (tmp_path / "example.env").write_text(
        'api_key = "your-key-placeholder-here"\npassword = "changeme"\n',
        encoding="utf-8",
    )
    report = hygiene.scan_repo(tmp_path)
    assert "secret" not in _categories(report)


def test_allow_marker_suppresses_secret(tmp_path: Path) -> None:
    aws_body = "AKIA" + "IOSFODNN7EXAMPLE"
    (tmp_path / "doc.md").write_text(
        f"example key {aws_body}  # {hygiene.ALLOW_MARKER}\n", encoding="utf-8"
    )
    report = hygiene.scan_repo(tmp_path)
    assert "secret" not in _categories(report)


def test_report_to_dict_is_serializable(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home=/usr\n")
    payload = hygiene.scan_repo(tmp_path).to_dict()
    assert payload["mode"] == "filesystem-walk"
    assert payload["errors"] >= 1
    assert isinstance(payload["findings"], list)


def test_real_repo_is_clean_under_git() -> None:
    report = hygiene.scan_repo(REPO_ROOT)
    assert report.mode == "git-tracked"
    assert report.scanned > 0
    # No error-severity findings should ever be committed to this repo.
    assert report.errors == [], f"unexpected hygiene errors: {report.errors}"
    assert report.ok() is True


def test_cli_hygiene_subcommand_on_clean_repo(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["hygiene", "--root", str(REPO_ROOT)])
    assert rc == 0
    assert "Repository hygiene scan" in capsys.readouterr().out
