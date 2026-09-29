"""End-to-end CLI tests, fully offline (demo provider / fake-free paths)."""

import json

import pytest

from src.csv_loader import load_inquiries
from src.main import EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR, EXIT_OK, main
from tests.conftest import SAMPLE_CSV


@pytest.fixture(autouse=True)
def isolate_home(tmp_path, monkeypatch):
    """Run from a scratch cwd so a developer's real .env can never leak into tests."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AI_INQUIRY_PROVIDER", raising=False)


class TestDemoMode:
    def test_full_pipeline_without_api_key(self, tmp_path, capsys):
        exit_code = main(
            [
                "--demo",
                "--input", str(SAMPLE_CSV),
                "--output-dir", str(tmp_path),
                "--log-level", "ERROR",
            ]
        )

        assert exit_code == EXIT_OK
        report = json.loads((tmp_path / "inquiries_report.json").read_text(encoding="utf-8"))
        assert report["summary"]["total"] == len(load_inquiries(SAMPLE_CSV))
        assert report["summary"]["succeeded"] == report["summary"]["total"]
        assert all(row["status"] == "success" for row in report["results"])
        assert report["report_metadata"]["provider"] == "mock"

        assert (tmp_path / "inquiries_report.csv").exists()
        stdout = capsys.readouterr().out
        assert "succeeded" in stdout
        assert "report:" in stdout

    def test_single_format_csv(self, tmp_path):
        exit_code = main(
            ["--demo", "--input", str(SAMPLE_CSV), "--output-dir", str(tmp_path),
             "--format", "csv", "--log-level", "ERROR"]
        )
        assert exit_code == EXIT_OK
        assert (tmp_path / "inquiries_report.csv").exists()
        assert not (tmp_path / "inquiries_report.json").exists()


class TestWithoutApiKey:
    def test_real_provider_without_key_returns_config_error(self, capsys):
        exit_code = main(["--input", str(SAMPLE_CSV), "--output-dir", "out", "--log-level", "ERROR"])
        assert exit_code == EXIT_CONFIG_ERROR
        assert "--demo" in capsys.readouterr().err

    def test_help_works_without_api_key(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        assert "--demo" in capsys.readouterr().out

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert capsys.readouterr().out.strip().endswith("1.0.0")


class TestInputErrors:
    def test_missing_input_file(self, tmp_path):
        exit_code = main(
            ["--demo", "--input", str(tmp_path / "missing.csv"),
             "--output-dir", str(tmp_path), "--log-level", "ERROR"]
        )
        assert exit_code == EXIT_INPUT_ERROR

    def test_invalid_header(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("name,note\nAlice,hi\n", encoding="utf-8")
        exit_code = main(
            ["--demo", "--input", str(bad),
             "--output-dir", str(tmp_path / "out"), "--log-level", "ERROR"]
        )
        assert exit_code == EXIT_INPUT_ERROR
