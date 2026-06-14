import pytest

from cli import main


def test_cli_prints_translation(capsys):
    rc = main(["--direction", "en-ng", "Hello"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "<meyabase/en-ng-translation>::Hello"


def test_cli_invalid_direction_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--direction", "zz-yy", "Hello"])
    assert exc.value.code == 2  # argparse choices error


def test_cli_requires_direction(capsys):
    with pytest.raises(SystemExit):
        main(["Hello"])
