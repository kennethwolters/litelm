import pytest

from scripts.ported_contract import load_contract_nodes


def test_load_contract_nodes_ignores_comments_and_blank_lines(tmp_path):
    manifest = tmp_path / "contract.txt"
    manifest.write_text(
        "# public contract\n\n"
        "tests/ported/test_one.py::test_one\n"
        "tests/ported/test_two.py::TestThing::test_two[param]\n"
    )
    (tmp_path / "tests/ported").mkdir(parents=True)
    (tmp_path / "tests/ported/test_one.py").touch()
    (tmp_path / "tests/ported/test_two.py").touch()

    assert load_contract_nodes(manifest, root=tmp_path) == [
        "tests/ported/test_one.py::test_one",
        "tests/ported/test_two.py::TestThing::test_two[param]",
    ]


def test_load_contract_nodes_rejects_duplicates(tmp_path):
    manifest = tmp_path / "contract.txt"
    manifest.write_text("tests/ported/test_one.py::test_one\n" * 2)
    (tmp_path / "tests/ported").mkdir(parents=True)
    (tmp_path / "tests/ported/test_one.py").touch()

    with pytest.raises(ValueError, match="duplicate"):
        load_contract_nodes(manifest, root=tmp_path)


@pytest.mark.parametrize(
    "node",
    [
        "tests/test_one.py::test_one",
        "tests/ported/test_one.py",
        "../tests/ported/test_one.py::test_one",
    ],
)
def test_load_contract_nodes_rejects_invalid_nodes(tmp_path, node):
    manifest = tmp_path / "contract.txt"
    manifest.write_text(node + "\n")

    with pytest.raises(ValueError, match="invalid contract node"):
        load_contract_nodes(manifest, root=tmp_path)


def test_load_contract_nodes_reports_removed_upstream_file(tmp_path):
    manifest = tmp_path / "contract.txt"
    manifest.write_text("tests/ported/missing.py::test_one\n")

    with pytest.raises(FileNotFoundError, match="missing.py"):
        load_contract_nodes(manifest, root=tmp_path)
