import tempfile
import pytest

from typing import NamedTuple, Optional, Set
from unittest.mock import patch

from Tensile.Com.ArchVariant import (
    ArchVariant,
    extractArchVariant,
    parseArchVariantString,
    matchArchVariant,
    LogicFileError,
)

class MockArchVariant(NamedTuple):
    Name: str
    Gfx: str
    DeviceIds: Optional[Set[str]]
    CUCount: Optional[int] 


def test_extractArchVariant_success1():
    content = """- {MinimumRequiredVersion: 4.33.0}
- some_arch_name 
- {Architecture: gfx942, CUCount: 64}
- [Device 1234, Device 5678]
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content.encode())
        tmp.flush()
        result = extractArchVariant(tmp.name)
        assert result == ArchVariant(
            Name="some_arch_name", Gfx="gfx942", DeviceIds={"1234", "5678"}, CUCount=64
        )


def test_extractArchVariant_success2():
    content = """- {MinimumRequiredVersion: 4.33.0}
- some_arch_name 
- gfx942
- [Device 1234, Device 5678]
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content.encode())
        tmp.flush()
        result = extractArchVariant(tmp.name)
        assert result == ArchVariant(
            Name="some_arch_name", Gfx="gfx942", DeviceIds={"1234", "5678"}, CUCount=None
        )


def test_extractArchVariant_missing_first_line():
    content = """
- some_arch_name 
- gfx942
- [Device 1234, Device 5678]
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content.encode())
        tmp.flush()
        with pytest.raises(LogicFileError, match=r"(.*)Expected minimum required version(.*)"):
            extractArchVariant(tmp.name)


def test_extractArchVariant_missing_architecture():
    content = """- {MinimumRequiredVersion: 4.33.0}
- some_arch_name 
- [Device 1234, Device 5678]
- Activation: false
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content.encode())
        tmp.flush()
        with pytest.raises(LogicFileError, match=r"(.*)Expected architecture and CU count(.*)"):
            extractArchVariant(tmp.name)


def test_extractArchVariant_incorrect_arch_format():
    content = """- {MinimumRequiredVersion: 4.33.0}
- some_arch_name 
- {Arch: gfx942, CUCount: 64}
- [Device 1234, Device 5678]
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content.encode())
        tmp.flush()
        with pytest.raises(LogicFileError, match=r"(.*)Expected architecture and CU count(.*)"):
            extractArchVariant(tmp.name)


def test_extractArchVariant_missing_CUCount():
    content = """- {MinimumRequiredVersion: 4.33.0}
- some_arch_name 
- {Architecture: gfx942}
- [Device 1234, Device 5678]
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content.encode())
        tmp.flush()
        with pytest.raises(LogicFileError, match=r"(.*)Expected architecture and CU count(.*)"):
            extractArchVariant(tmp.name)


def test_extractArchVariant_incorrect_device_ids_format():
    content = """- {MinimumRequiredVersion: 4.33.0}
- some_arch_name 
- {Architecture: gfx942, CUCount: 64}
- Device 1234, Device 5678
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content.encode())
        tmp.flush()
        with pytest.raises(LogicFileError, match=r"(.*)No device IDs found(.*)"):
            extractArchVariant(tmp.name)


def test_extractArchVariant_missing_device_ids():
    content = """- {MinimumRequiredVersion: 4.33.0}
- some_arch_name 
- {Architecture: gfx942, CUCount: 64}
- Activation: false
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content.encode())
        tmp.flush()
        with pytest.raises(LogicFileError):
            extractArchVariant(tmp.name)


def test_extractArchVariant_file_not_found():
    non_existent_file = "non_existent_file.txt"
    with pytest.raises(FileNotFoundError):
        extractArchVariant(non_existent_file)


def test_extractArchVariant_empty_file():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"")
        tmp.flush()
        with pytest.raises(LogicFileError):
            extractArchVariant(tmp.name)


def test_parseArchVariantString_success():
    # Valid specification with device IDs and CU counts
    spec = "id=1234;cu=64;id=5678;cu=32"
    deviceIds, cuCounts = parseArchVariantString(spec)
    assert deviceIds == {"1234", "5678"}
    assert cuCounts == {64, 32}

    # Valid specification with only device IDs
    spec = "id=1234;id=5678"
    deviceIds, cuCounts = parseArchVariantString(spec)
    assert deviceIds == {"1234", "5678"}
    assert cuCounts == set()

    # Valid specification with only CU counts
    spec = "cu=64;cu=32"
    deviceIds, cuCounts = parseArchVariantString(spec)
    assert deviceIds == set()
    assert cuCounts == {64, 32}

    # Empty specification string
    spec = ""
    deviceIds, cuCounts = parseArchVariantString(spec)
    assert deviceIds == set()
    assert cuCounts == set()


def test_parseArchVariantString_failure():
    # Invalid specification format
    spec = "id=1234;invalid=64"
    with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
        parseArchVariantString(spec)

    spec = "id=abcd;cd=ef"
    with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
        parseArchVariantString(spec)

    spec = "id=12345"
    with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
        parseArchVariantString(spec)

    spec = "cu=-12"
    with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
        parseArchVariantString(spec)

@patch("Tensile.Com.ArchVariant.extractArchVariant")
def test_matchArchVariant_success(mock_extract):
    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"1234", "5678"}, 64)
    assert matchArchVariant({"gfx942"}, {"1234"}, {64}, "valid_file.txt") == True

    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"1234", "5678"}, None)
    assert matchArchVariant({"gfx942"}, {"1234"}, {64}, "valid_file.txt") == True

    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"1234", "5678"}, None)
    assert matchArchVariant({"gfx942"}, {"1234"}, set(), "valid_file.txt") == True

@patch("Tensile.Com.ArchVariant.extractArchVariant")
def test_matchArchVariant_failure(mock_extract):
    # Gfx name doesn't match
    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"1234", "5678"}, 64)
    assert matchArchVariant({"gfx906"}, {"1234"}, {64}, "valid_file.txt") == False

    # Device ID doesn't match
    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"1234", "5678"}, 64)
    assert matchArchVariant({"gfx942"}, {"9999"}, {64}, "valid_file.txt") == False

    # CU count doesn't match
    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"1234", "5678"}, 64)
    assert matchArchVariant({"gfx942"}, {"1234"}, {32}, "valid_file.txt") == False

    assert matchArchVariant({"gfx942"}, {"1234"}, {64}, "experimental/file.txt") == False