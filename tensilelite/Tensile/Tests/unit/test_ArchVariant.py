import tempfile
import pytest

from typing import NamedTuple, Optional, Set
from unittest.mock import patch
from pathlib import Path

from Tensile.Com.ArchVariant import (
    ArchVariant,
    extractArchVariant,
    matchArchVariant,
    LogicFileError,
)

class MockArchVariant(NamedTuple):
    Name: str
    Gfx: str
    DeviceIds: Optional[Set[str]]
    CUCount: Optional[str] 


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
            Name="some_arch_name", Gfx="gfx942", DeviceIds={"id=1234", "id=5678"}, CUCount="cu=64"
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
            Name="some_arch_name", Gfx="gfx942", DeviceIds={"id=1234", "id=5678"}, CUCount=None
        )

def test_extractArchVariant_success3():
    content = """- {MinimumRequiredVersion: 4.33.0}
- some_arch_name 
- gfx942
- [All devices]
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content.encode())
        tmp.flush()
        result = extractArchVariant(tmp.name)
        assert result == ArchVariant(
            Name="some_arch_name", Gfx="gfx942", DeviceIds=None, CUCount=None
        )

def test_extractArchVariant_success4():
    content = """- {MinimumRequiredVersion: 4.33.0}
- some_arch_name 
- {Architecture: gfx942, CUCount: 98}
- [All devices]
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content.encode())
        tmp.flush()
        result = extractArchVariant(tmp.name)
        assert result == ArchVariant(
            Name="some_arch_name", Gfx="gfx942", DeviceIds=None, CUCount="cu=98"
        )

def test_extractArchVariant_success5():
    content = """- {MinimumRequiredVersion: 4.33.0}
- some_arch_name 
- gfx942
- [Device 0050]
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content.encode())
        tmp.flush()
        result = extractArchVariant(tmp.name)
        assert result == ArchVariant(
            Name="some_arch_name", Gfx="gfx942", DeviceIds=None, CUCount=None
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


# def test_parseArchVariantString_success():
#     # Valid specification with device IDs and CU counts
#     spec = {"id=1234","cu=64","id=5678","cu=32"}
#     deviceIds, cuCounts = parseArchVariantString(spec)
#     assert deviceIds == {"1234", "5678"}
#     assert cuCounts == {64, 32}

#     # Valid specification with only device IDs
#     spec = {"id=1234","id=5678"}
#     deviceIds, cuCounts = parseArchVariantString(spec)
#     assert deviceIds == {"1234", "5678"}
#     assert cuCounts == set()

#     # Valid specification with only CU counts
#     spec = {"cu=64","cu=32"}
#     deviceIds, cuCounts = parseArchVariantString(spec)
#     assert deviceIds == set()
#     assert cuCounts == {64, 32}

#     spec = {"id=abcd"}
#     deviceIds, cuCounts = parseArchVariantString(spec)
#     assert deviceIds == {"abcd"}
#     assert cuCounts == set()

#     # Empty specification
#     spec = {}
#     deviceIds, cuCounts = parseArchVariantString(spec)
#     assert deviceIds == set()
#     assert cuCounts == set()

#     spec = ""
#     deviceIds, cuCounts = parseArchVariantString(spec)
#     assert deviceIds == set()
#     assert cuCounts == set()


# def test_parseArchVariantString_failure():
#     # Invalid specification format
#     spec = {"id=1234", "invalid=64"}
#     with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
#         parseArchVariantString(spec)

#     spec = "id=1234;invalid=64"
#     with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
#         parseArchVariantString(spec)

#     spec = {"id=12345"}
#     with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
#         parseArchVariantString(spec)

#     spec = "id=12345"
#     with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
#         parseArchVariantString(spec)

#     spec = {"cu=ab"}
#     with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
#         parseArchVariantString(spec)

#     spec = "cu=ab"
#     with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
#         parseArchVariantString(spec)

#     spec = {"cu=-12"}
#     with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
#         parseArchVariantString(spec)

#     spec = "cu=-12"
#     with pytest.raises(ValueError, match=r"Invalid architecture variant string(.*)"):
#         parseArchVariantString(spec)

@patch("Tensile.Com.ArchVariant.extractArchVariant")
def test_matchArchVariant_success(mock_extract):
    targetFile = Path.cwd() / "valid_file.txt"

    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"id=5678"}, "cu=64")
    variantMap = {"gfx942": {"id=1234": set(), "id=5678": {targetFile.name}, "cu=64": set()}} 
    assert matchArchVariant(variantMap, targetFile) == True
    assert variantMap == {"gfx942": {"id=1234": set(), "id=5678": {targetFile.name}, "cu=64": {targetFile.name}}}

    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"id=1234", "id=5678"}, None)
    variantMap = {"gfx942": {"id=1234": set(), "id=5678": set(), "cu=64": set()}} 
    assert matchArchVariant(variantMap, targetFile) == True
    assert variantMap == {"gfx942": {"id=1234": {targetFile.name}, "id=5678": {targetFile.name}, "cu=64": set()}}

    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"id=1234", "id=5678"}, None)
    variantMap = {"gfx942": {"id=1234": set()}}
    assert matchArchVariant(variantMap, targetFile) == True
    assert variantMap == {"gfx942": {"id=1234": {targetFile.name}}}

    mock_extract.return_value = MockArchVariant("foo", "gfx942", None, "cu=64")
    variantMap = {"gfx942": {"id=1234": set(), "cu=64": set()}}
    assert matchArchVariant(variantMap, targetFile) == True
    assert variantMap == {"gfx942": {"id=1234": set(), "cu=64": {targetFile.name}}}


@patch("Tensile.Com.ArchVariant.extractArchVariant")
def test_matchArchVariant_failure(mock_extract):
    targetFile = Path.cwd() / "valid_file.txt"

    # Gfx name doesn't match
    mock_extract.return_value = MockArchVariant("foo", "gfx90a", {"id=1234", "id=5678"}, "cu=64")
    variantMap = {"gfx942": {"id=1234": set(), "cu=64": set()}}
    assert matchArchVariant(variantMap, targetFile) == False
    assert variantMap == {"gfx942": {"id=1234": set(), "cu=64": set()}}

    # Device ID doesn't match
    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"id=1234", "id=5678"}, None)
    variantMap = {"gfx942": {"id=9999": set(), "cu=64": set()}}
    assert matchArchVariant(variantMap, targetFile) == False
    assert variantMap == {"gfx942": {"id=9999": set(), "cu=64": set()}}

    # CU count doesn't match
    mock_extract.return_value = MockArchVariant("foo", "gfx942", None, "cu=64")
    variantMap = {"gfx942": {"id=1234": set(), "cu=32": set()}}
    assert matchArchVariant(variantMap, targetFile) == False
    assert variantMap == {"gfx942": {"id=1234": set(), "cu=32": set()}}

    # Extracted variant already exists on variantMap
    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"id=1234", "id=5678"}, None)
    variantMap = {"gfx942": {"id=5678": {targetFile.name}}} 
    assert matchArchVariant(variantMap, targetFile) == False
    assert variantMap == {"gfx942": {"id=5678": {targetFile.name}}}

    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"id=1234", "id=5678"}, "cu=72")
    variantMap = {"gfx942": {"id=5678": {targetFile.name}, "cu=72": {targetFile.name}}} 
    assert matchArchVariant(variantMap, targetFile) == False
    assert variantMap == {"gfx942": {"id=5678": {targetFile.name}, "cu=72": {targetFile.name}}}

    assert matchArchVariant({"gfx942": {"id=1234": set(), "cu=32": set()}}, Path.cwd()/"experimental"/"file.txt") == False


@patch("Tensile.Com.ArchVariant.extractArchVariant")
def test_matchArchVariant_fallback(mock_extract):
    targetFile = Path.cwd() / "valid_file.txt"

    # Fallback file (CUCount and DeviceIds are None)
    mock_extract.return_value = MockArchVariant("foo", "gfx942", None, None)
    variantMap = {"gfx942": {"id=1234": set(), "cu=64": set()}}
    assert matchArchVariant(variantMap, targetFile) == True
    assert variantMap == {"gfx942": {"id=1234": set(), "cu=64": set()}}

    # Fallback file already in one variant but not another
    mock_extract.return_value = MockArchVariant("foo", "gfx942", None, None)
    variantMap = {"gfx942": {"id=1234": {targetFile.name}, "cu=64": set()}}
    assert matchArchVariant(variantMap, targetFile) == True
    assert variantMap == {"gfx942": {"id=1234": {targetFile.name}, "cu=64": set()}}

    # Fallback file already in all variantMap entries
    mock_extract.return_value = MockArchVariant("foo", "gfx942", None, None)
    variantMap = {"gfx942": {"id=1234": {targetFile.name}, "cu=64": {targetFile.name}}}
    assert matchArchVariant(variantMap, targetFile) == False
    assert variantMap == {"gfx942": {"id=1234": {targetFile.name}, "cu=64": {targetFile.name}}}


@patch("Tensile.Com.ArchVariant.extractArchVariant")
def test_matchArchVariant_partial_match(mock_extract):
    targetFile1 = Path.cwd() / "valid_file1.txt"
    targetFile2 = Path.cwd() / "valid_file2.txt"

    # Matching GFX and CU Count, Device ID is None
    mock_extract.return_value = MockArchVariant("foo", "gfx942", None, "cu=64")
    variantMap = {"gfx942": {"id=1234": set(), "cu=64": {targetFile1.name}}}
    assert matchArchVariant(variantMap, targetFile2) == True
    assert variantMap == {"gfx942": {"id=1234": set(), "cu=64": {targetFile1.name, targetFile2.name}}}

    # Matching GFX and CU Count, Device ID is None
    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"id=1234"}, "cu=64")
    variantMap = {"gfx942": {"id=1234": set(), "cu=64": {targetFile1.name}}}
    assert matchArchVariant(variantMap, targetFile2) == True
    assert variantMap == {"gfx942": {"id=1234": {targetFile2.name}, "cu=64": {targetFile1.name, targetFile2.name}}}

    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"id=1234"}, "cu=64")
    variantMap = {"gfx942": {"id=1234": {targetFile1.name}, "id=abcd": set(), "cu=64": {targetFile1.name}}}
    assert matchArchVariant(variantMap, targetFile1) == False
    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"id=abcd"}, "cu=64")
    assert matchArchVariant(variantMap, targetFile2) == True
    assert variantMap == {"gfx942": {"id=1234": {targetFile1.name}, "id=abcd": {targetFile2.name}, "cu=64": {targetFile1.name, targetFile2.name}}}

    # Matching GFX and Device ID, CU Count is None
    mock_extract.return_value = MockArchVariant("foo", "gfx942", {"id=1234"}, None)
    variantMap = {"gfx942": {"id=1234": set(), "cu=64": set()}}
    assert matchArchVariant(variantMap, targetFile1) == True
    assert variantMap == {"gfx942": {"id=1234": {targetFile1.name}, "cu=64": set()}}
    assert matchArchVariant(variantMap, targetFile1) == False
    assert variantMap == {"gfx942": {"id=1234": {targetFile1.name}, "cu=64": set()}}