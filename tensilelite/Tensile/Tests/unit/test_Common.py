import pytest

from Tensile.Common import splitArchsFromGlobal

def test_splitArchsFromGlobal_allArchs():
    globalParameters = {
        "Architecture": "all",
        "AsmCaps": {
            (9,0,6): {"SupportedISA": True, "SupportedSource": True},
            (9,0,8): {"SupportedISA": True, "SupportedSource": True},
            (9,0,10): {"SupportedISA": True, "SupportedSource": True},
            (9,4,0): {"SupportedISA": True, "SupportedSource": True},
            (9,4,1): {"SupportedISA": True, "SupportedSource": True},
            (9,4,2): {"SupportedISA": True, "SupportedSource": True},
        },
        "SupportedISA": [(9,0,6), (9,0,8), (9,0,10), (9,4,0), (9,4,1), (9,4,2)]
    }

    gfxArchs, cmdlineArchs, variants = splitArchsFromGlobal(globalParameters)
    assert gfxArchs == {"gfx906-xnack-", "gfx908-xnack-", "gfx90a-xnack+", "gfx90a-xnack-", "gfx940-xnack-", "gfx941-xnack-", "gfx942-xnack-"}
    assert cmdlineArchs == {"gfx906:xnack-", "gfx908:xnack-", "gfx90a:xnack+", "gfx90a:xnack-", "gfx940:xnack-", "gfx941:xnack-", "gfx942:xnack-"}
    assert variants == dict()

def test_splitArchsFromGlobal_specificArchs():
    globalParameters = {
        "Architecture": "gfx906;gfx908",
        "AsmCaps": {
            "gfx906": {"SupportedISA": True, "SupportedSource": True},
            "gfx908": {"SupportedISA": True, "SupportedSource": True},
        },
        "SupportedISA": ["gfx906", "gfx908"]
    }

    gfxArchs, cmdlineArchs, variants = splitArchsFromGlobal(globalParameters)
    assert gfxArchs == {"gfx906", "gfx908"}
    assert cmdlineArchs == {"gfx906", "gfx908"}
    assert variants == dict()

def test_splitArchsFromGlobal_withVariantsBadDelimiter():
    globalParameters = {
        "Architecture": "gfx906[id=abcd;id=dcba];gfx908[cu=40]",  # should use comma in arch variant string
        "AsmCaps": {
            "gfx906": {"SupportedISA": True, "SupportedSource": True},
            "gfx908": {"SupportedISA": True, "SupportedSource": True},
        },
        "SupportedISA": ["gfx906", "gfx908"]
    }

    # Fails because of semicolon in variant string
    with pytest.raises(ValueError, match="Architecture gfx906\\[id=abcd not supported"):
        splitArchsFromGlobal(globalParameters)

def test_splitArchsFromGlobal_withInvalidVariants():
    globalParameters = {
        "Architecture": "gfx906[variant1]",  # must contain either 'id=' or 'cu='
        "AsmCaps": {
            "gfx906": {"SupportedISA": True, "SupportedSource": True},
            "gfx908": {"SupportedISA": True, "SupportedSource": True},
        },
        "SupportedISA": ["gfx906", "gfx908"]
    }

    with pytest.raises(ValueError, match=r"Invalid architecture variant(.*)"):
        splitArchsFromGlobal(globalParameters)

    globalParameters["Architecture"] = "gfx906[id=12345]"  # id must be 4 chars in length
    with pytest.raises(ValueError, match=r"Invalid architecture variant(.*)"):
        splitArchsFromGlobal(globalParameters)

    globalParameters["Architecture"] = "gfx906[id=1234,cu=QW]"  # cu must contain only digits
    with pytest.raises(ValueError, match=r"Invalid architecture variant(.*)"):
        splitArchsFromGlobal(globalParameters)


def test_splitArchsFromGlobal_withVariants1():
    globalParameters = {
        "Architecture": "gfx906[id=abcd,id=CDEF];gfx908[cu=999]",
        "AsmCaps": {
            "gfx906": {"SupportedISA": True, "SupportedSource": True},
            "gfx908": {"SupportedISA": True, "SupportedSource": True},
        },
        "SupportedISA": ["gfx906", "gfx908"]
    }

    gfxArchs, cmdlineArchs, variants = splitArchsFromGlobal(globalParameters)
    assert gfxArchs == {"gfx906", "gfx908"}
    assert cmdlineArchs == {"gfx906", "gfx908"}
    assert variants == {"gfx906": ["id=abcd", "id=cdef"], "gfx908": ["cu=999"]}

def test_splitArchsFromGlobal_withSpacesInVariants():
    globalParameters = {
        "Architecture": "gfx90a:xnack-[ id=74a5  ,   cu=42 ];gfx942[ id = 5432, cu=4]",
        "AsmCaps": {
            "gfx90a:xnack-": {"SupportedISA": True, "SupportedSource": True},
            "gfx942": {"SupportedISA": True, "SupportedSource": True},
        },
        "SupportedISA": ["gfx90a:xnack-", "gfx942"]
    }

    gfxArchs, cmdlineArchs, variants = splitArchsFromGlobal(globalParameters)
    assert gfxArchs == {"gfx90a-xnack-", "gfx942"}
    assert cmdlineArchs == {"gfx90a:xnack-", "gfx942"}
    assert variants == {"gfx90a:xnack-": ["id=74a5", "cu=42"], "gfx942": ["id=5432", "cu=4"]}

def test_splitArchsFromGlobal_unsupportedArch():
    globalParameters = {
        "Architecture": "gfxUnsupported",
        "AsmCaps": {},
        "SupportedISA": []
    }

    with pytest.raises(ValueError, match="Architecture gfxUnsupported not supported"):
        splitArchsFromGlobal(globalParameters)
