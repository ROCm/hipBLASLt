import pytest

from Tensile.Common import splitArchsFromGlobal

def test_splitArchsFromGlobal_all_archs():
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
    assert variants == set()

def test_splitArchsFromGlobal_specific_archs():
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
    assert variants == set()

def test_splitArchsFromGlobal_with_variants0():
    globalParameters = {
        "Architecture": "gfx906[variant1;variant2];gfx908[variant3]",
        "AsmCaps": {
            "gfx906": {"SupportedISA": True, "SupportedSource": True},
            "gfx908": {"SupportedISA": True, "SupportedSource": True},
        },
        "SupportedISA": ["gfx906", "gfx908"]
    }

    with pytest.raises(ValueError, match="Architecture gfx906\\[variant1 not supported"):
        splitArchsFromGlobal(globalParameters)

def test_splitArchsFromGlobal_with_variants1():
    globalParameters = {
        "Architecture": "gfx906[variant1];gfx908[variant2]",
        "AsmCaps": {
            "gfx906": {"SupportedISA": True, "SupportedSource": True},
            "gfx908": {"SupportedISA": True, "SupportedSource": True},
        },
        "SupportedISA": ["gfx906", "gfx908"]
    }

    gfxArchs, cmdlineArchs, variants = splitArchsFromGlobal(globalParameters)
    assert gfxArchs == {"gfx906", "gfx908"}
    assert cmdlineArchs == {"gfx906", "gfx908"}
    assert variants == {"variant1", "variant2"}

def test_splitArchsFromGlobal_with_variants2():
    globalParameters = {
        "Architecture": "gfx90a:xnack-[variant1,variant2];gfx942[variant3]",
        "AsmCaps": {
            "gfx90a:xnack-": {"SupportedISA": True, "SupportedSource": True},
            "gfx942": {"SupportedISA": True, "SupportedSource": True},
        },
        "SupportedISA": ["gfx90a:xnack-", "gfx942"]
    }

    gfxArchs, cmdlineArchs, variants = splitArchsFromGlobal(globalParameters)
    assert gfxArchs == {"gfx90a-xnack-", "gfx942"}
    assert cmdlineArchs == {"gfx90a:xnack-", "gfx942"}
    assert variants == {"variant1", "variant2", "variant3"}

def test_splitArchsFromGlobal_unsupported_arch():
    globalParameters = {
        "Architecture": "gfxUnsupported",
        "AsmCaps": {},
        "SupportedISA": []
    }

    with pytest.raises(ValueError, match="Architecture gfxUnsupported not supported"):
        splitArchsFromGlobal(globalParameters)