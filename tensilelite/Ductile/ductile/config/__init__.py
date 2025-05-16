from pydantic.v1.utils import deep_update
from types import MappingProxyType

import yaml
import os


def update(cfg):
    defaults = dict(DEFAULTS)
    return deep_update(defaults, cfg)


def load(path=None):
    if not path:
        return yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "defaults.yaml")))
    cfg = yaml.safe_load(open(path))
    return update(cfg)


def populate(conf, name):
    section = conf[name]
    if "name" not in conf[name]:
        raise ValueError(f"missing 'name' field for section '{name}'")
    sel = conf[name]["name"]
    res = {"name": sel}
    if sel in section:
        res = res | section[sel]
    if "common" in section:
        res = res | section["common"]
    return res


DEFAULTS = MappingProxyType(load())
