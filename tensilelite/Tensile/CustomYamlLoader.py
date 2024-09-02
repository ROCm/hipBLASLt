import yaml
from pathlib import Path

def parse_general(loader: yaml.Loader):
    if loader.check_event(yaml.MappingStartEvent):
        return parse_mapping(loader)
    elif loader.check_event(yaml.SequenceStartEvent):
        return parse_sequence(loader)
    elif loader.check_event(yaml.ScalarEvent):
        return parse_scalar(loader)

def parse_sequence(loader: yaml.Loader):
    ret = []
    #pop sequence start event
    loader.get_event()
    while not loader.check_event(yaml.SequenceEndEvent):
        ret.append(parse_general(loader))
    #pop sequence end event
    loader.get_event()
    return ret

def parse_mapping(loader: yaml.Loader):
    ret = {}
    k, v = None, None
    #pop mapping start event
    loader.get_event()
    while not loader.check_event(yaml.MappingEndEvent):
        if k is None:
            k = parse_scalar(loader)
        elif v is None:
            v = parse_general(loader)
            ret[k] = v
            k, v = None, None

    #pop mapping end event
    loader.get_event()
    return ret

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_scalar(loader: yaml.Loader):
    assert loader.check_event(yaml.ScalarEvent)
    value: str = loader.get_event().value
    value_lower: str = value.lower()

    if value_lower in ('true', 'yes',):
        return True
    elif value_lower in ('false', 'no',):
        return False
    elif value_lower in ('null',):
        return None
    elif value_lower.lstrip('+-').isnumeric():
        return int(value_lower)
    elif is_float(value_lower):
        return float(value_lower)

    return value

def load_yaml_stream(yaml_path: Path, loader_type):
    with open(yaml_path, 'r') as f:
        loader = loader_type(f)
        assert loader.check_event(yaml.StreamStartEvent)
        loader.get_event()
        assert loader.check_event(yaml.DocumentStartEvent)
        loader.get_event()

        # assume the root element is a sequence
        assert loader.check_event(yaml.SequenceStartEvent)
        loader.get_event()
        logic = []

        # now while the next event does not end the sequence, process each item
        while not loader.check_event(yaml.SequenceEndEvent):
            logic.append(parse_general(loader))

        # assume document ends and no further documents are in stream
        loader.get_event()
        assert loader.check_event(yaml.DocumentEndEvent)
        loader.get_event()
        assert loader.check_event(yaml.StreamEndEvent)
        return logic
