"""A run's Config: a dataclass of dataclass groups, one CLI flag per field. Shared by src.rnn and src.transformer,
each with its own Config class."""
import argparse
from dataclasses import asdict, fields, is_dataclass


def config_from_dict(values, cls):
    """Config of a checkpoint (asdict): fields it lacks take their defaults, fields since removed are dropped."""
    known = {f.name: f for f in fields(cls)}
    return cls(**{k: config_from_dict(v, known[k].type) if is_dataclass(known[k].type) else v
                  for k, v in values.items() if k in known})


def dotted_fields(cfg):
    """(`model.hidden_size`, value) for every field of the Config and of its groups: the CLI flags and run names."""
    for key, value in asdict(cfg).items():
        if isinstance(value, dict):
            yield from ((f"{key}.{k}", v) for k, v in value.items())
        else:
            yield key, value


def changed_fields(cfg, skip=()):
    """`hidden_size=512` for every field off its default, but those in `skip`: the run name."""
    default = dict(dotted_fields(type(cfg)()))
    return [f"{key.split('.')[-1]}={value}" for key, value in dotted_fields(cfg)
            if key.split(".")[-1] not in skip and value != default[key]]


def config_from_args(cls, argv=None):
    """`--epochs 10`, `--model.hidden_size 512`: one flag per field of `cls`."""
    parser = argparse.ArgumentParser()
    for key, default in dotted_fields(cls()):
        kind = (lambda v: v.lower() in {"1", "true", "yes"}) if isinstance(default, bool) else type(default)
        parser.add_argument(f"--{key}", type=kind, default=default)
    cfg = cls()
    for key, value in vars(parser.parse_args(argv)).items():
        *group, name = key.split(".")
        setattr(getattr(cfg, group[0]) if group else cfg, name, value)
    return cfg
