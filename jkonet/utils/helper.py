#!/usr/bin/python3
# author: Charlotte Bunne

# imports
import jax
import yaml
import collections
import ml_collections


def count_parameters(model):
    return sum(map(lambda x: x.size, jax.tree_flatten(model)[0]))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def nest_dict(d):
    result = {}
    for k, v in d.items():
        # for each key split_rec splits keys to form recursively nested dict
        split_rec(k, v, result)
    return result


def split_rec(k, v, out, sep='.'):
    # splitting keys in dict, calling recursively to break items on '.'
    k, *rest = k.split(sep, 1)
    if rest:
        split_rec(rest[0], v, out.setdefault(k, {}))
    else:
        out[k] = v


def flat_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flat_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def merge(a, b, path=None):
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def config_from_wandb(path):
    config = yaml.load(open(path), yaml.UnsafeLoader)
    del config['wandb_version']
    del config['_wandb']
    for key, val in config.items():
        val = val['value']
        config[key].pop('desc', None)
        config[key].pop('value', None)
        config[key] = val

    return ml_collections.ConfigDict(nest_dict(config))
