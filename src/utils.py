from types import NoneType
import yaml


class ObjectLikeDictionary(dict):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError


def convert_to_dict(cfg_node, key_list):
    if not isinstance(cfg_node, ObjectLikeDictionary):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
    return cfg_dict


def add_attr_interface(element):
    if isinstance(element, list):
        return [add_attr_interface(e) for e in element]
    if isinstance(element, tuple):
        return tuple(add_attr_interface(e) for e in element)
    if isinstance(element, dict):
        return ObjectLikeDictionary(**{k: add_attr_interface(v) for k, v in element.items()})
    if isinstance(element, (int, str, float, NoneType)):
        return element
    raise Exception(f"type {type(element)} is not supported")
    