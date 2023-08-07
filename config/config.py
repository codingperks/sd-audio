import json


def parse_json(config_json):
    """
    For loading model config jsons
    """
    with open(config_json, "r") as file:
        return json.load(file)
