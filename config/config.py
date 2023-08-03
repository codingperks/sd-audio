import json


def parse_json(config_json):
    with open(config_json, "r") as file:
        return json.load(file)
