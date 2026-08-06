import json
import os
from typing import List

type_entityid_2_relationids = "entityid_2_relationids"
type_relationid_2_passageids = "relationid_2_passageids"

def save(type, data: List[List[int]]):
    path = os.path.join("resources", f"{type}.json")
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load(type):
    path = os.path.join("resources", f"{type}.json")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = {int(k): v for k, v in data.items()}
    return data
