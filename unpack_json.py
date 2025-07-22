import json
from pathlib import Path

for p in Path("optimized_cuda_code").glob("*.json"):
    data = json.load(p.open())
    device_name = p.stem
    for level in "123":
        for task in data[level]:
            root = Path("unpacked", device_name, level, str(task["task_id"]))
            root.mkdir(parents=True, exist_ok=True)
            (root / "ref_code.py").write_text(task["ref_code"])
            (root / "custom_code.py").write_text(task["custom_code"])
