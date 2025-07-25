import os
import json
from datetime import datetime
import os
import json
from datetime import datetime

def params_damp(configure: dict, base_log_dir: str = "./logs") -> str:

    sub_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, sub_dir)
    os.makedirs(log_dir, exist_ok=True)


    file_path = os.path.join(log_dir, "params.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(configure, f, indent=4, ensure_ascii=False)

    print(f"[params_dump] params json saved at {file_path}")
    return log_dir

if __name__ == "__main__":
    cfg = {
        "task": "CAV",
        "batch_size": 32,
        "lr": 0.001,
        "num_epochs": 100,
        "network_module": "MLP",
        "MLP_layers_config": [128, 64],
        "write_interval": "auto",
        "ckp_interval": "auto"
    }
    logdir = params_damp(cfg, "./logs/CAV/")
    print(logdir)