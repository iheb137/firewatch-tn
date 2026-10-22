"""Entraînement du Module B.

    python -m ml.burn_torch.train --config configs/burn_torch.yaml
    python -m ml.burn_torch.train --set split.test_fold=2 split.val_fold=3

Produit dans ``<output_dir>/<run>/`` : ``best.pt``, ``history.csv``, ``config.yaml``, ``eval.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ml.burn_torch.constants import FOLDS
from ml.burn_torch.dataset import BurnDataConfig, CaBuArDataset, num_channels, worker_init
from ml.burn_torch.metrics import SegMeter
from ml.burn_torch.model import BurnModelConfig, DiceFocalLoss, build_model
from ml.common.config import load_config, seed_everything
from ml.common.tracking import Tracker


def fold_split(test_fold: int, val_fold: int) -> tuple[list[int], list[int], list[int]]:
    if test_fold == val_fold:
        raise ValueError("test_fold et val_fold doivent être différents")
    train = [f for f in FOLDS if f not in (test_fold, val_fold)]
    return train, [val_fold], [test_fold]


@torch.no_grad()
def run_eval(model, loader, device, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    meter = SegMeter(threshold=threshold)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        meter.update(torch.sigmoid(model(x)), y)
    return meter.compute()


def make_loaders(cfg: dict):
    data_cfg = BurnDataConfig(**cfg["data"], seed=cfg["seed"])
    train_f, val_f, test_f = fold_split(cfg["split"]["test_fold"], cfg["split"]["val_fold"])
    t = cfg["train"]
    common = {"num_workers": t["num_workers"], "worker_init_fn": worker_init,
              "pin_memory": torch.cuda.is_available(),
              "persistent_workers": t["num_workers"] > 0}
    train_dl = DataLoader(CaBuArDataset(data_cfg, train_f, training=True),
                          batch_size=t["batch_size"], shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(CaBuArDataset(data_cfg, val_f, training=False),
                        batch_size=t["batch_size"], **common)
    test_dl = DataLoader(CaBuArDataset(data_cfg, test_f, training=False),
                         batch_size=t["batch_size"], **common)
    return train_dl, val_dl, test_dl


def train(cfg: dict) -> Path:
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = cfg["train"]
    train_dl, val_dl, test_dl = make_loaders(cfg)

    model = build_model(BurnModelConfig(**cfg["model"],
                                        in_channels=num_channels(cfg["data"]["mode"])))
    model.to(device)
    criterion = DiceFocalLoss(**cfg["loss"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"],
                            weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    use_amp = bool(t.get("amp")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg['model']['arch']}-f{cfg['split']['test_fold']}-{stamp}"
    out = Path(t["output_dir"]) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    tracker = Tracker("firewatch-burn", run_name)
    best_iou, patience = -1.0, 0
    with tracker.run(), open(out / "history.csv", "w", newline="") as fh:
        tracker.log_params(cfg)
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_iou", "val_f1", "lr"])
        for epoch in range(t["epochs"]):
            model.train()
            total, n = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * x.size(0)
                n += x.size(0)
            sched.step()
            val = run_eval(model, val_dl, device)
            lr = opt.param_groups[0]["lr"]
            writer.writerow([epoch, total / max(n, 1), val["iou"], val["f1"], lr])
            fh.flush()
            tracker.log_metrics({"train_loss": total / max(n, 1), "val_iou": val["iou"],
                                 "val_f1": val["f1"]}, step=epoch)
            print(f"epoch {epoch:03d}  loss {total / max(n, 1):.4f}  "
                  f"val IoU {val['iou']:.4f}  F1 {val['f1']:.4f}", flush=True)
            if val["iou"] > best_iou:
                best_iou, patience = val["iou"], 0
                torch.save({"model": model.state_dict(), "config": cfg}, out / "best.pt")
            else:
                patience += 1
                if patience >= t["early_stopping_patience"]:
                    print("Arrêt anticipé.")
                    break

        ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        result = {"run": run_name, "best_val_iou": best_iou,
                  "val": run_eval(model, val_dl, device),
                  "test": run_eval(model, test_dl, device)}
        (out / "eval.json").write_text(json.dumps(result, indent=2))
        tracker.log_metrics({f"test_{k}": v for k, v in result["test"].items()})
        tracker.log_artifact(str(out / "best.pt"))
    print(json.dumps(result, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/burn_torch.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="clé=valeur")
    args = parser.parse_args()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
