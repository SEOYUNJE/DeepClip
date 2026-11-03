import os
import csv
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from deepclip import models
from deepclip.models.deepclip import available_models
from deepclip.data import (
    split_data,
    class_imbalance_handle,
    DeepClipDataset,
    DeepClipCollate,
    get_train_transforms,
    get_valid_transforms,
)
from deepclip.data.text import pair_frequency, region_frequency, type_weights, region_weights
from deepclip.losses import itc_loss, cls_loss, uncertainty_weighting
from deepclip.utils import (
    seed_everything,
    AverageMeter,
    build_metrics,
    to_two_class,
    PairAveragePrecision,
    PAIR_NAMES,
    EarlyStopping,
)
from deepclip.optimizer import build_optimizer
from deepclip.scheduler import fetch_scheduler
from colorama import Fore, Style

c_ = Fore.BLUE
s_ = Style.BRIGHT
r_ = Style.RESET_ALL

ROOT_DIR = Path(__file__).resolve().parent

class Trainer:
    """DeepClip Training: itc_loss + cls_loss with uncertainty weighting"""

    def __init__(self, model, cfg):

        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.use_amp = cfg.train.use_amp and cfg.device.startswith("cuda")

        # ======= Loss =======
        self.ckpt_saved = False # a _best.pt of an earlier run must not be loaded
        self.train_cls_loss = AverageMeter()
        self.train_itc_loss = AverageMeter()
        self.valid_cls_loss = AverageMeter()
        self.valid_itc_loss = AverageMeter()

        # ======= Early Stopping (valid cls_loss) ======
        self.early_stopping = EarlyStopping(
            patience = cfg.train.patience,
            min_delta = cfg.train.min_delta,
        )

        # ======= Metrics =======
        base_metrics = build_metrics(cfg.device)
        self.train_metrics = base_metrics.clone(prefix="train_")
        self.valid_metrics = base_metrics.clone(prefix="valid_")
        self.train_pair_ap = PairAveragePrecision(cfg.device)
        self.valid_pair_ap = PairAveragePrecision(cfg.device)
        self.pair_counts = torch.zeros(len(PAIR_NAMES), dtype=torch.long)

        # ======= Optimizer =======
        self.optimizer = build_optimizer(self.model, cfg)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ======= Scheduler ========
        self.scheduler = fetch_scheduler(self.optimizer, cfg)

        # ======= Log ========
        self.log_path = Path(cfg.output_dir) / f"{cfg.model_name}_log.csv"
        self.ckpt_path = Path(cfg.output_dir) / f"{cfg.model_name}_best.pt"

        print(f"{c_}{s_}\n✅ Trainer initialized | Device: {cfg.device}\n{r_}")

    def _build_collate(self, fake_train_df, seed=None):
        pair_labels = fake_train_df[self.cfg.pair_col].tolist()

        return DeepClipCollate(
            caption_to_idx = self.model.caption_to_idx,
            type_weights = type_weights(pair_frequency(pair_labels), self.cfg.text.type_sampling_alpha),
            region_weights = region_weights(region_frequency(pair_labels), self.cfg.text.region_sampling_beta),
            max_regions = self.model.max_regions,
            max_types = self.model.max_types,
            real_text_weight = self.cfg.loss.real_text_weight,
            fake_text_weight = self.cfg.loss.fake_text_weight,
            seed = seed,
        )

    def _loader(self, df, transforms, batch_size, shuffle, collate):
        if len(df) == 0:
            raise ValueError("No frames to load, check the split and the metadata csv")

        ds = DeepClipDataset(
                meta_df = df,
                img_size = self.cfg.model.img_size,
                transforms = transforms,
                img_col = self.cfg.img_col,
                label_col = self.cfg.label_col,
                pair_col = self.cfg.pair_col,
        )
        return DataLoader(
                ds,
                shuffle = shuffle,
                batch_size = batch_size,
                num_workers = self.cfg.num_workers,
                pin_memory = self.cfg.device.startswith("cuda"),
                drop_last = False,
                collate_fn = collate,
        )

    def _write_log(self, row):
        new_file = not self.log_path.exists()

        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def fit(self, train_df, valid_df):
        # ====== Calculate Ratio Real[0]/Fake[1] ======
        real_train_df = train_df[train_df[self.cfg.label_col] == 0].reset_index(drop=True)
        fake_train_df = train_df[train_df[self.cfg.label_col] == 1].reset_index(drop=True)
        n_splits = int(np.ceil(len(fake_train_df) / len(real_train_df)))

        # ====== Sampling weights of the train FAKE frames (computed once) ======
        train_collate = self._build_collate(fake_train_df)
        valid_collate = self._build_collate(fake_train_df, seed=self.cfg.seed) # same captions every epoch

        # ====== Valid DataLoader ======
        valid_loader = self._loader(valid_df, get_valid_transforms, self.cfg.train.valid_bs, False, valid_collate)

        if self.log_path.exists():
            self.log_path.unlink()

        print(f"{c_}{s_}\n🚀 Starting Training for {self.cfg.train.epochs}...{r_}")
        for epoch in range(1, self.cfg.train.epochs + 1):
            print(f"{c_}{s_}\n{'#' * 25}\n🌙 Epoch [{epoch}/{self.cfg.train.epochs}]\n{'#' * 25}{r_}")

            # ====== Handling Class Imbalance =======
            k = (epoch - 1) % n_splits
            balanced_df = class_imbalance_handle(real_train_df, fake_train_df, k)

            # ====== Train DataLoader ======
            train_loader = self._loader(balanced_df, get_train_transforms, self.cfg.train.train_bs, True, train_collate)

            # ====== Training / Validation ======
            lr = self.optimizer.param_groups[0]['lr']
            train_out = self.train_one_epoch(train_loader)
            valid_collate.reset()
            valid_out = self.valid_one_epoch(valid_loader)
            valid_loss = valid_out["valid_cls_loss"]

            # ====== Scheduler ======
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()

            # ====== Loss weights and temperatures ======
            scalars = {
                "w_cls_loss": torch.exp(-self.model.s_cls).item(),
                "w_itc_loss": torch.exp(-self.model.s_itc).item(),
                "t_cls": self.model.t_cls.exp().item(),
                "t_itc": self.model.t_itc.exp().item(),
                "b_itc": self.model.b_itc.item(),
            }
            print(f"⚖️  w_cls_loss={scalars['w_cls_loss']:.4f}, w_itc_loss={scalars['w_itc_loss']:.4f} | "
                  f"t_cls={scalars['t_cls']:.2f}, t_itc={scalars['t_itc']:.2f}, b_itc={scalars['b_itc']:.2f}")

            # ====== CSV Logging ======
            self._write_log({
                "epoch": epoch,
                "lr": lr,
                **{k: v for k, v in train_out.items() if k != "train_pair_ap"},
                **{k: v for k, v in valid_out.items() if k != "valid_pair_ap"},
                **scalars,
                **{f"train_ap_{k}": v for k, v in train_out["train_pair_ap"].items()},
                **{f"valid_ap_{k}": v for k, v in valid_out["valid_pair_ap"].items()},
                **{f"count_{k}": v for k, v in zip(PAIR_NAMES, self.pair_counts.tolist())},
            })

            # ====== Checkpoint ======
            # the checkpoint uses the same improvement rule (min_delta) as early stopping
            prev_best = self.early_stopping.best_loss
            stop = self.early_stopping(valid_loss)

            if self.early_stopping.improved:
                print(f"✅ valid_cls_loss improved ({prev_best:.4f} → {valid_loss:.4f})")
                torch.save(self.model.state_dict(), self.ckpt_path)
                self.ckpt_saved = True

            # ====== Early Stopping ======
            if stop:
                print(f"{c_}{s_}🛑 Early stopping triggered! Training stopped.{r_}")
                break

        if not self.ckpt_saved:
            raise RuntimeError("valid_cls_loss never improved (NaN?), so no checkpoint was saved")

        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.cfg.device))
        return self.model

    def _forward(self, batch):
        imgs = batch["images"].to(self.cfg.device, non_blocking=True) # (B,C,H,W)
        labels = batch["labels"].to(self.cfg.device) # (B,)
        text_idx = batch["text_idx"].to(self.cfg.device) # (U,)
        Y = batch["Y"].to(self.cfg.device) # (B,U)
        W = batch["W"].to(self.cfg.device) # (B,U)

        img_emb = self.model.encode_image(imgs) # (B,D)
        itc = itc_loss(self.model.itc_logits(img_emb, text_idx), Y, W)
        cls_logits = self.model.cls_logits(img_emb) # (B,)
        cls = cls_loss(cls_logits, labels)

        return img_emb, cls_logits, labels, itc, cls

    def _update_metrics(self, metrics, pair_ap, img_emb, cls_logits, labels, batch):
        with torch.no_grad():
            p_fake = torch.sigmoid(cls_logits.detach().float())
            metrics.update(to_two_class(p_fake), labels.long())

            # outside autocast: img_emb may be fp16 and text_table is fp32
            pair_scores = torch.sigmoid(self.model.pair_logits(img_emb.detach().float())) # (B,20)
            pair_ap.update(pair_scores, batch["pair_targets"].to(self.cfg.device))

    def train_one_epoch(self, loader):
        self.model.train()

        # ====== reset loss, metric ======
        self.train_cls_loss.reset()
        self.train_itc_loss.reset()
        self.train_metrics.reset()
        self.train_pair_ap.reset()
        self.pair_counts.zero_()

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Training for {self.cfg.dataset}")
        for step, batch in pbar:

            # ====== Auto Mixed Precision ======
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                img_emb, cls_logits, labels, itc, cls = self._forward(batch)
                loss = uncertainty_weighting(itc, cls, self.model.s_itc, self.model.s_cls)
                loss = loss / self.cfg.train.n_accumulate

            self.scaler.scale(loss).backward()

            # ====== Gradient Accumulation ======
            if (step + 1) % self.cfg.train.n_accumulate == 0 or (step + 1) == len(loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.model.clamp_logit_scales_()

            n = labels.size(0)
            self.train_cls_loss.update(cls.item(), n)
            self.train_itc_loss.update(itc.item(), n)
            self._update_metrics(self.train_metrics, self.train_pair_ap, img_emb, cls_logits, labels, batch)
            self.pair_counts += batch["pair_counts"]

            avg_metrics = self.train_metrics.compute()

            # ====== Calculate GPU Memory Usage =======
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0

            # ====== Display Info ======
            postfix = {
                'train_cls_loss': f'{self.train_cls_loss.avg:.4f}',
                'train_itc_loss': f'{self.train_itc_loss.avg:.4f}',
                'train_acc': f'{avg_metrics["train_acc"]:.4f}',
                'train_auc': f'{avg_metrics["train_auc"]:.4f}',
                'train_f1': f'{avg_metrics["train_f1"]:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.1e}',
                'gpu_mem': f'{mem:.1f} GB',
            }

            # pair_map is only computed at the end of the epoch
            if (step + 1) == len(loader):
                pair_map, pair_ap = self.train_pair_ap.compute()
                postfix['train_pair_map'] = f'{pair_map:.4f}'

            pbar.set_postfix(**postfix)

        torch.cuda.empty_cache()
        return {
            "train_cls_loss": self.train_cls_loss.avg,
            "train_itc_loss": self.train_itc_loss.avg,
            **{k: v.item() for k, v in avg_metrics.items()},
            "train_pair_map": pair_map,
            "train_pair_ap": pair_ap,
        }

    @torch.no_grad()
    def valid_one_epoch(self, loader):
        self.model.eval()

        # ====== reset loss, metric ======
        self.valid_cls_loss.reset()
        self.valid_itc_loss.reset()
        self.valid_metrics.reset()
        self.valid_pair_ap.reset()

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Validation for {self.cfg.dataset}")
        for step, batch in pbar:

            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                img_emb, cls_logits, labels, itc, cls = self._forward(batch)

            n = labels.size(0)
            self.valid_cls_loss.update(cls.item(), n)
            self.valid_itc_loss.update(itc.item(), n)
            self._update_metrics(self.valid_metrics, self.valid_pair_ap, img_emb, cls_logits, labels, batch)

            avg_metrics = self.valid_metrics.compute()

            # ====== Calculate GPU Memory Usage =======
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0

            # ====== Display Info ======
            postfix = {
                'valid_cls_loss': f'{self.valid_cls_loss.avg:.4f}',
                'valid_itc_loss': f'{self.valid_itc_loss.avg:.4f}',
                'valid_acc': f'{avg_metrics["valid_acc"]:.4f}',
                'valid_auc': f'{avg_metrics["valid_auc"]:.4f}',
                'valid_f1': f'{avg_metrics["valid_f1"]:.4f}',
                'gpu_mem': f'{mem:.1f} GB',
            }

            if (step + 1) == len(loader):
                pair_map, pair_ap = self.valid_pair_ap.compute()
                postfix['valid_pair_map'] = f'{pair_map:.4f}'

            pbar.set_postfix(**postfix)

        torch.cuda.empty_cache()
        return {
            "valid_cls_loss": self.valid_cls_loss.avg,
            "valid_itc_loss": self.valid_itc_loss.avg,
            **{k: v.item() for k, v in avg_metrics.items()},
            "valid_pair_map": pair_map,
            "valid_pair_ap": pair_ap,
        }

    def __str__(self):
        return "DeepClip Trainer"


def main():
    parser = argparse.ArgumentParser(description="DeepClip Training")

    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--model-name", default="deepclip_b0_r2_t2", choices=available_models())
    parser.add_argument("--config", default=None, help="yaml file (default: deepclip/config/deepclip_{b0,b5}.yaml from --model-name)")
    parser.add_argument("--output-dir", default=".", help="Directory of the best checkpoint and the log csv")

    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num-workers", default=max(os.cpu_count() - 1, 0), type=int)

    # Metadata
    parser.add_argument("--metadata", default="train_clip_frame_metadata.csv", type=str)
    parser.add_argument("--label-col", default='label', type=str)
    parser.add_argument("--img-col", default='img_path', type=str)
    parser.add_argument("--pair-col", default='pair_label', type=str)
    parser.add_argument("--ori-vid-col", default='ori_vid', type=str)
    parser.add_argument("--crop-dir", default='crops', type=str)

    args = parser.parse_args()
    seed_everything(args.seed)

    backbone = args.model_name.split("_")[1] # b0 or b5
    config_path = args.config or ROOT_DIR / "deepclip" / "config" / f"deepclip_{backbone}.yaml"

    cfg = OmegaConf.load(config_path)
    cfg.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    cfg.model_name = args.model_name
    cfg.output_dir = args.output_dir
    cfg.seed = args.seed
    cfg.num_workers = args.num_workers
    cfg.label_col = args.label_col
    cfg.img_col = args.img_col
    cfg.pair_col = args.pair_col
    os.makedirs(args.output_dir, exist_ok=True)

    # ==================== Build MetaData ==================================
    meta_df = pd.read_csv(os.path.join(args.root_dir, args.metadata),
                          dtype={'vid': str, args.ori_vid_col: str, args.pair_col: str})
    meta_df[args.label_col] = meta_df[args.label_col].map({"REAL": 0, "FAKE": 1})
    meta_df[args.img_col] = meta_df.apply(lambda x: os.path.join(args.root_dir, f"{args.crop_dir}/{x['vid']}/{x['frame_idx']}.png"), axis=1)

    # ==================== Build Dataset ==================================
    train_df, valid_df = split_data(meta_df=meta_df, seed=args.seed, label_col=args.label_col,
                                    ori_vid=args.ori_vid_col, dataset="ff++", debug=args.debug)

    # ==================== Build Model ===============================
    model = getattr(models, args.model_name)(**OmegaConf.to_container(cfg.model, resolve=True))
    print(f"📝 Text table: {model.num_captions} captions × {model.embed_dim} "
          f"({model.clip_name} text, built in {model.text_table_time:.1f}s)")

    # =================== Training ============================
    trainer = Trainer(model, cfg)
    trainer.fit(train_df, valid_df)

if __name__ == "__main__":
    main()
