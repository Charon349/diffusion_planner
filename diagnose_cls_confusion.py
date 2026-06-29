#!/usr/bin/env python3
"""
Diagnose current DP4 anchor-score selection confusion + per-anchor recall.

Current DP4 anchored inference returns:
  - anchor_predictions: all K refined ego modes after denoising
  - anchor_scores: score-head logits used to choose the branch
  - anchor_selected_index: the decoder's selected branch

This script builds two KxK diagnostics:
  1. refined-oracle confusion:
       row = argmin_k ADE(refined_pred_k, GT), col = argmax(anchor_scores)
  2. training-label confusion:
       row = argmin_k loss._anchor_distances(fixed_anchor_k, GT),
       col = argmax(anchor_scores)

Together they separate selection failure from denoiser slot quality. If the
training-label recall is good but refined-oracle recall is poor, the denoiser is
reshaping slots. If training-label recall is poor, the score head or anchor label
metric is the bottleneck.

Usage:
    python diagnose_cls_confusion.py \
        --checkpoint /path/to/latest.pth \
        --args_file /path/to/args.json \
        --data_dir /path/to/val_data_for_diffusion \
        --data_list /path/to/val_list.json \
        --normalization normalization.json \
        --output_dir ./diagnose_results

--anchor, --num_ego_anchors, --ego_anchor_state_format, future_len, and
predicted_neighbor_num are read from args.json when present. --data_dir and
--data_list should point to a held-out preprocessed validation set. Use
--allow_train_set_eval only for a quick training-set sanity check.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.loss import _anchor_distances
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.anchors import load_ego_anchors
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.train_utils import openjson


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",        required=True)
    p.add_argument("--label",             default=None)
    p.add_argument("--args_file",         default=None,
                   help="training args.json. If omitted, tries <checkpoint_dir>/args.json")
    p.add_argument("--anchor",            default=None,
                   help="ego anchor file. Defaults to ego_anchor_path in args.json")
    p.add_argument("--ego_anchor_state_format", default=None, choices=["delta", "absolute"],
                   help="Defaults to ego_anchor_state_format in args.json, or absolute")
    p.add_argument("--num_ego_anchors",   type=int, default=None,
                   help="Defaults to num_ego_anchors in args.json, or all anchors in file")
    p.add_argument("--data_dir",          default=None,
                   help="Held-out preprocessed .npz data dir")
    p.add_argument("--data_list",         default=None,
                   help="Held-out JSON list of .npz samples")
    p.add_argument("--allow_train_set_eval", action="store_true",
                   help="If data_dir/data_list are omitted, fall back to train_set/train_set_list "
                        "from args.json. This is only a training-set sanity check.")
    p.add_argument("--normalization",     default="normalization.json")
    p.add_argument("--future_len",        type=int,   default=None)
    p.add_argument("--predicted_neighbor_num", type=int, default=None)
    p.add_argument("--past_neighbor_num",      type=int, default=None)
    p.add_argument("--max_samples",       type=int,   default=3000)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--static_thresh",     type=float, default=1.0,
                   help="Anchor speed threshold for STATIC label (m/s)")
    p.add_argument("--turn_thresh_deg",   type=float, default=10.0,
                   help="|net-turn| above this (deg) labels an anchor as TURNING")
    p.add_argument("--selector", choices=["score", "cls", "auto", "selected"], default="score",
                   help="'score'/'cls' uses argmax(anchor_scores); 'selected' uses the "
                        "decoder's anchor_selected_index. For current DP4 these should match.")
    p.add_argument("--use_ema", action="store_true",
                   help="Load ema_state_dict from training checkpoints. Closed-loop planner can use this.")
    p.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir",        default="./diagnose_results")
    p.add_argument("--seed",              type=int, default=42)
    return p.parse_args()


class Cfg:
    pass


def build_config(args, norm_data, train_args=None):
    """Build the current DP4 model config from the saved training args."""
    train_args = train_args or {}
    cfg = Cfg()
    cfg.hidden_dim              = int(train_args.get("hidden_dim", 192))
    cfg.num_heads               = int(train_args.get("num_heads", 6))
    cfg.device                  = args.device
    cfg.encoder_drop_path_rate  = 0.0
    cfg.decoder_drop_path_rate  = 0.0
    cfg.encoder_depth           = int(train_args.get("encoder_depth", 3))
    cfg.decoder_depth           = int(train_args.get("decoder_depth", 3))
    cfg.route_num               = int(train_args.get("route_num", 25))
    cfg.lane_len                = int(train_args.get("lane_len", 20))
    cfg.diffusion_model_type    = train_args.get("diffusion_model_type", "x_start")
    cfg.future_len              = args.future_len
    cfg.time_len                = int(train_args.get("time_len", 21))
    cfg.predicted_neighbor_num  = args.predicted_neighbor_num
    cfg.use_ego_anchor          = True
    cfg.use_anchor_score        = bool(train_args.get("use_anchor_score", True))
    cfg.num_ego_anchors         = args.num_ego_anchors
    cfg.ego_anchor_path         = args.anchor
    cfg.ego_anchor_state_format = args.ego_anchor_state_format
    cfg.anchor_sampling_t_start = float(train_args.get("anchor_sampling_t_start",
                                          train_args.get("ego_anchor_t_max", 0.2)))
    cfg.anchor_sampling_steps   = int(train_args.get("anchor_sampling_steps", 10))
    cfg.anchor_neighbor_init    = train_args.get("anchor_neighbor_init", "cv")
    cfg.agent_state_dim         = int(train_args.get("agent_state_dim", 11))
    cfg.agent_num               = int(train_args.get("agent_num", 32))
    cfg.static_objects_state_dim = int(train_args.get("static_objects_state_dim", 10))
    cfg.static_objects_num      = int(train_args.get("static_objects_num", 5))
    cfg.lane_state_dim          = int(train_args.get("lane_state_dim", 12))
    cfg.lane_num                = int(train_args.get("lane_num", 70))
    cfg.route_len               = int(train_args.get("route_len", 20))
    cfg.route_state_dim         = int(train_args.get("route_state_dim", 12))
    cfg.state_normalizer = StateNormalizer(
        mean=[[norm_data["ego"]["mean"]]] + [[norm_data["neighbor"]["mean"]]] * args.predicted_neighbor_num,
        std= [[norm_data["ego"]["std"]]]  + [[norm_data["neighbor"]["std"]]]  * args.predicted_neighbor_num,
    )
    cfg.observation_normalizer = ObservationNormalizer.from_json(args.normalization)
    cfg.guidance_fn = None
    return cfg


def _resolve_existing_path(path, base_dir):
    if path is None or os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(base_dir, path)
    return candidate if os.path.exists(candidate) else path


def _infer_args_file(checkpoint):
    candidate = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "args.json")
    return candidate if os.path.exists(candidate) else None


def _load_train_args(args):
    if args.args_file is None:
        args.args_file = _infer_args_file(args.checkpoint)
    return openjson(args.args_file) if args.args_file else {}


def _resolve_runtime_args(args, train_args, script_dir):
    args.normalization = _resolve_existing_path(args.normalization, script_dir)
    if args.data_dir is None or args.data_list is None:
        if not args.allow_train_set_eval:
            raise ValueError(
                "Open-loop diagnosis needs a held-out preprocessed set. Pass --data_dir "
                "and --data_list for validation/test data, or add --allow_train_set_eval "
                "to explicitly run a training-set sanity check."
            )
        print("  [warn] Using train_set/train_set_list from args.json. "
              "This is NOT a held-out test metric.")
        args.data_dir = args.data_dir or train_args.get("train_set", None)
        args.data_list = args.data_list or train_args.get("train_set_list", None)
    args.data_dir = _resolve_existing_path(args.data_dir, script_dir)
    args.data_list = _resolve_existing_path(args.data_list, script_dir)
    if args.data_dir is None:
        raise ValueError("Missing --data_dir")
    if args.data_list is None:
        raise ValueError("Missing --data_list")
    args.future_len = int(args.future_len or train_args.get("future_len", 80))
    args.predicted_neighbor_num = int(
        args.predicted_neighbor_num or train_args.get("predicted_neighbor_num", 10)
    )
    args.past_neighbor_num = int(args.past_neighbor_num or train_args.get("agent_num", 32))
    args.anchor = args.anchor or train_args.get("ego_anchor_path", None)
    args.anchor = _resolve_existing_path(args.anchor, script_dir)
    args.ego_anchor_state_format = (
        args.ego_anchor_state_format
        or train_args.get("ego_anchor_state_format", "absolute")
    )
    if args.anchor is None:
        raise ValueError("Missing --anchor and no ego_anchor_path found in args.json")

    loaded = load_ego_anchors(
        args.anchor,
        future_len=args.future_len,
        num_anchors=int(args.num_ego_anchors or train_args.get("num_ego_anchors", 0)),
        state_format=args.ego_anchor_state_format,
    )
    if args.num_ego_anchors is None:
        args.num_ego_anchors = int(loaded.shape[0])
    if args.num_ego_anchors <= 1:
        raise ValueError("Confusion matrix only makes sense for num_ego_anchors > 1")
    if loaded.shape[0] != args.num_ego_anchors:
        raise ValueError(f"loaded anchor K={loaded.shape[0]} != num_ego_anchors={args.num_ego_anchors}")
    if args.label is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        # Include the model experiment directory name for uniqueness
        model_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint))))
        args.label = f"dp4_{model_dir}_{stem}_K{args.num_ego_anchors}"
    return loaded


def prepare_inputs(batch, device, obs_norm, n_nb):
    """Build observation-normalized inputs for current DP4 inference."""
    inputs = {
        'ego_current_state':           batch[0].to(device),
        'neighbor_agents_past':         batch[2].to(device),
        'lanes':                        batch[4].to(device),
        'lanes_speed_limit':            batch[5].to(device),
        'lanes_has_speed_limit':        batch[6].to(device),
        'route_lanes':                  batch[7].to(device),
        'route_lanes_speed_limit':      batch[8].to(device),
        'route_lanes_has_speed_limit':  batch[9].to(device),
        'static_objects':               batch[10].to(device),
    }
    inputs = obs_norm(inputs)
    return inputs


def anchor_labels(anchors_delta, dt, static_thresh, turn_thresh_deg):
    """Per-anchor (mean_speed m/s, net_turn deg) from current DP4 delta-state anchors."""
    delta_xy = anchors_delta[..., :2]
    speed = np.linalg.norm(delta_xy / dt, axis=-1).mean(axis=1)
    head = np.arctan2(anchors_delta[..., 3], anchors_delta[..., 2])
    d = head[:, -1] - head[:, 0]
    net_turn = np.degrees(np.arctan2(np.sin(d), np.cos(d)))
    net_turn[speed < static_thresh] = 0.0
    is_static = speed < static_thresh
    is_turn = np.abs(net_turn) > turn_thresh_deg
    return speed, net_turn, is_static, is_turn


def _fmt_turn(t):
    if abs(t) <= 1e-6:
        return "  0°"
    return f"{t:+4.0f}°"


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_args = _load_train_args(args)
    anchors_delta_t = _resolve_runtime_args(args, train_args, script_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    # Seed both RNGs. np.random fixes the val subset; torch fixes diffusion noise.
    # Without the torch seed, oracle/selected ADE drifts run-to-run on the same subset.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    K  = args.num_ego_anchors
    dt = 0.1
    T  = args.future_len

    # ── Normalization ──────────────────────────────────────────────────────
    norm_data = openjson(args.normalization)

    # ── Anchor speed / turn labels ───────────────────────────────────────────
    anchors_delta = anchors_delta_t.cpu().numpy()  # [K, T, 4] = [dx, dy, cos, sin]
    spd, turn, is_static, is_turn = anchor_labels(
        anchors_delta, dt, args.static_thresh, args.turn_thresh_deg)

    print("\nAnchor legend (id | speed | net-turn | kind):")
    for k in range(K):
        kind = "STATIC" if is_static[k] else ("TURN" if is_turn[k] else "straight")
        print(f"  [{k:2d}] {spd[k]:5.2f} m/s  {_fmt_turn(turn[k])}  {kind}")

    # Fixed anchor paths define the current DP4 score-head training label
    # (loss._anchor_distances), independent of the denoiser's refined output.
    anchors_delta_dev = anchors_delta_t.to(args.device)
    anchor_xy = torch.cumsum(anchors_delta_dev[..., :2], dim=1)  # [K,T,2]

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = DiffusionPlannerData(
        data_dir=args.data_dir,
        data_list=args.data_list,
        past_neighbor_num=args.past_neighbor_num,
        predicted_neighbor_num=args.predicted_neighbor_num,
        future_len=T,
    )
    n = min(args.max_samples, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)
    loader  = DataLoader(Subset(dataset, indices),
                         batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ── Model ──────────────────────────────────────────────────────────────
    cfg   = build_config(args, norm_data, train_args)
    model = Diffusion_Planner(cfg).to(args.device)
    ckpt  = torch.load(args.checkpoint, map_location=args.device)
    # Strip DDP "module." prefix (checkpoints are saved from the DDP-wrapped model);
    # without this, strict=False silently loads NOTHING → init model.
    if isinstance(ckpt, dict) and args.use_ema and "ema_state_dict" in ckpt:
        _sd = ckpt["ema_state_dict"]
        loaded_state_name = "ema_state_dict"
    else:
        _sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        loaded_state_name = "model"
    _sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in _sd.items()}
    _miss, _unexp = model.load_state_dict(_sd, strict=False)
    print(f"  [load] matched={len(model.state_dict())-len(_miss)}/{len(model.state_dict())} "
          f"missing={len(_miss)} unexpected={len(_unexp)} source={loaded_state_name}")
    model.eval()
    print(f"Loaded: {args.checkpoint}")
    sel_name = "AnchorScoreHead"
    print(f"  [select] prediction = {'decoder anchor_selected_index' if args.selector == 'selected' else 'argmax(anchor_scores)'}")

    # ── Collect (gt_k, pred_k) pairs + ADE penalty ───────────────────────────
    obs_norm = cfg.observation_normalizer
    confusion = np.zeros((K, K), dtype=np.int64)   # rows = refined oracle, cols = selected anchor
    row_ade_penalty = np.zeros(K, dtype=np.float64)  # sum of (selected_ade - oracle_ade) per row
    geo_confusion = np.zeros((K, K), dtype=np.int64)   # rows = training anchor label, cols = score pred
    geo_intent_mode_ade = np.zeros(K, dtype=np.float64)  # Σ refined-ADE of mode k WHEN k is training label
    all_oracle_ade, all_selected_ade, all_match = [], [], []
    N = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"anchor-score confusion [{args.label}]"):
            gt_future_raw = batch[1].to(args.device)      # [B, T, 3] GT ego [x,y,heading]
            # Convert heading angle to cos/sin, matching train_epoch.py preprocessing
            gt_future = torch.cat(
                [
                    gt_future_raw[..., :2],
                    torch.stack(
                        [gt_future_raw[..., 2].cos(), gt_future_raw[..., 2].sin()], dim=-1
                    ),
                ],
                dim=-1,
            )                                              # [B, T, 4] GT ego [x,y,cos,sin]
            gt_xy  = gt_future[:, :, :2]                  # [B, T, 2] GT ego positions
            inputs = prepare_inputs(batch, args.device, obs_norm, args.predicted_neighbor_num)

            try:
                _, decoder_outputs = model(inputs)
            except Exception as e:
                print(f"  [warn] {e}")
                continue
            if "anchor_predictions" not in decoder_outputs:
                print("  [warn] decoder output has no anchor_predictions; check --use_ego_anchor/args.json")
                continue

            B = gt_xy.shape[0]
            x0_all = decoder_outputs["anchor_predictions"]
            if x0_all.shape[1] < K:
                print(f"  [warn] anchor_predictions has {x0_all.shape[1]} slots, expected at least {K}")
                continue

            # Current DP4 returns integrated ego trajectories in ego-centric xy.
            ego_xy  = x0_all[:, :K, :, :2]                            # [B, K, T, 2]
            gt_exp  = gt_xy[:, None].expand(-1, K, -1, -1)
            mode_ade = (ego_xy - gt_exp).norm(dim=-1).mean(dim=-1)   # [B, K]

            oracle_ade_b, gt_k_b = mode_ade.min(dim=1)               # GT = position-ADE oracle
            if args.selector == "selected" or "anchor_scores" not in decoder_outputs:
                pred_k_b = decoder_outputs.get(
                    "anchor_selected_index",
                    torch.zeros(B, dtype=torch.long, device=args.device),
                )
            else:
                logits = decoder_outputs["anchor_scores"][:, :K]      # [B, K]
                pred_k_b = logits.argmax(dim=1)                       # pred = score-head argmax
            selected_ade_b = mode_ade[torch.arange(B, device=mode_ade.device), pred_k_b]

            gt_k  = gt_k_b.cpu().numpy()
            pred_k = pred_k_b.cpu().numpy()
            np.add.at(confusion, (gt_k, pred_k), 1)
            np.add.at(row_ade_penalty, gt_k, (selected_ade_b - oracle_ade_b).cpu().numpy())

            # Current DP4 score-head training label: nearest fixed anchor under
            # loss._anchor_distances' lat/lon weighted metric.
            train_d = _anchor_distances(
                anchors_delta_dev,
                gt_future,
                float(train_args.get("anchor_w_lat", 1.0)),
                float(train_args.get("anchor_w_lon", 0.2)),
            )
            geo_k_b = train_d.argmin(dim=1)                                        # [B]
            intent_mode_ade_b = mode_ade[torch.arange(B, device=mode_ade.device), geo_k_b]
            geo_k = geo_k_b.cpu().numpy()
            np.add.at(geo_confusion, (geo_k, pred_k), 1)
            np.add.at(geo_intent_mode_ade, geo_k, intent_mode_ade_b.cpu().numpy())

            all_oracle_ade.extend(oracle_ade_b.cpu().tolist())
            all_selected_ade.extend(selected_ade_b.cpu().tolist())
            all_match.extend((gt_k_b == pred_k_b).cpu().tolist())
            N += B

    if N == 0:
        print("[ERROR] No data captured — check args/checkpoint anchor config and data paths.")
        return

    # ── Per-anchor recall / precision / mis-route ────────────────────────────
    support   = confusion.sum(axis=1)                 # GT count per anchor
    predicted = confusion.sum(axis=0)                 # selected count per anchor
    diag      = np.diag(confusion).astype(np.float64)
    recall    = np.divide(diag, support,   out=np.zeros(K), where=support > 0)
    precision = np.divide(diag, predicted, out=np.zeros(K), where=predicted > 0)
    mean_penalty = np.divide(row_ade_penalty, support, out=np.zeros(K), where=support > 0)

    # Top mis-route destination per GT row (largest off-diagonal column)
    misroute = []
    for k in range(K):
        off = confusion[k].copy(); off[k] = 0
        if off.sum() == 0:
            misroute.append((-1, 0.0))
        else:
            j = int(off.argmax())
            misroute.append((j, off[j] / max(support[k], 1)))

    overall_acc = diag.sum() / N
    oracle_ade = float(np.mean(all_oracle_ade))
    selected_ade = float(np.mean(all_selected_ade))

    # ── Print confusion matrix (row-normalized %) ────────────────────────────
    print(f"\n{'='*64}")
    print(f"  ANCHOR-SCORE CONFUSION MATRIX  [{args.label}]  K={K}  N={N}")
    print(f"  rows = GT (position-ADE oracle best after denoising),  cols = {sel_name} argmax")
    print(f"  cell = % of each GT row routed to that column")
    print(f"{'='*64}")
    header = "  gt\\pred " + "".join(f"{j:>5d}" for j in range(K)) + "   | recall  supp"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in range(K):
        cells = ""
        for j in range(K):
            pct = 100.0 * confusion[k, j] / support[k] if support[k] > 0 else 0.0
            mark = "*" if j == k else " "       # mark diagonal
            cells += f"{pct:4.0f}{mark}"
        print(f"  [{k:2d}]   {cells}   | {recall[k]*100:5.1f}% {int(support[k]):>5d}")

    # ── Per-anchor recall table, worst first ─────────────────────────────────
    print(f"\n  Per-anchor recall (worst first) — low recall = score head mis-routes this anchor")
    print(f"  {'id':>3}  {'spd':>5}  {'turn':>5}  {'kind':>8}  {'supp':>6}  "
          f"{'recall':>7}  {'prec':>6}  {'→top wrong':>14}  {'ΔADE m':>7}")
    print("  " + "-" * 88)
    for k in sorted(range(K), key=lambda i: recall[i]):
        kind = "STATIC" if is_static[k] else ("TURN" if is_turn[k] else "straight")
        j, share = misroute[k]
        dest = f"[{j:>2}] {share*100:4.0f}%" if j >= 0 else "      —"
        print(f"  {k:>3}  {spd[k]:5.2f}  {_fmt_turn(turn[k]):>5}  {kind:>8}  "
              f"{int(support[k]):>6}  {recall[k]*100:6.1f}%  {precision[k]*100:5.1f}%  "
              f"{dest:>14}  {mean_penalty[k]:+7.3f}")

    # ── Turning-vs-straight breakdown (the headline question) ────────────────
    def _grp_recall(mask):
        s = support[mask].sum()
        return (diag[mask].sum() / s) if s > 0 else float("nan"), int(s)
    turn_r, turn_s   = _grp_recall(is_turn)
    str_mask         = (~is_turn) & (~is_static)
    straight_r, str_s = _grp_recall(str_mask)
    stat_r, stat_s   = _grp_recall(is_static)

    print(f"\n  Recall by anchor kind:")
    print(f"    STATIC   : {stat_r*100:5.1f}%  (support {stat_s})")
    print(f"    straight : {straight_r*100:5.1f}%  (support {str_s})")
    print(f"    TURNING  : {turn_r*100:5.1f}%  (support {turn_s})   "
          f"← sparse turn anchors / high-speed-right gap show up here")

    print(f"\n  Overall  : top-1 acc {overall_acc*100:5.1f}%   "
          f"oracle_ADE {oracle_ade:.3f} m   selected_ADE {selected_ade:.3f} m   "
          f"gap {selected_ade - oracle_ade:+.3f} m")
    if not np.isnan(turn_r) and not np.isnan(straight_r) and turn_r < straight_r - 0.15:
        print("  ⚠  Turning-anchor recall is well below straight: score-head mis-selection "
              "concentrates on turn modes (confirms the Stage-1 worry).")
    elif overall_acc > 0.8:
        print("  ✓  score head reproduces the refined-oracle pick well across kinds.")

    # ── Training-label diagnostic (denoiser-independent GT) ───────────────────
    # Splits the turn failure into its three possible layers:
    #   geo_recall[k]  = of samples whose GT PATH maps to training label k, how often
    #                    score head picks k. GT here does NOT depend on the denoiser.
    #   intent_modeADE = mean refined-ADE of mode k WHEN k is the intent = denoiser slot
    #                    quality. If HIGH for turn anchors, the denoiser never produces a
    #                    good turn trajectory → selection cannot help (denoiser bottleneck).
    geo_support = geo_confusion.sum(axis=1)
    geo_diag    = np.diag(geo_confusion).astype(np.float64)
    geo_recall  = np.divide(geo_diag, geo_support, out=np.zeros(K), where=geo_support > 0)
    intent_modeADE = np.divide(geo_intent_mode_ade, geo_support,
                               out=np.zeros(K), where=geo_support > 0)
    print(f"\n  Training-label intent (denoiser-INDEPENDENT GT = loss._anchor_distances):")
    print(f"  {'id':>3}  {'kind':>8}  {'geo_supp':>8}  {'geo_recall':>10}  "
          f"{'intent_modeADE':>14}  {'→top score':>10}")
    print("  " + "-" * 70)
    for k in sorted(range(K), key=lambda i: (bool(is_turn[i]), -spd[i])):  # turns last
        kind = "STATIC" if is_static[k] else ("TURN" if is_turn[k] else "straight")
        off = geo_confusion[k].copy(); off[k] = 0
        dest = f"[{int(off.argmax()):>2}]" if (geo_support[k] and off.sum()) else "   —"
        print(f"  {k:>3}  {kind:>8}  {int(geo_support[k]):>8}  {geo_recall[k]*100:9.1f}%  "
              f"{intent_modeADE[k]:>13.3f}m  {dest:>10}")

    # Verdict: contrast the turn anchors' SELECTION (geo_recall) vs DENOISER quality
    # (intent_modeADE) against the straight anchors'.
    tr_mask = is_turn & (geo_support > 0)
    st_mask = (~is_turn) & (~is_static) & (geo_support > 0)
    if tr_mask.any() and st_mask.any():
        tr_modeADE = float(np.average(intent_modeADE[tr_mask], weights=geo_support[tr_mask]))
        st_modeADE = float(np.average(intent_modeADE[st_mask], weights=geo_support[st_mask]))
        tr_geo_rec = float(geo_diag[tr_mask].sum() / geo_support[tr_mask].sum())
        print(f"\n  TURN vs straight (geometric intent):  geo_recall {tr_geo_rec*100:.1f}% vs "
              f"{float(geo_diag[st_mask].sum()/geo_support[st_mask].sum())*100:.1f}%   |   "
              f"intent_modeADE {tr_modeADE:.3f}m vs {st_modeADE:.3f}m")
        if tr_modeADE > st_modeADE * 1.8:
            print("  ⇒ DENOISER bottleneck: turn slots are refined far worse than straight — "
                  "no good turn trajectory exists to select. Fix the denoiser (turn-slot "
                  "weight/supervision), not the score head.")
        elif tr_geo_rec < 0.15:
            print("  ⇒ SELECTION/LABEL bottleneck: turn modes are refined fine but the "
                  "score head/training label still avoids them. Revisit anchor score "
                  "supervision or the anchor distance metric.")

    # ── Save ─────────────────────────────────────────────────────────────────
    summary = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "anchor": args.anchor,
        "args_file": args.args_file,
        "use_ema": bool(args.use_ema),
        "ego_anchor_state_format": args.ego_anchor_state_format,
        "anchor_w_lat": float(train_args.get("anchor_w_lat", 1.0)),
        "anchor_w_lon": float(train_args.get("anchor_w_lon", 0.2)),
        "selector": sel_name,
        "K": K, "n_samples": int(N),
        "overall_top1_acc": float(overall_acc),
        "oracle_ADE_mean_m": oracle_ade,
        "selected_ADE_mean_m": selected_ade,
        "ADE_gap_m": float(selected_ade - oracle_ade),
        "pct_match": float(np.mean(all_match) * 100),
        "confusion_matrix": confusion.tolist(),
        "anchor_stats": [
            {
                "anchor_id": k,
                "mean_speed_ms": float(spd[k]),
                "net_turn_deg": float(turn[k]),
                "is_static": bool(is_static[k]),
                "is_turning": bool(is_turn[k]),
                "support": int(support[k]),
                "predicted": int(predicted[k]),
                "recall": float(recall[k]),
                "precision": float(precision[k]),
                "top_misroute_anchor": int(misroute[k][0]),
                "top_misroute_share": float(misroute[k][1]),
                "mean_ade_penalty_m": float(mean_penalty[k]),
                "geo_support": int(geo_support[k]),
                "geo_recall": float(geo_recall[k]),
                "intent_mode_ade_m": float(intent_modeADE[k]),
            }
            for k in range(K)
        ],
        "geo_confusion_matrix": geo_confusion.tolist(),
        "recall_by_kind": {
            "static":   {"recall": float(stat_r),     "support": stat_s},
            "straight": {"recall": float(straight_r), "support": str_s},
            "turning":  {"recall": float(turn_r),     "support": turn_s},
        },
    }
    out_json = os.path.join(args.output_dir, f"{args.label}_anchor_score_confusion.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {out_json}")


if __name__ == "__main__":
    main()
