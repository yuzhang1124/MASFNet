import os
import csv
import cv2
import argparse
import numpy as np
import py_sod_metrics


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def parse_seeds(seed_str):
    return [s.strip() for s in seed_str.split(",") if s.strip() != ""]


def get_mask_list(mask_root):
    names = []
    for name in os.listdir(mask_root):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            names.append(name)
    names.sort()
    return names


def build_metrics():
    FM = py_sod_metrics.Fmeasure()
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()

    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
    overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)

    FMv2 = py_sod_metrics.FmeasureV2(
        metric_handlers={
            "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
            "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
            "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
            "rec": py_sod_metrics.RecallHandler(**sample_gray),
            "fpr": py_sod_metrics.FPRHandler(**sample_gray),
            "iou": py_sod_metrics.IOUHandler(**sample_gray),
            "dice": py_sod_metrics.DICEHandler(**sample_gray),
            "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
            "ber": py_sod_metrics.BERHandler(**sample_gray),
            "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
            "kappa": py_sod_metrics.KappaHandler(**sample_gray),

            "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
            "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
            "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
            "sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
            "sample_bifpr": py_sod_metrics.FPRHandler(**sample_bin),
            "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
            "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
            "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
            "sample_biber": py_sod_metrics.BERHandler(**sample_bin),
            "sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
            "sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),

            "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
            "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
            "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
            "overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
            "overall_bifpr": py_sod_metrics.FPRHandler(**overall_bin),
            "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
            "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
            "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
            "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
            "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
            "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
        }
    )

    return FM, WFM, SM, EM, MAE, FMv2


def evaluate_one_seed(dataset_name, pred_root, mask_root, seed):
    FM, WFM, SM, EM, MAE, FMv2 = build_metrics()

    mask_name_list = get_mask_list(mask_root)

    valid_num = 0
    missing_num = 0

    print("=" * 80)
    print(f"Evaluating {dataset_name} | seed_{seed}")
    print(f"Prediction path: {pred_root}")
    print("=" * 80)

    for i, mask_name in enumerate(mask_name_list):
        mask_path = os.path.join(mask_root, mask_name)
        pred_name = os.path.splitext(mask_name)[0] + ".png"
        pred_path = os.path.join(pred_root, pred_name)

        if not os.path.exists(pred_path):
            print(f"[Missing] {pred_path}")
            missing_num += 1
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"[Skip] Cannot read mask: {mask_path}")
            continue

        if pred is None:
            print(f"[Skip] Cannot read pred: {pred_path}")
            continue

        if pred.shape != mask.shape:
            pred = cv2.resize(
                pred,
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)
        FMv2.step(pred=pred, gt=mask)

        valid_num += 1

        if valid_num % 200 == 0:
            print(f"[seed_{seed}] Processed {valid_num}/{len(mask_name_list)}")

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]
    fmv2 = FMv2.get_results()

    curr_results = {
        "dataset": dataset_name,
        "seed": seed,
        "mDice": float(fmv2["dice"]["dynamic"].mean()),
        "mIoU": float(fmv2["iou"]["dynamic"].mean()),
        "Smeasure": float(sm),
        "wFmeasure": float(wfm),
        "Fmeasure": float(fm["adp"]),
        "Emeasure": float(em["curve"].mean()),
        "MAE": float(mae),
        "valid_num": valid_num,
        "missing_num": missing_num,
    }

    print(f"\nResults of seed_{seed}:")
    print("mDice:       ", format(curr_results["mDice"], ".3f"))
    print("mIoU:        ", format(curr_results["mIoU"], ".3f"))
    print("S_{alpha}:   ", format(curr_results["Smeasure"], ".3f"))
    print("F^{w}_{beta}:", format(curr_results["wFmeasure"], ".3f"))
    print("F_{beta}:    ", format(curr_results["Fmeasure"], ".3f"))
    print("E_{phi}:     ", format(curr_results["Emeasure"], ".3f"))
    print("MAE:         ", format(curr_results["MAE"], ".3f"))
    print("Valid num:   ", valid_num)
    print("Missing num: ", missing_num)
    print("=" * 80)

    return curr_results


def save_csv(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "dataset",
        "seed",
        "mDice",
        "mIoU",
        "Smeasure",
        "wFmeasure",
        "Fmeasure",
        "Emeasure",
        "MAE",
        "valid_num",
        "missing_num",
    ]

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow(r)

    print(f"Five-seed results saved to: {save_path}")


def save_mean_std(results, save_path):
    metric_names = [
        "mDice",
        "mIoU",
        "Smeasure",
        "wFmeasure",
        "Fmeasure",
        "Emeasure",
        "MAE",
    ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "mean±std"])

        for metric in metric_names:
            values = np.array([r[metric] for r in results], dtype=np.float64)
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0

            writer.writerow([
                metric,
                f"{mean:.6f}",
                f"{std:.6f}",
                f"{mean:.3f}±{std:.3f}",
            ])

    print(f"Mean-std results saved to: {save_path}")

    print("\n========== Mean ± Std ==========")
    for metric in metric_names:
        values = np.array([r[metric] for r in results], dtype=np.float64)
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"{metric}: {mean:.6f} ± {std:.6f}")
    print("================================")


def main(args):
    seeds = parse_seeds(args.seeds)

    all_results = []

    for seed in seeds:
        pred_root = os.path.join(args.pred_root, f"seed_{seed}")

        if not os.path.exists(pred_root):
            raise FileNotFoundError(f"Prediction folder not found: {pred_root}")

        result = evaluate_one_seed(
            dataset_name=args.dataset_name,
            pred_root=pred_root,
            mask_root=args.gt_path,
            seed=seed
        )

        all_results.append(result)

    os.makedirs(args.save_root, exist_ok=True)

    result_csv = os.path.join(
        args.save_root,
        f"{args.dataset_name}_five_seed_eval_results.csv"
    )

    mean_std_csv = os.path.join(
        args.save_root,
        f"{args.dataset_name}_five_seed_mean_std.csv"
    )

    save_csv(all_results, result_csv)
    save_mean_std(all_results, mean_std_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True,
                        help="dataset name")
    parser.add_argument("--pred_root", type=str, required=True,
                        help="root path of five seed prediction folders")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="path to the ground truth masks")
    parser.add_argument("--save_root", type=str, required=True,
                        help="path to save evaluation csv files")
    parser.add_argument("--seeds", type=str, default="2024,2025,2026,2027,2028",
                        help="seed list, separated by comma")

    args = parser.parse_args()
    main(args)