import argparse
import pickle
import numpy as np


def _compute_component_metrics(pc_scores, spatial_vec, tr, n_bins):
    """
    Compute phase-delay metrics for one complex principal component.
    """
    temporal_phase = np.unwrap(np.angle(pc_scores))
    n_timepoints = temporal_phase.shape[0]

    # Average cycle duration in timepoints and seconds.
    avg_phase_step = np.abs(temporal_phase[-1] - temporal_phase[0]) / n_timepoints
    avg_cycle_timepoints = (2 * np.pi) / avg_phase_step
    avg_cycle_seconds = avg_cycle_timepoints * tr

    # Spatial phase-delay range.
    phase_weights = np.angle(spatial_vec)
    spatial_phase_range_ratio = (
        (np.max(phase_weights) - np.min(phase_weights)) / (2 * np.pi)
    )

    # Duration implied by the spatial phase range.
    phase_weights_duration_timepoints = spatial_phase_range_ratio * avg_cycle_timepoints
    phase_weights_duration_seconds = phase_weights_duration_timepoints * tr

    # Seconds represented by each reconstruction bin.
    seconds_per_bin = (phase_weights_duration_timepoints / n_bins) * tr

    return {
        "avg_cycle_timepoints": float(avg_cycle_timepoints),
        "avg_cycle_seconds": float(avg_cycle_seconds),
        "spatial_phase_range_ratio": float(spatial_phase_range_ratio),
        "phase_weights_duration_timepoints": float(phase_weights_duration_timepoints),
        "phase_weights_duration_seconds": float(phase_weights_duration_seconds),
        "seconds_per_bin": float(seconds_per_bin),
        "phase_weights_radians": phase_weights,
    }


def compute_phase_delay_maps(cpca_result_path, tr=0.72, n_bins=30, n_comps=3):
    """
    Extract and compute phase-delay map metrics from complex PCA results.
    """
    cpca_res = pickle.load(open(cpca_result_path, "rb"))
    pca_res = cpca_res["pca"]

    metrics = {}
    for comp_idx in range(n_comps):
        metrics[f"comp{comp_idx}"] = _compute_component_metrics(
            pca_res["pc_scores"][:, comp_idx],
            pca_res["Va"][comp_idx, :],
            tr,
            n_bins,
        )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract phase-delay map metrics from cPCA results"
    )
    parser.add_argument(
        "-i",
        "--input_cpca",
        required=True,
        type=str,
        help="Path to cPCA results pickle (e.g., pca_rest_complex_results.pkl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="phase_delay_map_metrics.pkl",
        type=str,
        help="Output pickle path",
    )
    parser.add_argument(
        "--tr",
        default=0.72,
        type=float,
        help="TR in seconds",
    )
    parser.add_argument(
        "--n_bins",
        default=30,
        type=int,
        help="Number of phase bins used for reconstruction",
    )
    parser.add_argument(
        "--n_comps",
        default=3,
        type=int,
        help="Number of components to extract",
    )
    args = parser.parse_args()

    metrics = compute_phase_delay_maps(
        cpca_result_path=args.input_cpca,
        tr=args.tr,
        n_bins=args.n_bins,
        n_comps=args.n_comps,
    )
    pickle.dump(metrics, open(args.output, "wb"))
