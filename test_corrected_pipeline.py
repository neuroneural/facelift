"""
test_corrected_pipeline.py
--------------------------
Standalone test script that mirrors the corrected preprocess_mrn_dask.py logic
for a single subject.  Fetches data from MongoDB (or reuses local .nii files),
runs every step of the fixed pipeline, and writes NIfTI outputs you can open in
ITK-SNAP / FSLeyes for visual inspection.

Corrections vs. the old test_face_mask_corrected.py
----------------------------------------------------
1. head_mask is derived from the ALIGNED REFERENCE (not the subject T1), so the
   face mask reflects where the reference face tissue is, not the subject's.
2. The head_mask threshold is percentile-based (2nd percentile of non-zero voxels)
   rather than a hard value of 10.
3. Intensity harmonisation: the reference face is linearly rescaled to match the
   subject's face P5→P95 range before transplanting, handling scanner /
   field-strength differences.

Outputs saved in the workspace directory (all .nii.gz, uint8)
--------------------------------------------------------------
  subject_<id>_t1.nii.gz                   — subject T1 (cached)
  subject_<id>_brain_mask.nii.gz           — brain mask from label3 (used in pipeline)
  subject_<id>_brain_mask_label50.nii.gz   — brain mask from label50 (inspection only)
  subject_<id>_brain_mask_label104.nii.gz  — brain mask from label104 (inspection only)
  ref_id<ref>.nii.gz                       — reference T1 (cached)
  aligned_ref_to_<id>.nii.gz             — reference aligned to subject space (cached)
  subject_<id>_face_mask.nii.gz           — face mask
  subject_<id>_face_highlighted.nii.gz    — T1 with face region set to 255
  subject_<id>_hybrid_harmonized.nii.gz   — training target (harmonised transplant)
  subject_<id>_hybrid_raw.nii.gz         — same without harmonisation (comparison only)

Usage
-----
  python test_corrected_pipeline.py --id 477
  python test_corrected_pipeline.py --id 477 --ref 963 --host 127.0.0.1 --port 27017
  python test_corrected_pipeline.py --id 477 --force-fetch   # re-download from MongoDB
"""

import argparse
import gzip
import io
import os
import struct
import subprocess
import sys

import lz4.frame
import numpy as np
import pymongo
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = "/Users/splis/soft/src/dev/craft/meshnet/facelift"
DB_NAME = "MindfulTensors"
COLLECTION_NAME = "MRN"
DEFAULT_REFERENCE_ID = 962

# ---------------------------------------------------------------------------
# Shared helpers (kept in sync with preprocess_mrn_dask.py)
# ---------------------------------------------------------------------------

def name2collections(name, db):
    return db[f"{name}.bin"], db[f"{name}.meta"]


def get_sample(subject_id, kind, collection, db):
    col_bin, _ = name2collections(collection, db)
    cursor = col_bin.find(
        {"id": subject_id, "kind": kind}, {"chunk_id": 1, "chunk": 1}
    ).sort("chunk_id", 1)
    chunks = [d["chunk"] for d in cursor]
    if not chunks:
        raise ValueError(f"No data for id={subject_id} kind={kind}")
    raw = b"".join(chunks)
    if raw[:4] == b'\x04\x22\x4d\x18':
        raw = lz4.frame.decompress(raw)
    return torch.load(io.BytesIO(raw), weights_only=True)


def make_nifti_header(shape, dtype):
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    dim = [3, shape[0], shape[1], shape[2], 1, 1, 1, 1]
    for i, d in enumerate(dim):
        struct.pack_into("<h", hdr, 40 + i * 2, d)
    if dtype == np.uint8:
        dt, bp = 2, 8
    elif dtype == np.int16:
        dt, bp = 4, 16
    else:
        dt, bp = 16, 32
    struct.pack_into("<h", hdr, 70, dt)
    struct.pack_into("<h", hdr, 72, bp)
    for i, p in enumerate([1.0] * 8):
        struct.pack_into("<f", hdr, 76 + i * 4, p)
    struct.pack_into("<f", hdr, 108, 352.0)
    hdr[344:348] = b'n+1\x00'
    return bytes(hdr) + b'\x00\x00\x00\x00'


def parse_nifti_bytes(raw):
    vox_offset = int(struct.unpack_from("<f", raw, 108)[0])
    hdr = raw[:vox_offset]
    dt_code = struct.unpack_from("<h", hdr, 70)[0]
    dtype = np.uint8 if dt_code == 2 else np.float32
    shape = [
        struct.unpack_from("<h", hdr, 42)[0],
        struct.unpack_from("<h", hdr, 44)[0],
        struct.unpack_from("<h", hdr, 46)[0],
    ]
    data = np.frombuffer(raw[vox_offset:], dtype=dtype).reshape(shape)
    return hdr, data


def save_nifti(array, path):
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    hdr = make_nifti_header(array.shape, array.dtype)
    data = hdr + array.tobytes()
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            f.write(data)
    else:
        with open(path, "wb") as f:
            f.write(data)
    print(f"  Saved: {os.path.basename(path)}")


def read_nifti_file(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return parse_nifti_bytes(f.read())
    else:
        with open(path, "rb") as f:
            return parse_nifti_bytes(f.read())


def otsu_threshold(arr):
    """
    Otsu's method: find the intensity threshold that maximises between-class
    variance for a two-class (background / tissue) split.  Operates on the
    non-zero voxels so the dominant zero-background peak does not bias the
    histogram.  Returns at least 1.0 so the result is always positive.
    """
    nonzero = arr[arr > 0].ravel()
    if nonzero.size == 0:
        return 10.0
    hist, edges = np.histogram(nonzero, bins=256,
                               range=(float(nonzero.min()), float(nonzero.max())))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 10.0
    w0, w1 = 0.0, 1.0
    sum_total = float((hist * centers).sum())
    sum0 = 0.0
    best_thresh, best_var = centers[0], 0.0
    for i in range(len(hist)):
        p = hist[i] / total
        w0 += p
        w1 -= p
        sum0 += p * centers[i]
        if w0 == 0 or w1 == 0:
            continue
        mu0 = sum0 / w0
        mu1 = (sum_total / total - sum0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_thresh = var, centers[i]
    return max(float(best_thresh), 1.0)


def simple_dilate(mask, iterations=3):
    out = mask.copy()
    for _ in range(iterations):
        p = np.pad(out, 1, mode='constant', constant_values=0)
        out = (p[2:,1:-1,1:-1] | p[:-2,1:-1,1:-1] |
               p[1:-1,2:,1:-1] | p[1:-1,:-2,1:-1] |
               p[1:-1,1:-1,2:] | p[1:-1,1:-1,:-2] | out)
    return out


def get_largest_connected_component(mask_np):
    if mask_np.sum() == 0:
        return mask_np
    hdr = make_nifti_header(mask_np.shape, mask_np.dtype)
    res = subprocess.run(
        ["niimath", "-", "-bwlabel", "26", "-"],
        input=hdr + mask_np.tobytes(),
        capture_output=True, check=True,
    )
    _, labeled = parse_nifti_bytes(res.stdout)
    labeled = labeled.reshape(mask_np.shape)
    labels, counts = np.unique(labeled, return_counts=True)
    fg = labels > 0
    if not np.any(fg):
        return mask_np
    largest = labels[fg][np.argmax(counts[fg])]
    return (labeled == largest).astype(np.uint8)


def clean_and_verify_label(arr, name):
    vals, cnts = np.unique(arr, return_counts=True)
    total = arr.size
    for v, c in zip(vals, cnts):
        if v != 0 and (c / total) > 0.5:
            print(f"  Warning: label {name} has dominant class {v} "
                  f"({100*c/total:.1f}% of voxels) — zeroing out.")
            out = arr.copy()
            out[arr == v] = 0
            return out, True
    return arr, False


def harmonize_intensity(aligned_np, subject_np, face_mask_np, p_low=5, p_high=95):
    """
    Linearly rescale the aligned reference face intensities to match the
    subject's face distribution (P5→P95 mapping), computed within face_mask_np.
    Returns float32. Caller handles clipping / dtype conversion.
    """
    ref_v = aligned_np[face_mask_np == 1].astype(np.float32)
    sub_v = subject_np[face_mask_np == 1].astype(np.float32)

    if ref_v.size == 0 or sub_v.size == 0:
        print("  Harmonisation skipped: empty face mask.")
        return aligned_np.astype(np.float32)

    ref_lo, ref_hi = np.percentile(ref_v, [p_low, p_high])
    sub_lo, sub_hi = np.percentile(sub_v, [p_low, p_high])

    print(f"  Intensity harmonisation:")
    print(f"    Reference face  P{p_low}={ref_lo:.1f}  P{p_high}={ref_hi:.1f}")
    print(f"    Subject face    P{p_low}={sub_lo:.1f}  P{p_high}={sub_hi:.1f}")

    ref_range = ref_hi - ref_lo
    if ref_range < 1.0:
        print("  Harmonisation skipped: reference face has negligible intensity range.")
        return aligned_np.astype(np.float32)

    scale = (sub_hi - sub_lo) / ref_range
    shift = sub_lo - ref_lo * scale
    print(f"    Linear map: scale={scale:.4f}  shift={shift:.2f}")

    return aligned_np.astype(np.float32) * scale + shift


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test the corrected facelift preprocessing pipeline on one subject."
    )
    parser.add_argument("--id",          type=int, required=True, help="Subject ID")
    parser.add_argument("--ref",         type=int, default=DEFAULT_REFERENCE_ID,
                        help="Reference subject ID (default: 963)")
    parser.add_argument("--host",        default="127.0.0.1", help="MongoDB host")
    parser.add_argument("--port",        type=int, default=27017)
    parser.add_argument("--force-fetch", action="store_true",
                        help="Re-download all data from MongoDB even if local files exist")
    args = parser.parse_args()

    sid  = args.id
    rid  = args.ref

    t1_path      = os.path.join(WORKSPACE, f"subject_{sid}_t1.nii.gz")
    brain_path   = os.path.join(WORKSPACE, f"subject_{sid}_brain_mask.nii.gz")
    brain_path50 = os.path.join(WORKSPACE, f"subject_{sid}_brain_mask_label50.nii.gz")
    brain_path104= os.path.join(WORKSPACE, f"subject_{sid}_brain_mask_label104.nii.gz")
    ref_path     = os.path.join(WORKSPACE, f"ref_id{rid}.nii.gz")
    aligned_path = os.path.join(WORKSPACE, f"aligned_ref_to_{sid}.nii.gz")

    # -----------------------------------------------------------------------
    # Step 1 — Fetch from MongoDB if needed
    # -----------------------------------------------------------------------
    need_t1       = args.force_fetch or not (os.path.exists(t1_path) and os.path.exists(brain_path))
    need_label50  = args.force_fetch or not os.path.exists(brain_path50)
    need_label104 = args.force_fetch or not os.path.exists(brain_path104)
    need_ref      = args.force_fetch or not os.path.exists(ref_path)

    need_mongo = need_t1 or need_label50 or need_label104 or need_ref

    if need_mongo:
        print(f"\n[1] Connecting to MongoDB at {args.host}:{args.port} ...")
        try:
            client = pymongo.MongoClient(f"mongodb://{args.host}:{args.port}/",
                                         serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            db = client[DB_NAME]
        except Exception as e:
            print(f"  MongoDB connection failed: {e}")
            sys.exit(1)

        if need_t1:
            print(f"  Fetching T1 + label3 for subject {sid} ...")
            t1_tensor = get_sample(sid, "T1", COLLECTION_NAME, db)
            save_nifti(t1_tensor, t1_path)

            l3 = get_sample(sid, "label3", COLLECTION_NAME, db).cpu().numpy()
            l3, _ = clean_and_verify_label(l3, "label3")
            save_nifti((l3 > 0).astype(np.uint8), brain_path)

        if need_label50:
            try:
                print(f"  Fetching label50 for subject {sid} ...")
                l50 = get_sample(sid, "label50", COLLECTION_NAME, db).cpu().numpy()
                l50, _ = clean_and_verify_label(l50, "label50")
                save_nifti((l50 > 0).astype(np.uint8), brain_path50)
            except Exception as e:
                print(f"  Warning: could not fetch label50 ({e}) — skipping.")

        if need_label104:
            try:
                print(f"  Fetching label104 for subject {sid} ...")
                l104 = get_sample(sid, "label104", COLLECTION_NAME, db).cpu().numpy()
                l104, _ = clean_and_verify_label(l104, "label104")
                save_nifti((l104 > 0).astype(np.uint8), brain_path104)
            except Exception as e:
                print(f"  Warning: could not fetch label104 ({e}) — skipping.")

        if need_ref:
            print(f"  Fetching T1 for reference {rid} ...")
            ref_tensor = get_sample(rid, "T1", COLLECTION_NAME, db)
            save_nifti(ref_tensor, ref_path)

        client.close()
    else:
        print(f"\n[1] Using cached local files for subject {sid} and reference {rid}.")

    # -----------------------------------------------------------------------
    # Step 2 — Load subject T1 + brain mask
    # -----------------------------------------------------------------------
    print(f"\n[2] Loading subject T1 and brain mask ...")
    _, t1 = read_nifti_file(t1_path)
    _, brain_mask = read_nifti_file(brain_path)

    brain_mask = (brain_mask > 0)
    print(f"  T1 shape: {t1.shape}  dtype: {t1.dtype}")
    print(f"  Brain mask voxels: {brain_mask.sum()}")

    if brain_mask.sum() == 0:
        print("  Error: brain mask is empty — aborting.")
        sys.exit(1)
    if brain_mask.sum() > (t1.size * 0.5):
        print("  Error: brain mask covers >50% of volume — likely corrupted.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3 — Align reference to subject space
    # -----------------------------------------------------------------------
    need_align = args.force_fetch or not os.path.exists(aligned_path)
    if need_align:
        print(f"\n[3] Running niimath -allineate (reference → subject space) ...")
        hdr_sub = make_nifti_header(t1.shape, t1.dtype)
        try:
            res = subprocess.run(
                ["niimath", ref_path, "-allineate", "-", "-", "-odt", "char"],
                input=hdr_sub + t1.tobytes(),
                capture_output=True, check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  niimath failed: {e.stderr.decode()}")
            sys.exit(1)
        _, aligned_np = parse_nifti_bytes(res.stdout)
        aligned_np = aligned_np.reshape(t1.shape)
        save_nifti(aligned_np.astype(np.uint8), aligned_path)
        print(f"  Alignment complete.")
    else:
        print(f"\n[3] Loading cached aligned reference ...")
        _, aligned_np = read_nifti_file(aligned_path)
        aligned_np = aligned_np.reshape(t1.shape)

    # -----------------------------------------------------------------------
    # Step 4 — Compute face mask (CORRECTED)
    # -----------------------------------------------------------------------
    print(f"\n[4] Computing face mask (corrected pipeline) ...")

    # Brain protection: dilate subject brain mask  (unchanged from original)
    dilated_brain = simple_dilate(brain_mask, iterations=12)
    print(f"  Dilated brain mask voxels: {dilated_brain.sum()}")

    # Otsu thresholding on the subject T1 — automatically finds the optimal
    # background/tissue split regardless of scanner or field strength.
    # More robust than a percentile: when scanner noise produces non-zero
    # background voxels, P2 of non-zero voxels lands near zero and the entire
    # geometric quadrant passes as "face" rather than just the tissue portion.
    head_threshold = otsu_threshold(t1)
    print(f"  Head threshold (Otsu): {head_threshold:.2f}")
    head_mask = (t1 > head_threshold)
    print(f"  Head mask voxels: {head_mask.sum()}")

    # Geometric cutoffs anchored to subject brain center (unchanged from original)
    brain_idx = np.where(brain_mask)
    c_z = float(np.mean(brain_idx[0]))
    c_y = float(np.mean(brain_idx[1]))
    c_x = float(np.mean(brain_idx[2]))
    print(f"  Subject brain CoM (Z, Y, X): ({c_z:.1f}, {c_y:.1f}, {c_x:.1f})")

    z_dim, y_dim, x_dim = t1.shape
    Z, Y, X = np.ogrid[:z_dim, :y_dim, :x_dim]
    geom = (X > (c_x - 35)) & (Y > (c_y - 23))

    # warp_mask: pure geometry — every voxel in the face quadrant that isn't
    # brain. No intensity threshold, no LCC. Reproducible at inference from
    # brain CoM alone, and generous enough to cover any face shape.
    warp_mask = ((~dilated_brain) & geom).astype(np.uint8)
    print(f"  Warp mask voxels (geometric, no intensity filter): {warp_mask.sum()}")

    # ref_face_mask: Otsu on aligned reference + same geometric bounds + LCC.
    # Used only to gate the hybrid transplant — not stored.
    ref_head_threshold = otsu_threshold(aligned_np)
    print(f"  Reference head threshold (Otsu): {ref_head_threshold:.2f}")
    ref_head_mask = (aligned_np > ref_head_threshold)
    ref_initial = (ref_head_mask & (~dilated_brain) & geom).astype(np.uint8)
    print(f"  Reference face mask voxels (before LCC): {ref_initial.sum()}")
    ref_face_mask = get_largest_connected_component(ref_initial)
    print(f"  Reference face mask voxels (after LCC): {ref_face_mask.sum()}")

    if ref_face_mask.sum() == 0:
        print("  Error: reference face mask is empty after LCC — aborting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 5 — Intensity harmonisation
    # -----------------------------------------------------------------------
    print(f"\n[5] Harmonising intensities ...")
    # Compute statistics only over the overlap of ref_face_mask and subject tissue.
    # Using ref_face_mask alone skews P5/P95 with voxels where the subject has
    # background/air but the aligned reference has tissue, producing a bad scale.
    subject_head_mask = (t1 > otsu_threshold(t1))
    harmonize_region = (ref_face_mask & subject_head_mask).astype(np.uint8)
    print(f"  Harmonisation region voxels (ref_face ∩ subject_tissue): {harmonize_region.sum()}")
    aligned_harmonized = harmonize_intensity(aligned_np, t1, harmonize_region)

    # -----------------------------------------------------------------------
    # Step 6 — Save all outputs
    # -----------------------------------------------------------------------
    print(f"\n[6] Saving outputs ...")

    # warp_mask — the generous geometric quadrant for velocity field gating
    save_nifti(warp_mask, os.path.join(WORKSPACE, f"subject_{sid}_warp_mask.nii.gz"))

    # warp_mask highlighted (T1 with warp region set to 255)
    vis_warp = t1.copy()
    vis_warp[warp_mask == 1] = 255
    save_nifti(vis_warp, os.path.join(WORKSPACE, f"subject_{sid}_warp_highlighted.nii.gz"))

    # ref_face_mask highlighted (T1 with reference face region set to 255)
    vis_ref = t1.copy()
    vis_ref[ref_face_mask == 1] = 255
    save_nifti(vis_ref, os.path.join(WORKSPACE, f"subject_{sid}_ref_face_highlighted.nii.gz"))

    # Hybrid: subject T1 with harmonised reference face transplanted at ref_face_mask.
    # Using the reference mask ensures the full reference face is shown — not a
    # version cropped to the subject's own face extent.
    hybrid_np = t1.astype(np.float32).copy()
    hybrid_np[ref_face_mask == 1] = aligned_harmonized[ref_face_mask == 1]
    hybrid_np = np.clip(hybrid_np, 0, 255).astype(np.uint8)
    save_nifti(hybrid_np, os.path.join(WORKSPACE, f"subject_{sid}_hybrid_harmonized.nii.gz"))

    # Raw hybrid (no harmonisation) for comparison
    hybrid_raw = t1.astype(np.float32).copy()
    hybrid_raw[ref_face_mask == 1] = np.clip(aligned_np, 0, 255).astype(np.float32)[ref_face_mask == 1]
    hybrid_raw = np.clip(hybrid_raw, 0, 255).astype(np.uint8)
    save_nifti(hybrid_raw, os.path.join(WORKSPACE, f"subject_{sid}_hybrid_raw.nii.gz"))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\nDone — subject {sid} vs reference {rid}.")
    print(f"Open in ITK-SNAP / FSLeyes to compare:")
    print(f"  subject_{sid}_t1.nii.gz                   — original T1 (model input)")
    print(f"  subject_{sid}_brain_mask.nii.gz           — brain mask from label3")
    print(f"  subject_{sid}_brain_mask_label50.nii.gz   — brain mask from label50 (inspection)")
    print(f"  subject_{sid}_brain_mask_label104.nii.gz  — brain mask from label104 (inspection)")
    print(f"  subject_{sid}_warp_mask.nii.gz            — generous geometric quadrant (no LCC)")
    print(f"  subject_{sid}_warp_highlighted.nii.gz     — warp mask region set to 255")
    print(f"  subject_{sid}_ref_face_highlighted.nii.gz — reference face mask (LCC) set to 255")
    print(f"  subject_{sid}_hybrid_harmonized.nii.gz    — training target (transplant via ref mask)")
    print(f"  subject_{sid}_hybrid_raw.nii.gz           — same without harmonisation (comparison)")


if __name__ == "__main__":
    main()
