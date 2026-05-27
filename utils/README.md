# Face Refacing Auxiliary and Preprocessing Tools

This directory contains utility and preprocessing scripts for the Face Refacing project — deforming the face region of an MRI T1 volume to match a reference subject (id=963), while preserving the brain and all non-face anatomy exactly.

---

## Database Schema (`MindfulTensors.MRN`)

Each subject has several stored kinds. The ones directly relevant to training are:

| Kind | Shape / dtype | Description |
|------|--------------|-------------|
| `T1` | `(Z,Y,X)` uint8 | Raw subject T1 — the model input |
| `hybrid` | `(Z,Y,X)` uint8 | Pre-computed training target (see below) |
| `face_mask` | `(Z,Y,X)` uint8 | Binary face region mask (see below) |
| `label3` | `(Z,Y,X)` uint8 | Gray/white matter segmentation (3 classes) |
| `label50` | `(Z,Y,X)` uint8 | Mindboggle atlas (50 classes) |
| `label104` | `(Z,Y,X)` uint8 | FreeSurfer Destrieux atlas (104 classes) |

### Why `hybrid` and not `aligned_ref`

The naive approach would be to store the aligned reference T1 (after `niimath -allineate` maps id=963 into subject space) and compose the training target on the fly. We rejected this for two reasons:

1. **No stenciling problem.** If we stored `aligned_ref * face_mask` we would introduce artificial zero targets in any part of the face mask that the reference volume doesn't fill perfectly. The model would be penalized for not suppressing real tissue. Storing the full hybrid avoids this entirely.
2. **Training time.** Each epoch would need to load `T1`, `aligned_ref`, and `face_mask`, compute the blend, and harmonize intensities — all on the critical path. Pre-computing the hybrid means each epoch needs only `T1` and `hybrid`, both already uint8 tensors.

### What `hybrid` contains

The hybrid volume is computed once during preprocessing:

```
hybrid = subject_T1.copy()
hybrid[face_mask == 1] = harmonize(aligned_ref)[face_mask == 1]
```

Inside the face mask the hybrid carries the reference subject's anatomy, harmonized to the subject's local intensity range. Outside the face mask it is byte-for-byte identical to the subject's own T1. The model never needs to know where the boundary is — the hybrid simply tells it "this is what your output should look like."

### Intensity harmonization

Before transplanting the reference face, a P5→P95 linear rescale is applied within the face mask:

```
scale = (sub_P95 - sub_P5) / (ref_P95 - ref_P5)
shift = sub_P5 - ref_P5 * scale
harmonized = aligned_ref * scale + shift
```

This corrects for field strength differences and scanner gain without altering the reference anatomy's spatial structure.

### What `face_mask` is used for

The face mask is **not** used in the training loss. It is stored for two purposes:

- **Inference-time gating.** At inference, the predicted velocity field is hard-zeroed outside the face mask before integration: `velocity[face_mask == 0] = 0`. This is a safety guarantee — no matter what the model predicts, the brain and skull cannot be displaced.
- **Validation metrics.** Per-region metrics (face SSIM, brain SSIM) require knowing which voxels belong to the face.

---

## Face Mask Computation

The face mask is derived entirely from the **subject's own** anatomy. The reference is never used to derive the mask, because the aligned reference fills its entire resampled bounding box with scanner noise and interpolation artefacts, making tissue-vs-background separation unreliable.

Pipeline:

1. **Head mask** — threshold the subject T1 at the 2nd percentile of non-zero voxels. This robustly separates tissue from background without a hard magic number.
2. **Brain mask** — binarize `label3 > 0` (any segmented voxel = brain).
3. **Dilated brain** — dilate brain mask by 12 voxels (6-connectivity) to protect skull and meninges. Anything inside this shell is excluded from the face mask.
4. **Geometric bounds** — restrict to the anterior/inferior quadrant:
   - `X > c_x - 35` (captures face and ears, excludes back of head)
   - `Y > c_y - 23` (captures chin through forehead, excludes very top of skull)
   where `(c_z, c_y, c_x)` is the brain center of mass.
5. **Initial mask** = `head_mask AND NOT dilated_brain AND geometric_bounds`
6. **Largest connected component** — pipe through `niimath - -bwlabel 26 -` to eliminate isolated air-pocket voxels and background fragments that pass the geometric filter.

---

## Training Design

### Loss function

```
L = L1(warp(T1, velocity_field), hybrid) + λ · smoothness(velocity_field)
```

- `warp(T1, v)` applies the diffeomorphic displacement field obtained by scaling-and-squaring the stationary velocity field `v` predicted by MeshNet.
- `hybrid` is the pre-stored target.
- `smoothness` is a gradient-magnitude penalty on `v` that prevents folding.
- No masking is applied to the loss — it is computed over the full volume.

### Why no mask in the loss

The hybrid encodes brain preservation implicitly. Inside the brain the hybrid is identical to the subject T1, so warping any brain voxel away from its resting position increases the L1 loss. The model is punished for touching the brain without any explicit mask. In the face region the hybrid differs from the subject T1, providing a gradient signal that drives the face transformation. An explicit face-region mask in the loss would be redundant and would remove the implicit brain-preservation signal from non-face voxels.

### Smoothness regularizer

The regularizer penalizes the spatial gradient of the velocity field, encouraging smooth deformations. λ is a hyperparameter; a value around 0.1–1.0 is a reasonable starting point depending on the scale of the velocity field.

### Inference

At inference the model receives only `T1` and produces a velocity field. A brain mask is computed from the inference subject (using the same MeshNet segmenter or an equivalent tool), dilated to a face mask, and the velocity field is zeroed outside the face mask before integration. This hard constraint ensures the displacement never touches the brain regardless of what the model predicts.

---

## Scripts

### `preprocess_mrn_dask.py`

Parallel Dask pipeline that processes all subjects in `MindfulTensors.MRN`. For each subject it:

1. Fetches `T1` and `label3` from MongoDB.
2. Computes the face mask using the pipeline described above.
3. Aligns the reference T1 (id=963) to subject space via `niimath -allineate` (in-memory, stdin→stdout).
4. Harmonizes the aligned reference intensities to the subject's face region statistics.
5. Composites the hybrid volume.
6. Stores `hybrid` and `face_mask` back to MongoDB; removes any stale `aligned_ref` documents.

### `test_corrected_pipeline.py` (top-level)

Standalone diagnostic script for inspecting the pipeline on a single subject. Writes `.nii.gz` outputs to the workspace:

```bash
python3 test_corrected_pipeline.py --id <subject_id>
python3 test_corrected_pipeline.py --id <subject_id> --force-fetch   # re-download from MongoDB
```

Outputs:
- `subject_<id>_brain_mask.nii.gz` — from label3
- `subject_<id>_brain_mask_label50.nii.gz` — from label50
- `subject_<id>_brain_mask_label104.nii.gz` — from label104
- `subject_<id>_face_mask.nii.gz` — computed face mask
- `subject_<id>_face_highlighted.nii.gz` — T1 with face region set to 255 (QC overlay)
- `subject_<id>_hybrid_harmonized.nii.gz` — final hybrid target (with harmonization)
- `subject_<id>_hybrid_raw.nii.gz` — hybrid without intensity harmonization (for comparison)

### `test_face_mask_corrected.py` (utils/)

Earlier diagnostic script (superseded by `test_corrected_pipeline.py`). Retained for reference.

### `export_subject.py`

Exports a specific kind for a subject from MongoDB to a local NIfTI file:

```bash
python3 export_subject.py --id <subject_id> --kind <T1|label3|hybrid|face_mask> --output <filename.nii>
```

### `preprocess_single_mrn.py`

Simplified single-subject version of the preprocessing pipeline, useful for testing changes before running the full Dask job.

---

## Setting up MongoDB Tunnel

```bash
ssh -o ExitOnForwardFailure=yes -f -N -L 27017:localhost:27017 mongoserver
```
