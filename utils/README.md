# Face Refacing Auxiliary and Preprocessing Tools

This directory contains utility and preprocessing scripts for the Face Refacing project (deforming MRI face areas using a 16-channel MeshNet autoencoder while keeping the brain and other skull structures 100% anatomically intact).

## Preprocessing Pipeline

### `preprocess_mrn_dask.py`
A parallel Dask-enabled MongoDB pipeline script that processes all subjects in the `MindfulTensors.MRN` collection.
* **Alignment:** Registers the reference face (Subject `963`) to each subject's space in-memory using `niimath -allineate`.
* **Anatomical Face Mask:** Computes a restricted face mask using a combination of brain mask dilation (12 iterations), voxel intensity thresholding (excluding background voxels $\le 10$), relative coordinates from the brain's center of mass, and connected component filtering using `niimath -bwlabel 26`.
* **Storage Optimization:** Applies the binary face mask to the aligned reference volume before storing it. Because everything outside the face is zeroed out, the LZ4-compressed size drops from **~12MB** per subject to **<1MB**.
* **Database Updates:** Saves the compressed `aligned_ref` and `face_mask` tensors back to MongoDB and updates metadata descriptions.

## Local Test and Diagnostic Scripts

These scripts are useful for verifying the pipeline logic, debugging face mask bounds, and checking registration quality on individual subjects before launching the full Dask cluster job.

### `test_face_mask_corrected.py`
A standalone diagnostic script to compute and visualize the face mask and hybrid target locally.
* **Usage:**
  ```bash
  python3 test_face_mask_corrected.py --id <subject_id> --threshold 10
  ```
* **Process:** 
  1. Fetches T1 and label3 (brain mask) for the subject from MongoDB over the SSH tunnel.
  2. Dilates the brain mask to protect internal head anatomy.
  3. Excludes near-zero voxels (intensity $\le 10$).
  4. Applies coordinates bounds ($X > c_x - 35$ and $Y > c_y - 23$) to capture the face, ears, and chin.
  5. Computes the largest connected component using `niimath -bwlabel 26` to eliminate noise/air pocket voxels.
  6. Aligns the reference (id=963) and generates a `subject_<id>_hybrid_target_masked_corrected.nii` file to inspect face refacing quality.

### `export_subject.py`
A tool to export specific subject data fields (e.g. `T1`, `label3`, `label104`, `aligned_ref`, `face_mask`) from MongoDB to local NIfTI `.nii` files.
* **Usage:**
  ```bash
  python3 export_subject.py --id <subject_id> --kind <T1|label3|aligned_ref|face_mask> --output <filename.nii>
  ```

### `preprocess_single_mrn.py`
A simplified single-subject registration and hybrid target script.

---

## Setting up MongoDB Tunnel

To run the local scripts, ensure you establish an SSH tunnel to forward the MongoDB port:
```bash
ssh -o ExitOnForwardFailure=yes -f -N -L 27017:localhost:27017 mongoserver
```
