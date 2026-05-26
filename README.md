# Face Refacing Model (facelift)

This repository contains the code and utilities for training a MeshNet-based face refacing model. The model deforms any subject's face to match a reference subject (`id=963` from `MindfulTensors.MRN`), while keeping the brain and other non-face head anatomy 100% intact.

## Repository Structure

* **`utils/`**: Preprocessing and diagnostic tools.
  * `preprocess_mrn_dask.py`: Parallel Dask pipeline to register the reference and generate compressed face masks in MongoDB.
  * `test_face_mask_corrected.py`: Standalone local script to test face mask and alignment settings.
  * `export_subject.py`: Local utility to export T1 MRI and mask tensors from MongoDB to NIfTI files.
  * See [utils/README.md](utils/README.md) for detailed usage and setup instructions.

## Next Steps

1. **Database Preprocessing:** Run `utils/preprocess_mrn_dask.py` on the cluster to populate MongoDB with `aligned_ref` and `face_mask` fields.
2. **Model Definition:** Define the browser-deployable 16-channel MeshNet architecture.
3. **Training:** Train the model using unrolled multi-step backpropagation through time (BPTT).
