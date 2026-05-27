import io
import os
import sys
import struct
import subprocess
import lz4.frame
import torch
import numpy as np
import bson
from pymongo import MongoClient
import dask
from dask.distributed import Client as DaskClient

# MongoDB configuration
MONGOHOST = "localhost"  # default local or port-forwarded host
DB_NAME = "MindfulTensors"
COLLECTION_NAME = "MRN"
REFERENCE_ID = 962

def name2collections(name: str, database):
    collection_bin = database[f"{name}.bin"]
    collection_meta = database[f"{name}.meta"]
    return collection_bin, collection_meta

def get_sample(subject_id, kind, collection, db):
    collection_bin, _ = name2collections(collection, db)
    data_cursor = collection_bin.find(
        {"id": subject_id, "kind": kind}, {"chunk_id": 1, "chunk": 1}
    ).sort("chunk_id", 1)
    chunks = [d["chunk"] for d in data_cursor]
    if not chunks:
         raise ValueError(f"No data found for id {subject_id} with kind {kind} in {collection}.bin")
    tensor_binary = b"".join(chunks)
    
    LZ4_MAGIC = b'\x04\x22\x4d\x18'
    if tensor_binary[:4] == LZ4_MAGIC:
        tensor_binary = lz4.frame.decompress(tensor_binary)
    buffer = io.BytesIO(tensor_binary)
    tensor = torch.load(buffer, weights_only=True)
    return tensor

def make_nifti_header(shape, dtype):
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    dim = [3, shape[0], shape[1], shape[2], 1, 1, 1, 1]
    for i, d in enumerate(dim):
        struct.pack_into("<h", header, 40 + i*2, d)
    if dtype == np.uint8:
        datatype = 2
        bitpix = 8
    elif dtype == np.int16:
        datatype = 4
        bitpix = 16
    elif dtype == np.float32:
        datatype = 16
        bitpix = 32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    struct.pack_into("<h", header, 70, datatype)
    struct.pack_into("<h", header, 72, bitpix)
    pixdim = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i, p in enumerate(pixdim):
        struct.pack_into("<f", header, 76 + i*4, p)
    struct.pack_into("<f", header, 108, 352.0)
    header[344:348] = b'n+1\x00'
    return bytes(header) + b'\x00\x00\x00\x00'

def parse_nifti_bytes(raw_bytes):
    vox_offset = int(struct.unpack_from("<f", raw_bytes, 108)[0])
    header = raw_bytes[:vox_offset]
    data_bytes = raw_bytes[vox_offset:]
    dt_code = struct.unpack_from("<h", header, 70)[0]
    dtype = np.uint8 if dt_code == 2 else np.float32
    shape = [
        struct.unpack_from("<h", header, 42)[0],
        struct.unpack_from("<h", header, 44)[0],
        struct.unpack_from("<h", header, 46)[0]
    ]
    data = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    return header, data

def chunk_binobj(tensor_compressed, subject_id, kind, chunksize_mb=12):
    chunksize_bytes = chunksize_mb * 1024 * 1024
    num_chunks = len(tensor_compressed) // chunksize_bytes
    if len(tensor_compressed) % chunksize_bytes != 0:
        num_chunks += 1
    for i in range(num_chunks):
        start = i * chunksize_bytes
        end = min((i + 1) * chunksize_bytes, len(tensor_compressed))
        chunk = tensor_compressed[start:end]
        yield {
            "id": subject_id,
            "chunk_id": i,
            "kind": kind,
            "chunk": bson.Binary(chunk),
        }

def tensor2bin_compressed(tensor):
    """Serialize tensor to LZ4-compressed binary."""
    buffer = io.BytesIO()
    torch.save(tensor.to(torch.uint8), buffer)
    return lz4.frame.compress(buffer.getvalue())

def otsu_threshold(arr):
    """
    Otsu's method: find the intensity threshold that maximises between-class
    variance for a two-class (background / tissue) split.  Operates on the
    non-zero voxels only so that the dominant zero-background peak does not
    bias the histogram.  Returns at least 1.0 so the result is always positive.
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


def harmonize_intensity(aligned_np, subject_np, face_mask_np, p_low=5, p_high=95):
    """
    Linearly rescale the aligned reference face intensities to match the subject's
    face intensity distribution, using a percentile-pair (p_low, p_high) mapping
    for robustness against outliers.

    Statistics are computed only within the face mask. The result is returned as
    float32 — the caller is responsible for any downstream clipping / dtype cast.
    """
    ref_voxels = aligned_np[face_mask_np == 1].astype(np.float32)
    sub_voxels = subject_np[face_mask_np == 1].astype(np.float32)

    if ref_voxels.size == 0 or sub_voxels.size == 0:
        return aligned_np.astype(np.float32)

    ref_lo, ref_hi = np.percentile(ref_voxels, [p_low, p_high])
    sub_lo, sub_hi = np.percentile(sub_voxels, [p_low, p_high])

    ref_range = ref_hi - ref_lo
    if ref_range < 1.0:
        # Reference face has essentially no intensity variation — skip rescaling
        # to avoid a degenerate scale factor.
        return aligned_np.astype(np.float32)

    scale = (sub_hi - sub_lo) / ref_range
    shift = sub_lo - ref_lo * scale

    return aligned_np.astype(np.float32) * scale + shift


def simple_dilate(mask, iterations=3):
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode='constant', constant_values=0)
        out = (padded[2:, 1:-1, 1:-1] | padded[:-2, 1:-1, 1:-1] |
               padded[1:-1, 2:, 1:-1] | padded[1:-1, :-2, 1:-1] |
               padded[1:-1, 1:-1, 2:] | padded[1:-1, 1:-1, :-2] | out)
    return out

def get_largest_connected_component(mask_np):
    # If mask is already empty, return it directly to avoid niimath error
    if mask_np.sum() == 0:
        return mask_np
    # Package mask to NIfTI bytes
    hdr = make_nifti_header(mask_np.shape, mask_np.dtype)
    mask_bytes = hdr + mask_np.tobytes()
    
    # Run niimath stdin -bwlabel 26 stdout
    cmd = ["niimath", "-", "-bwlabel", "26", "-"]
    res = subprocess.run(cmd, input=mask_bytes, capture_output=True, check=True)
    
    # Parse labeled bytes using local parser
    _, labeled_np = parse_nifti_bytes(res.stdout)
    labeled_np = labeled_np.reshape(mask_np.shape)
    
    # Find unique labels and counts (excluding background label 0)
    labels, counts = np.unique(labeled_np, return_counts=True)
    if len(labels) <= 1:
        return mask_np
        
    # Filter out background label 0
    non_bg = (labels > 0)
    labels = labels[non_bg]
    counts = counts[non_bg]
    
    # Find label of largest connected component
    largest_label = labels[np.argmax(counts)]
    
    # Keep only the largest component
    largest_mask = (labeled_np == largest_label).astype(np.uint8)
    return largest_mask

def clean_and_verify_label(tensor, label_name):
    """
    Check if a label tensor has a dominant non-zero class (>50% of voxels).
    If so, zero it out and return (cleaned_tensor, True).
    Otherwise, return (tensor, False).
    """
    unique_vals, counts = torch.unique(tensor, return_counts=True)
    total_voxels = tensor.numel()
    
    for val, count in zip(unique_vals, counts):
        val_item = val.item()
        pct = (count.item() / total_voxels) * 100
        if val_item != 0 and pct > 50.0:
            print(f"Warning: Subject label {label_name} has dominant class {val_item} covering {pct:.2f}% of volume. Zeroing it out.")
            cleaned = tensor.clone()
            cleaned[tensor == val] = 0
            return cleaned, True
    return tensor, False

def process_subject(subject_id, db_host=MONGOHOST):
    # Connect to MongoDB inside the worker process
    client = MongoClient(f"mongodb://{db_host}:27017/")
    db = client[DB_NAME]
    col_bin, col_meta = name2collections(COLLECTION_NAME, db)

    # 1. Ensure reference T1 is written to local /tmp/ of the worker node
    ref_path = f"/tmp/ref_id{REFERENCE_ID}.nii"
    if not os.path.exists(ref_path):
        try:
            ref_t1 = get_sample(REFERENCE_ID, "T1", COLLECTION_NAME, db)
            ref_np = ref_t1.cpu().numpy()
            hdr_ref = make_nifti_header(ref_np.shape, ref_np.dtype)
            with open(ref_path, "wb") as f:
                f.write(hdr_ref + ref_np.tobytes())
        except Exception as e:
            print(f"Error fetching reference T1 in worker for subject {subject_id}: {e}")
            return {"id": subject_id, "status": "failed", "error": f"fetch_reference: {str(e)}"}

    # 2. Fetch subject T1 and labels (label3, label50, label104, volumes104)
    try:
        subject_t1 = get_sample(subject_id, "T1", COLLECTION_NAME, db)
        subject_np = subject_t1.cpu().numpy()
    except Exception as e:
        print(f"Error fetching T1 for subject {subject_id}: {e}")
        return {"id": subject_id, "status": "failed", "error": f"fetch_subject_t1: {str(e)}"}

    try:
        subject_label3 = get_sample(subject_id, "label3", COLLECTION_NAME, db)
    except Exception as e:
        print(f"Error fetching label3 for subject {subject_id}: {e}")
        return {"id": subject_id, "status": "failed", "error": f"fetch_subject_label3: {str(e)}"}

    try:
        subject_label50 = get_sample(subject_id, "label50", COLLECTION_NAME, db)
    except Exception as e:
        print(f"Error fetching label50 for subject {subject_id}: {e}")
        return {"id": subject_id, "status": "failed", "error": f"fetch_subject_label50: {str(e)}"}

    try:
        subject_label104 = get_sample(subject_id, "label104", COLLECTION_NAME, db)
    except Exception as e:
        print(f"Error fetching label104 for subject {subject_id}: {e}")
        return {"id": subject_id, "status": "failed", "error": f"fetch_subject_label104: {str(e)}"}

    try:
        subject_volumes104 = get_sample(subject_id, "volumes104", COLLECTION_NAME, db)
    except Exception as e:
        print(f"Error fetching volumes104 for subject {subject_id}: {e}")
        return {"id": subject_id, "status": "failed", "error": f"fetch_subject_volumes104: {str(e)}"}

    # Clean labels on the fly if corrupted
    cleaned_label3, l3_modified = clean_and_verify_label(subject_label3, "label3")
    cleaned_label50, l50_modified = clean_and_verify_label(subject_label50, "label50")
    cleaned_label104, l104_modified = clean_and_verify_label(subject_label104, "label104")

    if l104_modified:
        # volumes104 = (bincount(cleaned_label104) % 256) cast to uint8
        counts = torch.bincount(cleaned_label104.flatten(), minlength=104)
        cleaned_volumes104 = (counts % 256).to(torch.uint8)
        v104_modified = True
    else:
        cleaned_volumes104 = subject_volumes104
        v104_modified = False

    # 3. Align reference T1 to subject T1 in-memory
    hdr_sub = make_nifti_header(subject_np.shape, subject_np.dtype)
    subject_bytes = hdr_sub + subject_np.tobytes()

    cmd = ["niimath", ref_path, "-allineate", "-", "-"]
    try:
        res = subprocess.run(cmd, input=subject_bytes, capture_output=True, check=True)
        aligned_bytes = res.stdout
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode() if e.stderr else "unknown error"
        print(f"niimath failed for subject {subject_id}: {stderr_msg}")
        return {"id": subject_id, "status": "failed", "error": f"niimath: {stderr_msg}"}

    # 4. Parse the aligned reference bytes and save back to MongoDB
    # Also compute face mask and save back to MongoDB
    try:
        _, aligned_np = parse_nifti_bytes(aligned_bytes)
        aligned_np = aligned_np.reshape(subject_np.shape)
        
        # Compute face mask (geometrically) using cleaned label3
        label3_np = cleaned_label3.cpu().numpy()
        brain_mask = (label3_np > 0)
        
        # Check for corrupted brain masks (covering too much of the volume or empty)
        if brain_mask.sum() > (subject_np.size * 0.5):
            print(f"Warning: Subject {subject_id} has abnormally large brain mask ({brain_mask.sum()} voxels). Skipping.")
            return {"id": subject_id, "status": "failed", "error": "corrupted_brain_mask"}
        if brain_mask.sum() == 0:
            print(f"Warning: Subject {subject_id} has empty brain mask. Skipping.")
            return {"id": subject_id, "status": "failed", "error": "empty_brain_mask"}
            
        dilated_brain = simple_dilate(brain_mask, iterations=12)

        # Brain CoM for geometric bounds
        brain_indices = np.where(brain_mask)
        y_center = int(np.mean(brain_indices[1])) if len(brain_indices[1]) > 0 else 128
        x_center = int(np.mean(brain_indices[2])) if len(brain_indices[2]) > 0 else 96

        z_dim, y_dim, x_dim = subject_np.shape
        Z, Y, X = np.ogrid[:z_dim, :y_dim, :x_dim]
        geom = (X > (x_center - 35)) & (Y > (y_center - 23))

        # warp_mask: pure geometry — every voxel in the face quadrant that isn't
        # brain. No intensity threshold, no LCC. Reproducible at inference from
        # brain CoM alone, and generous enough to cover any face shape.
        warp_mask_np = ((~dilated_brain) & geom).astype(np.uint8)
        warp_mask_tensor = torch.from_numpy(warp_mask_np)

        # ref_face_mask: Otsu on the aligned reference + same geometric bounds + LCC.
        # Used only to gate the hybrid transplant — not stored.
        ref_head_mask = (aligned_np > otsu_threshold(aligned_np))
        ref_initial = (ref_head_mask & (~dilated_brain) & geom).astype(np.uint8)
        ref_face_mask_np = get_largest_connected_component(ref_initial)
        if ref_face_mask_np.sum() == 0:
            print(f"Warning: Subject {subject_id} has empty reference face mask. Skipping.")
            return {"id": subject_id, "status": "failed", "error": "empty_ref_face_mask"}

        # Harmonize within the overlap of ref_face_mask and subject tissue —
        # avoids skewing P5/P95 with background voxels where the subject has air
        # but the reference has tissue.
        subject_head_mask = (subject_np > otsu_threshold(subject_np))
        harmonize_region = ref_face_mask_np & subject_head_mask
        aligned_harmonized = harmonize_intensity(aligned_np, subject_np, harmonize_region)

        # Build the hybrid: subject T1 with the harmonized reference face
        # transplanted at ref_face_mask (reference's own extent, not subject's).
        # Training target: L1(warp(T1, velocity), hybrid) over the full volume.
        hybrid_np = subject_np.astype(np.float32).copy()
        hybrid_np[ref_face_mask_np == 1] = aligned_harmonized[ref_face_mask_np == 1]
        hybrid_np = np.clip(hybrid_np, 0, 255).astype(np.uint8)
        hybrid_tensor = torch.from_numpy(hybrid_np)

        # Delete existing chunks for all kinds we are about to write, plus
        # stale kinds from previous pipeline versions.
        kinds_to_delete = ["hybrid", "warp_mask", "face_mask", "aligned_ref"]
        if l3_modified:
            kinds_to_delete.append("label3")
        if l50_modified:
            kinds_to_delete.append("label50")
        if l104_modified:
            kinds_to_delete.extend(["label104", "volumes104"])

        col_bin.delete_many({"id": subject_id, "kind": {"$in": kinds_to_delete}})

        # Save hybrid
        compressed_hybrid = tensor2bin_compressed(hybrid_tensor)
        for chunk in chunk_binobj(compressed_hybrid, subject_id, "hybrid", 12):
            col_bin.insert_one(chunk)

        # Save warp_mask (generous geometric quadrant; gates velocity field at
        # training and inference; compresses very well due to large zero regions)
        compressed_warp = tensor2bin_compressed(warp_mask_tensor)
        for chunk in chunk_binobj(compressed_warp, subject_id, "warp_mask", 12):
            col_bin.insert_one(chunk)

        # Save cleaned labels if they were modified
        if l3_modified:
            compressed_l3 = tensor2bin_compressed(cleaned_label3)
            for chunk in chunk_binobj(compressed_l3, subject_id, "label3", 12):
                col_bin.insert_one(chunk)
            print(f"Saved cleaned label3 for subject {subject_id} to database.")

        if l50_modified:
            compressed_l50 = tensor2bin_compressed(cleaned_label50)
            for chunk in chunk_binobj(compressed_l50, subject_id, "label50", 12):
                col_bin.insert_one(chunk)
            print(f"Saved cleaned label50 for subject {subject_id} to database.")

        if l104_modified:
            compressed_l104 = tensor2bin_compressed(cleaned_label104)
            for chunk in chunk_binobj(compressed_l104, subject_id, "label104", 12):
                col_bin.insert_one(chunk)
            compressed_v104 = tensor2bin_compressed(cleaned_volumes104)
            for chunk in chunk_binobj(compressed_v104, subject_id, "volumes104", 12):
                col_bin.insert_one(chunk)
            print(f"Saved cleaned label104 and volumes104 for subject {subject_id} to database.")

        # Update metadata — remove stale aligned_ref description, add hybrid
        col_meta.update_one(
            {"id": subject_id},
            {
                "$set": {
                    "descriptions.hybrid": (
                        f"Subject T1 with harmonized reference face (id={REFERENCE_ID}) "
                        f"transplanted at the reference face mask. Training target for displacement model."
                    ),
                    "descriptions.warp_mask": (
                        "Generous geometric quadrant mask (no LCC, no intensity threshold). "
                        "Gates the displacement field at training and inference. "
                        "Derivable from brain CoM alone — no reference needed."
                    ),
                },
                "$unset": {
                    "descriptions.aligned_ref": "",
                    "descriptions.face_mask": "",
                },
            }
        )
    except Exception as e:
        print(f"Error saving aligned reference and mask for subject {subject_id}: {e}")
        return {"id": subject_id, "status": "failed", "error": f"save_db: {str(e)}"}

    return {"id": subject_id, "status": "success"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dask-based preprocessing pipeline for MRN face refacing")
    parser.add_argument("--host", default=MONGOHOST, help="MongoDB host address")
    parser.add_argument("--n_workers", type=int, default=None, help="Number of Dask workers")
    args = parser.parse_args()

    # Setup Dask Client
    dask_client = DaskClient(n_workers=args.n_workers)
    print(f"Dask Client: {dask_client}")

    # Fetch all subject IDs from MindfulTensors.MRN.meta
    client = MongoClient(f"mongodb://{args.host}:27017/")
    db = client[DB_NAME]
    _, col_meta = name2collections(COLLECTION_NAME, db)
    
    # We query all IDs except the reference ID (since reference aligned to itself is identity, but we can do it too)
    subject_docs = col_meta.find({}, {"id": 1}).sort("id", 1)
    subject_ids = [doc["id"] for doc in subject_docs]
    client.close()

    print(f"Found {len(subject_ids)} subjects to process.")

    # Submit tasks via Dask
    futures = []
    for sid in subject_ids:
        # Submit delayed task
        future = dask.delayed(process_subject)(sid, db_host=args.host)
        futures.append(future)

    print("Starting parallel processing...")
    results = dask.compute(*futures)

    # Process and summarize results
    succeeded = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(succeeded)} / {len(results)}")
    print(f"Failed: {len(failed)} / {len(results)}")
    if failed:
        print("First 10 failures:")
        for f in failed[:10]:
            print(f"Subject {f['id']}: {f['error']}")
