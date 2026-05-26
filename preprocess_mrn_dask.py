import io
import os
import sys
import struct
import subprocess
import lz4.frame
import torch
import numpy as np
import pymongo
from pymongo import MongoClient
import dask
from dask.distributed import Client as DaskClient

# MongoDB configuration
MONGOHOST = "localhost"  # default local or port-forwarded host
DB_NAME = "MindfulTensors"
COLLECTION_NAME = "MRN"
REFERENCE_ID = 963

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
            "chunk": pymongo.binary.Binary(chunk),
        }

def simple_dilate(mask, iterations=3):
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode='constant', constant_values=0)
        out = (padded[2:, 1:-1, 1:-1] | padded[:-2, 1:-1, 1:-1] |
               padded[1:-1, 2:, 1:-1] | padded[1:-1, :-2, 1:-1] |
               padded[1:-1, 1:-1, 2:] | padded[1:-1, 1:-1, :-2] | out)
    return out

def get_largest_connected_component(mask_np):
    # Package mask to NIfTI bytes
    hdr = make_nifti_header(mask_np.shape, mask_np.dtype)
    mask_bytes = hdr + mask_np.tobytes()
    
    # Run niimath stdin -bwlabel 26 stdout
    cmd = ["niimath", "-", "-bwlabel", "26", "-"]
    res = subprocess.run(cmd, input=mask_bytes, capture_output=True, check=True)
    
    # Parse labeled bytes
    _, labeled_np = parse_nifti_bytes(res.stdout)
    
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

    # 2. Fetch subject T1 and label3
    try:
        subject_t1 = get_sample(subject_id, "T1", COLLECTION_NAME, db)
        subject_np = subject_t1.cpu().numpy()
    except Exception as e:
        print(f"Error fetching T1 for subject {subject_id}: {e}")
        return {"id": subject_id, "status": "failed", "error": f"fetch_subject_t1: {str(e)}"}

    try:
        subject_label3 = get_sample(subject_id, "label3", COLLECTION_NAME, db)
        label3_np = subject_label3.cpu().numpy()
    except Exception as e:
        print(f"Error fetching label3 for subject {subject_id}: {e}")
        return {"id": subject_id, "status": "failed", "error": f"fetch_subject_label3: {str(e)}"}

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
        
        # Compute face mask (geometrically)
        brain_mask = (label3_np > 0)
        dilated_brain = simple_dilate(brain_mask, iterations=12)
        head_mask = (subject_np > 10)
        
        # Find brain center of mass Y and X coordinates
        brain_indices = np.where(brain_mask)
        y_center = int(np.mean(brain_indices[1])) if len(brain_indices[1]) > 0 else 128
        x_center = int(np.mean(brain_indices[2])) if len(brain_indices[2]) > 0 else 96
        
        # Grid coordinates for masking
        z_dim, y_dim, x_dim = subject_np.shape
        Z, Y, X = np.ogrid[:z_dim, :y_dim, :x_dim]
        
        initial_face_mask = head_mask & (~dilated_brain) & (X > (x_center - 35)) & (Y > (y_center - 23))
        initial_face_mask = initial_face_mask.astype(np.uint8)
        
        # Keep only the largest connected component using niimath -bwlabel 26
        face_mask_np = get_largest_connected_component(initial_face_mask)
        face_mask_tensor = torch.from_numpy(face_mask_np)

        # Cast aligned reference to uint8 and apply face mask to optimize storage compression
        aligned_np = np.clip(aligned_np, 0, 255).astype(np.uint8)
        aligned_np = aligned_np * face_mask_np
        aligned_tensor = torch.from_numpy(aligned_np)

        # Clean any existing chunks of these kinds for idempotency
        col_bin.delete_many({"id": subject_id, "kind": {"$in": ["aligned_ref", "face_mask"]}})

        # Save aligned_ref
        buffer_ref = io.BytesIO()
        torch.save(aligned_tensor, buffer_ref)
        compressed_ref = lz4.frame.compress(buffer_ref.getvalue())
        for chunk in chunk_binobj(compressed_ref, subject_id, "aligned_ref"):
            col_bin.insert_one(chunk)

        # Save face_mask
        buffer_mask = io.BytesIO()
        torch.save(face_mask_tensor, buffer_mask)
        compressed_mask = lz4.frame.compress(buffer_mask.getvalue())
        for chunk in chunk_binobj(compressed_mask, subject_id, "face_mask"):
            col_bin.insert_one(chunk)

        # Update metadata descriptions
        col_meta.update_one(
            {"id": subject_id},
            {
                "$set": {
                    "descriptions.aligned_ref": f"Reference face (id={REFERENCE_ID}) aligned to subject space",
                    "descriptions.face_mask": "Anatomically and geometrically restricted face mask for displacement masking"
                }
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
