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
REFERENCE_ID = 1

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

    # 2. Fetch subject T1
    try:
        subject_t1 = get_sample(subject_id, "T1", COLLECTION_NAME, db)
        subject_np = subject_t1.cpu().numpy()
    except Exception as e:
        print(f"Error fetching T1 for subject {subject_id}: {e}")
        return {"id": subject_id, "status": "failed", "error": f"fetch_subject: {str(e)}"}

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
    try:
        _, aligned_np = parse_nifti_bytes(aligned_bytes)
        
        # Cast aligned reference to uint8 to match T1 datatype and save space
        aligned_np = np.clip(aligned_np, 0, 255).astype(np.uint8)
        aligned_tensor = torch.from_numpy(aligned_np)

        # Compress to LZ4
        buffer = io.BytesIO()
        torch.save(aligned_tensor, buffer)
        compressed_bytes = lz4.frame.compress(buffer.getvalue())

        # Clean any existing chunks of this kind for idempotency
        col_bin.delete_many({"id": subject_id, "kind": "aligned_ref"})

        # Insert new chunks
        for chunk in chunk_binobj(compressed_bytes, subject_id, "aligned_ref"):
            col_bin.insert_one(chunk)

        # Update metadata descriptions if not already updated
        col_meta.update_one(
            {"id": subject_id},
            {
                "$set": {
                    "descriptions.aligned_ref": "Reference face (id=1) aligned to subject space"
                }
            }
        )
    except Exception as e:
        print(f"Error saving aligned reference for subject {subject_id}: {e}")
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
