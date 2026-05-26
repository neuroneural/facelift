import pymongo
import torch
import numpy as np
import struct
import io
import os
import sys
import subprocess

def name2collections(name: str, database):
    collection_bin = database[f"{name}.bin"]
    collection_meta = database[f"{name}.meta"]
    return collection_bin, collection_meta

def get_sample(id, kind, collection, db):
    collection_bin, collection_meta = name2collections(collection, db)
    data_cursor = collection_bin.find(
        {"id": id, "kind": kind}, {"chunk_id": 1, "chunk": 1}
    ).sort("chunk_id", 1)
    chunks = [d["chunk"] for d in data_cursor]
    if not chunks:
         raise ValueError(f"No data found for id {id} with kind {kind} in {collection}.bin")
    tensor_binary = b"".join(chunks)
    
    LZ4_MAGIC = b'\x04\x22\x4d\x18'
    if tensor_binary[:4] == LZ4_MAGIC:
        import lz4.frame
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

def save_nifti(tensor, path):
    data = tensor.cpu().numpy()
    hdr = make_nifti_header(data.shape, data.dtype)
    with open(path, "wb") as f:
        f.write(hdr + data.tobytes())
    print(f"Saved NIfTI to {path} (shape: {data.shape}, dtype: {data.dtype})")

def main():
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
        # Test connection
        client.admin.command('ping')
        print("Connected to MongoDB successfully!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        sys.exit(1)

    db_hcp1200z = client["HCP1200z"]
    db_mindful = client["MindfulTensors"]

    # 1. Fetch Reference Image: id=7 from HCP1200z.HCP
    print("Fetching reference T1 (id=7) from HCP1200z...")
    try:
        ref_t1 = get_sample(7, "T1", "HCP", db_hcp1200z)
        print("Reference T1 shape:", ref_t1.shape)
    except Exception as e:
        print("Failed to fetch reference T1:", e)
        sys.exit(1)

    # 2. Fetch Subject Image: Let's pick a subject from MindfulTensors.HCP.
    # Let's find a subject id in MindfulTensors.HCP.meta
    meta_col = db_mindful["HCP.meta"]
    sample_doc = meta_col.find_one()
    if not sample_doc:
        print("No subjects found in MindfulTensors.HCP.meta!")
        sys.exit(1)
    
    subject_id = sample_doc["id"]
    print(f"Selected subject id={subject_id} from MindfulTensors.HCP")

    print(f"Fetching subject T1 and label3 for id={subject_id}...")
    try:
        subject_t1 = get_sample(subject_id, "T1", "HCP", db_mindful)
        print("Subject T1 shape:", subject_t1.shape)
    except Exception as e:
        print("Failed to fetch subject T1:", e)
        sys.exit(1)

    try:
        subject_label3 = get_sample(subject_id, "label3", "HCP", db_mindful)
        print("Subject label3 shape:", subject_label3.shape)
    except Exception as e:
        print("Failed to fetch subject label3:", e)
        sys.exit(1)

    # Output directory
    output_dir = "/Users/splis/soft/src/dev/craft/meshnet/facelift"
    os.makedirs(output_dir, exist_ok=True)

    # Save files
    ref_path = os.path.join(output_dir, "ref_id7.nii")
    subject_t1_path = os.path.join(output_dir, f"subject_{subject_id}_t1.nii")
    subject_label3_path = os.path.join(output_dir, f"subject_{subject_id}_label3.nii")
    aligned_ref_path = os.path.join(output_dir, f"aligned_ref_to_{subject_id}.nii")

    save_nifti(ref_t1, ref_path)
    save_nifti(subject_t1, subject_t1_path)
    save_nifti(subject_label3, subject_label3_path)

    # Run niimath alignment: niimath ref_id7.nii -allineate subject_t1.nii aligned_ref.nii
    cmd = [
        "niimath",
        ref_path,
        "-allineate",
        subject_t1_path,
        aligned_ref_path
    ]
    print(f"Running niimath alignment: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, check=True)
        print("niimath finished successfully!")
        print("STDOUT:", res.stdout.decode() if res.stdout else "")
        print("STDERR:", res.stderr.decode() if res.stderr else "")
    except subprocess.CalledProcessError as e:
        print("niimath failed with exit code:", e.returncode)
        print("STDERR:", e.stderr.decode())
        sys.exit(1)

    print("All preprocessing steps for single subject completed successfully!")

if __name__ == "__main__":
    main()
