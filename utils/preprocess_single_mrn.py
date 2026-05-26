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

def save_nifti_file(tensor_or_array, path):
    if isinstance(tensor_or_array, torch.Tensor):
        data = tensor_or_array.cpu().numpy()
    else:
        data = tensor_or_array
    hdr = make_nifti_header(data.shape, data.dtype)
    with open(path, "wb") as f:
        f.write(hdr + data.tobytes())
    print(f"Saved NIfTI to {path} (shape: {data.shape}, dtype: {data.dtype})")

def main():
    # Connect to MongoDB via local port forwarding
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
        client.admin.command('ping')
        print("Connected to MongoDB successfully!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        sys.exit(1)

    db = client["MindfulTensors"]

    # 1. Fetch Reference Image: id=1 from MindfulTensors.MRN
    print("Fetching reference T1 (id=1) from MindfulTensors.MRN...")
    try:
        ref_t1 = get_sample(1, "T1", "MRN", db)
        print("Reference T1 shape:", ref_t1.shape)
    except Exception as e:
        print("Failed to fetch reference T1:", e)
        sys.exit(1)

    # 2. Fetch Sample Subject: Let's pick id=0 (or any other subject) from MindfulTensors.MRN
    subject_id = 0
    print(f"Fetching subject T1 and label3 for id={subject_id} from MindfulTensors.MRN...")
    try:
        subject_t1 = get_sample(subject_id, "T1", "MRN", db)
        subject_label3 = get_sample(subject_id, "label3", "MRN", db)
        print("Subject T1 shape:", subject_t1.shape)
        print("Subject label3 shape:", subject_label3.shape)
    except Exception as e:
        print(f"Failed to fetch subject {subject_id} data:", e)
        sys.exit(1)

    output_dir = "/Users/splis/soft/src/dev/craft/meshnet/facelift"
    os.makedirs(output_dir, exist_ok=True)

    # Save original inputs to disk for verification/reference
    ref_path = os.path.join(output_dir, "ref_id1.nii")
    subject_t1_path = os.path.join(output_dir, f"subject_{subject_id}_t1.nii")
    save_nifti_file(ref_t1, ref_path)
    save_nifti_file(subject_t1, subject_t1_path)

    # Generate subject's NIfTI bytes in-memory for piping
    subject_t1_np = subject_t1.cpu().numpy()
    hdr_sub = make_nifti_header(subject_t1_np.shape, subject_t1_np.dtype)
    subject_t1_bytes = hdr_sub + subject_t1_np.tobytes()

    # Perform the registration in-memory
    # Moving image: ref_path (id=1 on disk)
    # Fixed image: piped via stdin
    # Output: piped via stdout
    cmd = ["niimath", ref_path, "-allineate", "-", "-"]
    print(f"Running in-memory niimath registration: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, input=subject_t1_bytes, capture_output=True, check=True)
        print("Registration completed successfully!")
        aligned_ref_bytes = res.stdout
    except subprocess.CalledProcessError as e:
        print("niimath failed with exit code:", e.returncode)
        print("STDERR:", e.stderr.decode())
        sys.exit(1)

    # Parse the aligned reference bytes back to numpy array
    _, aligned_ref_np = parse_nifti_bytes(aligned_ref_bytes)
    
    # Save the aligned reference face NIfTI
    aligned_ref_path = os.path.join(output_dir, f"aligned_ref_to_{subject_id}.nii")
    save_nifti_file(aligned_ref_np, aligned_ref_path)

    # 3. Create Brain Mask and Hybrid Target
    # Brain mask: label3 > 0
    brain_mask_np = (subject_label3.cpu().numpy() > 0).astype(np.uint8)
    brain_mask_path = os.path.join(output_dir, f"subject_{subject_id}_brain_mask.nii")
    save_nifti_file(brain_mask_np, brain_mask_path)

    # Hybrid target = subject_t1 * brain_mask + aligned_ref * (1 - brain_mask)
    # Cast to float32 to do calculations, then clip and cast back to uint8
    subject_t1_float = subject_t1_np.astype(np.float32)
    aligned_ref_float = aligned_ref_np.astype(np.float32)
    brain_mask_float = brain_mask_np.astype(np.float32)

    hybrid_target_np = (subject_t1_float * brain_mask_float + 
                        aligned_ref_float * (1.0 - brain_mask_float))
    hybrid_target_np = np.clip(hybrid_target_np, 0, 255).astype(np.uint8)

    hybrid_target_path = os.path.join(output_dir, f"hybrid_target_for_{subject_id}.nii")
    save_nifti_file(hybrid_target_np, hybrid_target_path)

    print("\nPreproccessing and hybrid target creation completed successfully!")
    print(f"Verify files in {output_dir}:")
    print(f"1. Reference face: {os.path.basename(ref_path)}")
    print(f"2. Original sample: {os.path.basename(subject_t1_path)}")
    print(f"3. Aligned reference: {os.path.basename(aligned_ref_path)}")
    print(f"4. Brain mask: {os.path.basename(brain_mask_path)}")
    print(f"5. Hybrid target (Brain + Aligned Face): {os.path.basename(hybrid_target_path)}")

if __name__ == "__main__":
    main()
