import pymongo
import torch
import numpy as np
import struct
import io
import os
import sys
import argparse

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
    parser = argparse.ArgumentParser(description="Export a subject T1 and labels from MindfulTensors.MRN to NIfTI files")
    parser.add_argument("--id", type=int, required=True, help="Subject ID to export")
    parser.add_argument("--kind", type=str, default="T1", help="Data kind (T1, label3, label104, label50, aligned_ref)")
    parser.add_argument("--output", type=str, default=None, help="Output file path (defaults to subject_<id>_<kind>.nii)")
    parser.add_argument("--host", default="127.0.0.1", help="MongoDB host")
    parser.add_argument("--port", type=int, default=27017, help="MongoDB port")
    args = parser.parse_args()

    try:
        client = pymongo.MongoClient(f"mongodb://{args.host}:{args.port}/")
        client.admin.command('ping')
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        print("Make sure your SSH tunnel is active (e.g. ssh -o ExitOnForwardFailure=yes -f -N -L 27017:localhost:27017 mongoserver)")
        sys.exit(1)

    db = client["MindfulTensors"]

    # Verify subject exists in metadata
    meta_col = db["MRN.meta"]
    meta_doc = meta_col.find_one({"id": args.id})
    if not meta_doc:
        print(f"Subject with id={args.id} not found in MRN.meta!")
        # Suggest close IDs or show count
        count = meta_col.count_documents({})
        print(f"Total subjects in MRN: {count}")
        sys.exit(1)

    print(f"Found subject id={args.id} in metadata:")
    print(meta_doc)

    print(f"Fetching kind '{args.kind}'...")
    try:
        tensor = get_sample(args.id, args.kind, "MRN", db)
    except Exception as e:
        print(f"Failed to fetch kind '{args.kind}' for subject {args.id}: {e}")
        sys.exit(1)

    output_path = args.output
    if not output_path:
        output_path = f"subject_{args.id}_{args.kind}.nii"

    save_nifti_file(tensor, output_path)
    print("Export complete!")

if __name__ == "__main__":
    main()
