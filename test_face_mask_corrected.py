import numpy as np
import struct
import os
import sys
import argparse
import pymongo
import torch
import io
import subprocess

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
    elif dtype == np.float32:
        datatype = 16
        bitpix = 32
    else:
        datatype = 16
        bitpix = 32
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
    print(f"Saved NIfTI to {path}")

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
    print("Running niimath -bwlabel 26 to identify connected components...")
    res = subprocess.run(cmd, input=mask_bytes, capture_output=True, check=True)
    
    # Parse labeled bytes
    _, labeled_np = parse_nifti_bytes(res.stdout)
    
    # Find unique labels and counts (excluding background label 0)
    labels, counts = np.unique(labeled_np, return_counts=True)
    if len(labels) <= 1:
        print("Warning: No connected components found in mask!")
        return mask_np
        
    # Filter out background label 0
    non_bg = (labels > 0)
    labels = labels[non_bg]
    counts = counts[non_bg]
    
    # Find label of largest connected component
    largest_label = labels[np.argmax(counts)]
    print(f"Largest component label: {largest_label} (size: {counts[np.argmax(counts)]} voxels)")
    
    # Keep only the largest component
    largest_mask = (labeled_np == largest_label).astype(np.uint8)
    return largest_mask

def main():
    parser = argparse.ArgumentParser(description="Test and visualize the corrected face mask on any subject")
    parser.add_argument("--id", type=int, default=0, help="Subject ID to test")
    parser.add_argument("--threshold", type=float, default=10.0, help="Threshold to exclude voxels close to zero")
    parser.add_argument("--host", default="127.0.0.1", help="MongoDB host")
    parser.add_argument("--port", type=int, default=27017, help="MongoDB port")
    args = parser.parse_args()

    workspace = "/Users/splis/soft/src/dev/craft/meshnet/facelift"
    t1_path = os.path.join(workspace, f"subject_{args.id}_t1.nii")
    label_path = os.path.join(workspace, f"subject_{args.id}_brain_mask.nii")

    # If files don't exist, fetch them from MongoDB over the SSH tunnel
    if not os.path.exists(t1_path) or not os.path.exists(label_path):
        print(f"Subject {args.id} files not found locally. Connecting to MongoDB to fetch...")
        try:
            client = pymongo.MongoClient(f"mongodb://{args.host}:{args.port}/")
            client.admin.command('ping')
            db = client["MindfulTensors"]
            
            print(f"Fetching T1 and label3 for subject {args.id}...")
            subject_t1 = get_sample(args.id, "T1", "MRN", db)
            subject_label3 = get_sample(args.id, "label3", "MRN", db)
            
            # Save T1
            save_nifti_file(subject_t1, t1_path)
            
            # Save brain mask
            brain_mask_np = (subject_label3.cpu().numpy() > 0).astype(np.uint8)
            save_nifti_file(brain_mask_np, label_path)
        except Exception as e:
            print(f"Error fetching data from MongoDB: {e}")
            sys.exit(1)

    # Load T1 and brain mask from disk
    with open(t1_path, "rb") as f:
        _, t1 = parse_nifti_bytes(f.read())
    with open(label_path, "rb") as f:
        _, brain_mask = parse_nifti_bytes(f.read())

    # Get brain center of mass
    brain_idx = np.where(brain_mask > 0)
    if len(brain_idx[0]) == 0:
        print("Error: Brain mask is empty!")
        sys.exit(1)
        
    c_z = np.mean(brain_idx[0]) # Left-to-Right
    c_y = np.mean(brain_idx[1]) # Inferior-to-Superior (higher Y is Inferior)
    c_x = np.mean(brain_idx[2]) # Posterior-to-Anterior (higher X is Anterior)
    print(f"Subject {args.id} brain center of mass (Z, Y, X): ({c_z:.1f}, {c_y:.1f}, {c_x:.1f})")

    # 1. Dilate brain mask to protect brain + skull
    print("Dilating brain mask (iterations=12)...")
    dilated_brain = simple_dilate(brain_mask > 0, iterations=12)

    # 2. Exclude voxels closer to zero than args.threshold
    thresh = args.threshold
    print(f"Voxel intensity threshold (zeroing out <= {thresh}): {thresh:.1f}")
    head_mask = (t1 > thresh)

    # 3. Compute initial face mask
    # - Inside head mask (non-background / bright voxels)
    # - Outside dilated brain
    # - Anterior: X > c_x - 35 (captures ears!)
    # - Forehead down to chin: Y > c_y - 23 (excludes forehead above ears, chin goes all the way down)
    z_dim, y_dim, x_dim = t1.shape
    Z, Y, X = np.ogrid[:z_dim, :y_dim, :x_dim]

    initial_face_mask = head_mask & (~dilated_brain) & (X > (c_x - 35)) & (Y > (c_y - 23))
    initial_face_mask = initial_face_mask.astype(np.uint8)
    
    # 4. Extract largest connected component using niimath -bwlabel
    face_mask = get_largest_connected_component(initial_face_mask)

    # Save outputs
    mask_out_path = os.path.join(workspace, f"subject_{args.id}_face_mask_corrected.nii")
    vis_out_path = os.path.join(workspace, f"subject_{args.id}_face_highlighted_corrected.nii")
    
    save_nifti_file(face_mask, mask_out_path)

    # Vis
    vis = t1.copy()
    vis[face_mask == 1] = 255
    save_nifti_file(vis, vis_out_path)

    # Check if aligned reference exists to create hybrid target
    aligned_path = os.path.join(workspace, f"aligned_ref_to_{args.id}.nii")
    
    # If not exists, check if ref_id963.nii exists, and we can run alignment on the fly
    ref_path = os.path.join(workspace, "ref_id963.nii")
    if not os.path.exists(aligned_path) and os.path.exists(ref_path):
        print(f"Aligned reference file not found. Running in-memory registration for subject {args.id}...")
        # Prepare subject bytes
        hdr_sub = make_nifti_header(t1.shape, t1.dtype)
        subject_bytes = hdr_sub + t1.tobytes()
        
        cmd = ["niimath", ref_path, "-allineate", "-", "-"]
        try:
            res = subprocess.run(cmd, input=subject_bytes, capture_output=True, check=True)
            _, aligned_ref = parse_nifti_bytes(res.stdout)
            save_nifti_file(aligned_ref, aligned_path)
        except Exception as e:
            print("Failed to run alignment on the fly:", e)
            aligned_ref = None
    elif os.path.exists(aligned_path):
        with open(aligned_path, "rb") as f:
            _, aligned_ref = parse_nifti_bytes(f.read())
    else:
        aligned_ref = None

    if aligned_ref is not None:
        hybrid = t1.copy()
        hybrid[face_mask == 1] = aligned_ref[face_mask == 1].astype(np.uint8)
        hybrid_out_path = os.path.join(workspace, f"subject_{args.id}_hybrid_target_masked_corrected.nii")
        save_nifti_file(hybrid, hybrid_out_path)
        print(f"3. Refaced hybrid target: {os.path.basename(hybrid_out_path)}")

    print(f"\nCompleted face mask test for subject {args.id}!")
    print(f"1. Face mask: {os.path.basename(mask_out_path)}")
    print(f"2. Face highlighted: {os.path.basename(vis_out_path)}")

if __name__ == "__main__":
    main()
