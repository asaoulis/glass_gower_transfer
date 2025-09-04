import os
import tarfile
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Settings
input_dir = Path("/share/gpu5/asaoulis/gowerstreet")
max_workers = 58  # Adjust depending on disk load and CPU
min_files_expected = 90  # Change if you expect more per archive

def extract_tar(tar_path):
    base_name = tar_path.name.replace(".tar.gz", "")  # sim00066.tar.gz -> sim00066
    extract_dir = tar_path.parent
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            extract_dir.mkdir(parents=True, exist_ok=True)
            tar.extractall(path=extract_dir)

        # Integrity check: Ensure files were actually written
        file_count = sum(1 for _ in extract_dir.rglob('*') if _.is_file())
        if file_count < min_files_expected:
            return f"⚠️  {tar_path.name}: Only {file_count} files extracted (check integrity)"

        return f"✅ Extracted {tar_path.name} to {extract_dir.name} ({file_count} files)"
    except Exception as e:
        return f"❌ Failed to extract {tar_path.name}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Parallel tarball unpacker")
    parser.add_argument("--clean", action="store_true", help="Remove existing folders before unpacking")
    args = parser.parse_args()
    tar_files = sorted(input_dir.glob("*.tar.gz"))
    if not tar_files:
        print("No .tar.gz files found.", flush=True)
        return

    print(f"Found {len(tar_files)} tarballs. Starting extraction with {max_workers} workers...\n", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_tar, tar): tar for tar in tar_files}

        for future in as_completed(futures):
            print(future.result(), flush=True)

if __name__ == "__main__":
    main()
