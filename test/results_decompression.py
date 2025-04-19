import os
from src.decompression import FlexibleJpegDecompress


def decompress_all_results():
    # Path configuration
    results_root = os.path.join(os.getcwd(), "results")
    compression_config = os.path.join(os.getcwd(), "compression_configurations",
                                      "homemade_compression_jpeg_like.yaml")

    # Validate paths
    if not os.path.exists(results_root):
        raise FileNotFoundError(f"Results directory not found at: {results_root}")
    if not os.path.exists(compression_config):
        raise FileNotFoundError(f"Config file not found at: {compression_config}")

    # Initialize decompressor
    decompressor = FlexibleJpegDecompress(compression_config)

    # Track statistics
    stats = {
        'processed': 0,
        'skipped': 0,
        'failed': 0
    }

    # Process all subdirectories
    for root, _, files in os.walk(results_root):
        rde_files = [f for f in files if f.endswith('.rde')]
        if not rde_files:
            continue

        print(f"\n📂 Processing: {os.path.relpath(root, results_root)}")

        for rde_file in rde_files:
            base_name = os.path.splitext(rde_file)[0]
            input_path = os.path.join(root, rde_file)

            # Check if output already exists (assuming it saves as .png by default)
            output_path = os.path.join(root, f"{base_name}_compressed.png")
            if os.path.exists(output_path):
                if os.path.getmtime(output_path) >= os.path.getmtime(input_path):
                    print(f"  ✔ {rde_file} (up-to-date output exists)")
                    stats['skipped'] += 1
                    continue

            try:
                print(f"  🚀 Decompressing {rde_file}...", end=' ', flush=True)

                # Just call the decompressor - it handles saving internally
                decompressor(input_path)

                print("✓")
                stats['processed'] += 1

            except Exception as e:
                print(f"\n  ❌ Failed to decompress {rde_file}: {str(e)}")
                stats['failed'] += 1
                continue

    # Print summary
    print("\n📊 Decompression Complete:")
    print(f"• Successfully processed: {stats['processed']}")
    print(f"• Skipped (up-to-date): {stats['skipped']}")
    print(f"• Failed attempts: {stats['failed']}")


if __name__ == '__main__':
    decompress_all_results()