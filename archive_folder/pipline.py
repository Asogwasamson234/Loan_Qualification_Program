import pandas as pd
import os
import glob
from pathlib import Path


def explore_archive_folder():
    """
    Explore what's inside the archive_folder
    """
    print("📁 EXPLORING ARCHIVE_FOLDER")
    print("=" * 50)

    archive_folder = "archive_folder"

    if not os.path.exists(archive_folder):
        print(f"❌ {archive_folder} doesn't exist!")
        return None

    print(f"Contents of '{archive_folder}':")

    # List all items in the folder
    all_items = os.listdir(archive_folder)

    if not all_items:
        print("  📂 Folder is empty!")
        return None

    for i, item in enumerate(all_items, 1):
        item_path = os.path.join(archive_folder, item)
        item_type = "DIR" if os.path.isdir(item_path) else "FILE"
        print(f"  {i:2d}. [{item_type}] {item}")

    return all_items


def find_data_files_in_folder(folder_path):
    """
    Find all data files in a folder (recursively)
    """
    print(f"\n🔍 SEARCHING FOR DATA FILES IN: {folder_path}")

    data_files = []

    # Search for all data files (including subfolders)
    patterns = [
        "**/*.xlsx",
        "**/*.xls",
        "**/*.csv",  # Excel and CSV files
        "**/*.zip",
        "**/*.7z",
        "**/*.rar",  # Archive files
    ]

    for pattern in patterns:
        found_files = glob.glob(os.path.join(folder_path, pattern), recursive=True)
        for file in found_files:
            data_files.append(file)
            print(f"  ✅ Found: {file}")

    return data_files


def load_data_files(file_paths):
    """
    Load data from various file types
    """
    print(f"\n📊 LOADING {len(file_paths)} DATA FILES")
    print("=" * 50)

    all_datasets = []

    for file_path in file_paths:
        print(f"\nProcessing: {file_path}")

        try:
            if file_path.endswith(".zip"):
                # Handle ZIP files
                df = extract_from_zip(file_path)
                if df is not None:
                    all_datasets.append(df)

            elif file_path.endswith((".xlsx", ".xls")):
                # Handle Excel files
                df = pd.read_excel(file_path)
                all_datasets.append(df)
                print(f"  ✅ Excel loaded: {df.shape}")

            elif file_path.endswith(".csv"):
                # Handle CSV files
                df = pd.read_csv(file_path)
                all_datasets.append(df)
                print(f"  ✅ CSV loaded: {df.shape}")

            else:
                print(f"  ⚠️  Skipped (unsupported format): {file_path}")

        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")

    return all_datasets


def extract_from_zip(zip_path):
    """
    Extract and load data from ZIP files
    """
    import zipfile
    import shutil

    print(f"  📦 Extracting ZIP: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # List files in ZIP
            file_list = zip_ref.namelist()
            print(f"    Files in ZIP: {file_list}")

            # Extract to temporary folder
            temp_dir = f"./temp_extract_{Path(zip_path).stem}"
            zip_ref.extractall(temp_dir)

            # Look for data files in extracted contents
            data_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith((".xlsx", ".xls", ".csv")):
                        data_files.append(os.path.join(root, file))

            # Load the first data file found
            if data_files:
                first_data_file = data_files[0]
                print(f"    Loading: {first_data_file}")

                if first_data_file.endswith(".csv"):
                    df = pd.read_csv(first_data_file)
                else:
                    df = pd.read_excel(first_data_file)

                print(f"    ✅ Loaded: {df.shape}")

                # Clean up
                shutil.rmtree(temp_dir)
                return df
            else:
                print("    ❌ No data files found in ZIP")
                shutil.rmtree(temp_dir)
                return None

    except Exception as e:
        print(f"    ❌ Error extracting ZIP: {e}")
        return None


def combine_and_save_datasets(datasets, output_filename="final_combined_dataset.csv"):
    """
    Combine all datasets and save the result
    """
    if not datasets:
        print("❌ No datasets to combine!")
        return None

    print(f"\n🔄 COMBINING {len(datasets)} DATASETS")
    print("=" * 50)

    # Combine all datasets
    combined_data = pd.concat(datasets, ignore_index=True, sort=False)

    print(f"🎉 SUCCESSFULLY CREATED MASTER DATASET!")
    print(f"Final shape: {combined_data.shape}")
    print(f"Total records: {len(combined_data):,}")
    print(f"Total columns: {len(combined_data.columns)}")

    # Save the combined dataset
    combined_data.to_csv(output_filename, index=False)
    print(f"💾 Saved as: '{output_filename}'")

    # Show preview
    print(f"\n📋 COLUMNS ({len(combined_data.columns)} total):")
    for i, col in enumerate(combined_data.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\n👀 FIRST 3 ROWS:")
    print(combined_data.head(3))

    return combined_data


# 🚀 MAIN EXECUTION
def main():
    """
    Complete data loading pipeline from archive_folder
    """
    print("🤖 LOADING DATA FROM ARCHIVE_FOLDER")
    print("=" * 60)

    # Step 1: Explore the archive_folder
    folder_contents = explore_archive_folder()

    if not folder_contents:
        print("No contents found in archive_folder!")
        return None

    # Step 2: Find all data files (including in subfolders)
    data_files = find_data_files_in_folder("archive_folder")

    if not data_files:
        print("❌ No data files found in archive_folder!")
        return None

    # Step 3: Load all data files
    datasets = load_data_files(data_files)

    if not datasets:
        print("❌ No datasets were successfully loaded!")
        return None

    # Step 4: Combine and save
    final_dataset = combine_and_save_datasets(datasets)

    print(f"\n🚀 READY FOR AI TRAINING!")
    print(f"Use 'final_combined_dataset.csv' for your loan prediction model!")

    return final_dataset


# Run the complete pipeline
if __name__ == "__main__":
    final_data = main()
    

# Check if the combined dataset exists
print("🔍 CHECKING FOR COMBINED DATASET")
print("=" * 50)

current_files = os.listdir()
print("Files in current directory:")
for file in current_files:
    if "combined" in file.lower() or "final" in file.lower():
        print(f"🎯 FOUND: {file}")
    else:
        print(f"  - {file}")

# Check specifically for our file
combined_files = [
    f for f in current_files if "combined" in f.lower() or "final" in f.lower()
]

if combined_files:
    print(f"\n✅ Combined dataset found: {combined_files[0]}")
else:
    print("\n❌ No combined dataset found. Let's create one...")
