import os
import pandas as pd
import rasterio
import numpy as np

# Base Folder (just the patches folder)
base_folder = r"../dataset/patches"

all_summaries = []

# Loop through patch folders directly inside "patches"
for fname in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, fname)

    if not os.path.isdir(folder_path):
        continue

    tmrt_path = os.path.join(folder_path, "Tmrt_average.tif")
    buffer_path = os.path.join(folder_path, "buffer_mask.tif")
    summary_csv_path = os.path.join(folder_path, "patch_summary.csv")

    if not (os.path.exists(tmrt_path) and os.path.exists(buffer_path) and os.path.exists(summary_csv_path)):
        print(f"Missing files in {folder_path}, skipping.")
        continue

    try:
        with rasterio.open(tmrt_path) as tmrt_src, rasterio.open(buffer_path) as buffer_src:
            tmrt = tmrt_src.read(1).astype(np.float32)
            buffer = buffer_src.read(1)

            if tmrt.shape != buffer.shape:
                print(f"Shape mismatch in {folder_path}, skipping.")
                continue

            mask = (buffer == 1)
            if not np.any(mask):
                print(f"No buffer mask in {folder_path}, skipping.")
                continue

            tmrt_masked = np.where(mask, tmrt, np.nan)

            # Mask nodata values from original raster
            nodata_val = tmrt_src.nodata
            if nodata_val is not None:
                tmrt_masked = np.where(tmrt == nodata_val, np.nan, tmrt_masked)

            valid_vals = tmrt_masked[~np.isnan(tmrt_masked)]
            if valid_vals.size == 0:
                print(f"No valid Tmrt pixels in {folder_path}, skipping.")
                continue

            # Save masked raster as new GeoTIFF
            masked_tif_path = os.path.join(folder_path, "Tmrt_buffer_masked.tif")
            output_nodata = -9999
            tmrt_masked_filled = np.where(np.isnan(tmrt_masked), output_nodata, tmrt_masked)

            profile = tmrt_src.profile
            profile.update(dtype=rasterio.float32, nodata=output_nodata)

            with rasterio.open(masked_tif_path, 'w', **profile) as dst:
                dst.write(tmrt_masked_filled.astype(np.float32), 1)

            print(f"Masked Tmrt saved: {masked_tif_path}")

            stats = {
                'Tmrt_Buffer_Mean': round(float(np.nanmean(valid_vals)), 2),
                'Tmrt_Buffer_Min': round(float(np.nanmin(valid_vals)), 2),
                'Tmrt_Buffer_Max': round(float(np.nanmax(valid_vals)), 2),
                'Tmrt_Buffer_Std': round(float(np.nanstd(valid_vals)), 2),
                'Tmrt_Buffer_Median': round(float(np.nanmedian(valid_vals)), 2),
                'Tmrt_Buffer_Count': int(valid_vals.size)
            }

            # Update CSV in folder
            df_summary = pd.read_csv(summary_csv_path)

            for key, value in stats.items():
                df_summary[key] = value

            df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            print(f"Updated: {summary_csv_path}")

            # Append to global list
            df_summary["Patch_Folder"] = folder_path
            all_summaries.append(df_summary)

    except Exception as e:
        print(f"Error processing {folder_path}: {e}")

# Save global CSV
if all_summaries:
    df_all = pd.concat(all_summaries, ignore_index=True)
    global_csv_path = os.path.join(base_folder, "all_patch_summaries.csv")
    df_all.to_csv(global_csv_path, index=False, encoding="utf-8-sig")
    print(f"\nGlobal summary saved: {global_csv_path}")
else:
    print("\nNo summaries collected. Global CSV not created.")