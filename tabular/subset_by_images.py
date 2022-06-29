"""Get subsets of daily tabular data corresponding to the times images were taken"""

import os
import xarray as xr
from datetime import datetime
import numpy as np

time_threshold = np.timedelta64(10, "s")  # Amount before and after a given timestamp to include
tabular_root = os.path.abspath("../tabular/touse")
image_root = os.path.abspath("../Sources/Raw/touse")

def tabular_to_boat(fname):
    return os.path.basename(fname).split('_')[3].split('-')[1][2:]

def tabular_to_datetime(fname):
    return datetime.strptime(os.path.basename(fname).split('_')[3].split('-')[2], "%Y%m%dT%H%M%S") 

def image_to_datetime(fname):
    return datetime.strptime(os.path.basename(fname).split("_")[1], "%Y-%m-%dT%H-%M-%S")

def main():
    print(f"Getting all tabular data within {time_threshold} of any image")
    files_of_interest = [f.path for f in os.scandir(tabular_root) if f.is_file()]
    image_boats = set(os.path.basename(f.path) for f in os.scandir(image_root) if f.is_dir())
    tabular_boats = set(map(tabular_to_boat, files_of_interest))
    full_boats = sorted(image_boats & tabular_boats)
    print(f"Finding tabular subsets for boats {', '.join(full_boats)}.")

    for boat in full_boats:
        images = [f.path for f in os.scandir(os.path.join(image_root, boat))
            if f.is_file() and os.path.basename(f.path)[0] != '.']
        img_times = np.array(list(map(image_to_datetime, images)), dtype=np.datetime64)
        # Guarantee that when we concat everything together later, the time axis is strictly increasing:
        img_times.sort()

        files_for_boat = list(filter(lambda f: tabular_to_boat(f) == boat, files_of_interest))
        print(f"For {boat}, found {len(files_for_boat)} tabular files.")
        boat_ds = xr.open_mfdataset(files_for_boat, combine="nested", concat_dim="obs")
        swapped = boat_ds.isel(trajectory=0).swap_dims(obs="time")

        sliceds = []
        for i,t0 in enumerate(img_times):
            sliced = swapped.sel(time=slice(t0-time_threshold, t0+time_threshold))
            if len(sliced["time"]) == 0:
                print(f" --- DATA NOT FOUND FOR {t0} --- ")
                continue
            if i % 100 == 0: print(f"{i}/{len(img_times)}")
            # sliced.to_netcdf(path=f"../temp/sliced_{boat}_{i:04d}.nc")
            sliceds.append(sliced)
        print("Concatting...")
        sliced_combined = xr.concat(sliceds, dim="time")  # Is quite fast but leads to duplicate values from overlapping ranges
        # sliced_combined = xr.merge(sliceds)  # Would get rid of duplicate values but kills our memory performance
        print("Deduplicating...")
        sliced_combined = sliced_combined.drop_duplicates("time")  # Seems to work fine?
        print("Saving...")
        # Due to lazy loading, most of the time is spent here:
        sliced_combined.to_netcdf(path=f"../tabular/excerpts/excerpt_{boat}.nc")

if __name__ == "__main__":
    main()
