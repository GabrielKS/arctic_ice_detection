"""Figure out which days our images correspond to"""

import os
from datetime import datetime
input_dir = os.path.abspath("../Sources/Raw/touse/")
subdirs = [f.path for f in os.scandir(input_dir) if f.is_dir()]
for dir in subdirs:
    dates = set()
    for f in os.scandir(dir):
        if not f.is_file(): continue
        fname = os.path.basename(f.path)
        if fname[0] == ".": continue
        date_str = fname.split("_")[1]
        date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H-%M-%S")
        dates.add(date_obj.strftime("%Y-%m-%d"))
    print(f"{os.path.basename(dir)}: {', '.join(sorted(dates))}")
