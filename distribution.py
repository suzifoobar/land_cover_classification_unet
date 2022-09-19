import os
import glob

import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


img_dir = "data/training_data"  # Enter Directory of all images
data_path = os.path.join(img_dir, 'masks')
# local path when data is manually downloaded as a zip
# data_path = os.path.join(img_dir, '*.png')

files = glob.glob(data_path)
data = []
yellow_count = 0
magenta_count = 0
cyan_count = 0
black_count = 0
blue_count = 0
white_count = 0
green_count = 0


for f1 in files:
    # print(f1)
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # urban_land
    cyan_count += np.count_nonzero((img == [0, 255, 255]).all(axis=2))
    # agriculture
    yellow_count += np.count_nonzero((img == [255, 255, 0]).all(axis=2))
    # range_land
    magenta_count += np.count_nonzero((img == [255, 0, 255]).all(axis=2))
    # forest_land
    green_count += np.count_nonzero((img == [0, 255, 0]).all(axis=2))
    # water
    blue_count += np.count_nonzero((img == [0, 0, 255]).all(axis=2))
    # barren_land
    white_count += np.count_nonzero((img == [255, 255, 255]).all(axis=2))
    # unknown
    black_count += np.count_nonzero((img == [0, 0, 0]).all(axis=2))


print("yellow:", yellow_count/1e6)
print("magenta:", magenta_count/1e6)
print("cyan:", cyan_count/1e6)
print("black:", black_count/1e6)
print("white:", white_count/1e6)
print("blue:", blue_count/1e6)
print("green:", green_count/1e6)
n_bins = 7

df = pd.DataFrame({'Land Type': ['urban',
                                 'agri',
                                 'range',
                                 'forest',
                                 'water',
                                 'barren',
                                 'unknown'
                                 ],
                   'Pixels': [cyan_count,
                              yellow_count,
                              magenta_count,
                              green_count,
                              blue_count,
                              white_count,
                              black_count]})

ax = df.plot.bar(x='Land Type', y='Pixels',  width=1)
plt.show()
df.to_csv(os.path.join(Path(img_dir).parent, 'image_stats.csv'))
