# astrosat-processing

# Postprocessing steps.

1. Folder structure: zip files with the processed images should be placed in individual folders for each galaxy. For a few targets in the PHANGS Astrosat proposal, this required a by-hand copying for ~5 targets.
2. Set input and output directories in `extract_convert_reproject_sample.py`. These are hardcoded in the script but are easy to find (they're all `pathlib.Path` objects).
3. Run `extract_convert_reproject_sample.py` to unzip files, assign filter properties and convert to flux units, reproject to square pixels and swap E-W directions in the header, and export final products per filter for each galaxy.

Reprojection uses [adaptive resampling](https://reproject.readthedocs.io/en/stable/celestial.html#adaptive-resampling) from the reproject package by default.

`extract_convert_reproject_sample.py` will also produce the `astrosat_filter_coverage.csv` file with the filters available for each galaxy.

Final products are split into a Local Group, PHANGS, and "other" samples.
