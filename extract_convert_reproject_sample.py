
'''
The folder structure is anything but homogeneous.

1. Extract and rename here for further processing
2. Apply unit conversion
3. Reproject to final headers
4. Export to final data directory
'''

import os
import zipfile
from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from datetime import datetime
from reproject import reproject_interp
from spectral_cube import Projection
from scipy import ndimage as nd

repo_path = Path(os.path.expanduser("~/ownCloud/observing_code/PHANGS/astrosat-processing"))

filt_tab = Table.read(repo_path / "astrosat_filters.csv")
phangs_tab = Table.read(repo_path.parent / "phangs_sample.csv")

def cts_to_flux_factor(hdu, filename):
    '''
    Convert from counts to flux.
    '''

    # Filter conversion
    band_name = hdu.header['DETECTOR']
    filt_name = hdu.header['FILTERID']

    # There's some ambiguity with certain filter names. We're going to
    # assume the more standard filter when there's ambiguity (F148W vs F148Wa)
    # mult_names = ['CaF2', 'Silica']
    # mult_bands = ['FUV', 'NUV']

    if filt_name == 'CaF2' and band_name == 'FUV':
        filt_name += '-1'
    if filt_name == 'Silica' and band_name == 'NUV':
        filt_name += '-1'

    # Here's special cases where the naming convention doesn't match
    # the calibration paper (or the other docs, from what I see?)
    if filt_name == 'Silica15' and band_name == 'NUV':
        filt_name = 'Silica-1'

    match = np.logical_and(filt_tab['band'] == band_name,
                           filt_tab['filter'] == filt_name)

    if not match.any():
        raise ValueError(f"Unable to match filter for {filename}")

    if match.sum() > 1:
        raise ValueError(f"Found multiple filter matches for {filename}")

    unit_conv = float(filt_tab['unit_conv'][match])

    flux_unit = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1

    # Record info for metadata.
    metadata = {'band': band_name,
                'filtername': filt_tab['filter_name'][match][0],
                'filter': filt_name,
                'wave_mean': filt_tab['lam_mean'][match][0],
                'wave_delta': filt_tab['lam_delta'][match][0],
                'CTSTOFLUX': unit_conv,
                'IntTime': hdu.header['DATATIME']}

    # Loop through the metadata to append as a comment to the header
    new_header = hdu.header.copy()

    del new_header['COMMENT']

    # fits verify complaining about the datetime string... force str conversion
    timenow = str(datetime.today().strftime('%Y/%m/%d %H:%M:%S'))
    new_header['COMMENT'] = f"Converted from cts to flux on {timenow} using the following"
    for key in metadata:
        new_header['COMMENT'] = f"{key}: {metadata[key]}"
    new_header['COMMENT'] = f"See https://github.com/PhangsTeam/astrosat-processing"
    new_header['COMMENT'] = "Conversion factors from 2017AJ....154..128T"

    new_header['BUNIT'] = flux_unit.to_string()

    new_data = (hdu.data * unit_conv/ hdu.header['DATATIME']) * flux_unit

    hdu_in_flux = fits.PrimaryHDU(new_data.value, new_header)

    return hdu_in_flux, metadata


def reproject_data(hdu, trim_shape=True, verbose=False):

    mywcs = WCS(hdu.header)

    # Get the pixel scales. We want to reproject into square pixels.
    sq_pix_scale = proj_plane_pixel_scales(mywcs).max()

    # Round to within 100 nanoarcsec (pix are ~0.4", this is plenty of precision)
    sq_pix_scale = np.round(sq_pix_scale, 6)

    new_header = mywcs.to_header()
    # Delete the PC matrix. Going to square pixels.
    for key in ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']:
        del new_header[key]

    new_header['CDELT1'] = -sq_pix_scale
    new_header['CDELT2'] = sq_pix_scale

    # Flipping RA, so adjust the CRPIX
    new_header['CRPIX1'] = hdu.data.shape[0] - hdu.header['CRPIX1']

    # Shape params
    new_header['NAXIS'] = hdu.header['NAXIS']
    new_header['NAXIS1'] = hdu.header['NAXIS1']
    new_header['NAXIS2'] = hdu.header['NAXIS2']

    # Loop through and add in info to the new header.
    # Skip a few patterns
    skip_patterns = ['WCP', 'WCV', 'CPIX1RM', 'CPIX2RM', 'CPIX1RS', 'CPIX2RS',
                     'CVAL1RM', 'CVAL2RM', 'CVAL1RS', 'CVAL2RS',
                     'CPIXRM', 'CPIXRS', 'CVALRM', 'CVALRS',
                     'CD1', 'CD2', 'CROTA', 'CCVALD', 'CCVALS',
                     'WIN_', 'RA_', 'DEC_', 'SCANRATE', 'ROTATN']

    has_comment = False
    for key in hdu.header:
        if key == "HISTORY":
            continue

        if key == "COMMENT":
            if not has_comment:
                for line in hdu.header['COMMENT']:
                    new_header['COMMENT'] = line
                has_comment = True

        if key not in new_header:
            match = False
            for skips in skip_patterns:
                if skips in key:
                    if verbose:
                        print(f"Skipping {key}")
                    match = True
                    break

            if match:
                continue

            new_header[key] = hdu.header[key]

    if 'MJD-OBS' in new_header:
        del new_header['MJD-OBS']

    rep_data, footprint = reproject_interp(hdu, new_header, order='bilinear')

    rep_hdu = fits.PrimaryHDU(rep_data, new_header)

    # Convert to a projection to easily slice the data.
    if trim_shape:
        proj = Projection.from_hdu(rep_hdu)
        proj = proj[nd.find_objects(np.isfinite(proj))[0]]

        rep_hdu = proj.hdu

        del proj

    return rep_hdu


if __name__ == "__main__":

    data_path = Path(os.path.expanduser("~/space/ekoch/AstroSat/processed"))
    product_path = Path(os.path.expanduser("~/space/ekoch/AstroSat/products/"))

    if not data_path.exists():
        data_path.mkdir()

    if not product_path.exists():
        product_path.mkdir()

    # Define output paths for PHANGS vs. Local Group
    phangs_product_path = product_path / "phangs"
    if not phangs_product_path.exists():
        phangs_product_path.mkdir()

    lg_product_path = product_path / "localgroup"
    if not lg_product_path.exists():
        lg_product_path.mkdir()

    # Record which filters each target has for an output table.
    data_filter_dict = {}
    for filtname in filt_tab['filter_name']:
        data_filter_dict[filtname] = []

    target_list = []

    # Grab all directories
    for zip_filename in data_path.glob("**/*.zip"):

        print(f"Extracting {zip_filename}")

        # Extract to the same folder
        out_path = zip_filename.parent

        # Folders have the right names. Processed FITS files may not
        # if the original target does not match.
        target_name = out_path.name

        # Is this a PHANGS target?
        is_phangs = target_name.replace('_', '') in phangs_tab['Name']

        if is_phangs:
            print("This is a PHANGS target.")
        else:
            print("This is a LG target.")

        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(out_path)

        # Loop through the resulting FITS files and move to a common naming
        # format.

        target_filter_dict = {}
        for filt in filt_tab['filter_name']:
            target_filter_dict[filt] = False

        for fits_file in out_path.glob("*.fits"):

            # Get out properties from the name

            orig_target, band, filt = fits_file.name.split("_")[:3]

            with fits.open(fits_file) as hdulist:

                if len(hdulist) > 1:
                    raise ValueError(f"{fits_file} has >1 extension. Check this.")

                hdu = hdulist[0]

                # Flux conversion and identify filter
                hdu_in_flux, metadata = cts_to_flux_factor(hdu, fits_file)

                target_filter_dict[metadata['filtername']] = True

                # Reprojection
                print("Reprojecting")
                rep_hdu = reproject_data(hdu_in_flux, trim_shape=True, verbose=False)

            # Save to the right output
            out_name = f"{target_name.replace('_', '')}_{band}_{metadata['filtername']}_flux_reproj.fits"

            if is_phangs:
                this_product_path = phangs_product_path
            else:
                this_product_path = lg_product_path

            out_rep_path = this_product_path / target_name.replace('_', '')
            if not out_rep_path.exists():
                out_rep_path.mkdir()

            rep_hdu.writeto(out_rep_path / out_name, overwrite=True)

            del hdu_in_flux, rep_hdu

        # Append filter matches to the master dict
        for key in target_filter_dict:
            data_filter_dict[key].append(target_filter_dict[key])

        target_list.append(target_name)


    tab_has_data = Table(data_filter_dict, names=target_list)
    tab_has_data.write(repo_path / "astrosat_filter_coverage.csv")
