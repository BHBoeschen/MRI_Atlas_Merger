from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd




def load_atlas(atlas_path):
    atlas = nib.load(atlas_path)
    data = atlas.get_fdata().astype(int)
    header = atlas.header
    affine = atlas.affine

    return data, header, affine



def load_roi_list(file_path):
    with open(file_path, 'r') as f:
        roi_list = [line.strip() for line in f.readlines()]
    return roi_list



def split_atlas(atlas, metadata, prefix=""):
    label_indices = {}

    for label in np.unique(atlas[atlas > 0]):
        name = metadata[metadata["index"] == label].name.iloc[0]
        indices = np.where(atlas == label)
        label_indices[f"{prefix}{name}"] = indices

    return label_indices



def merge_atlases(base_indices, overlay_indices, base_shape):
    merged_atlas = np.zeros(base_shape, dtype=np.uint16)
    merged_metadata = []

    index = 1
    for atlas_name, indices_dict in (("base", base_indices), ("overlay", overlay_indices)):
        for name, indices in indices_dict.items():
            merged_atlas[indices] = index
            merged_metadata.append({"index": index, "name": f"{atlas_name} {name}"})
            index += 1

    return merged_atlas, pd.DataFrame(merged_metadata)





def gen_parser():
    parser = ArgumentParser()
    parser.add_argument("base_map")
    parser.add_argument("base_metadata")
    parser.add_argument("overlay_map")
    parser.add_argument("overlay_metadata")
    parser.add_argument("out_map_path")
    parser.add_argument("out_metadata_path")
    parser.add_argument("roi_file", help="Path to the (manually created) Regions of Interest file, ROIs seperated by ENTER")
    return parser


def main():
    parser = gen_parser()
    args = parser.parse_args()
    
    #Loading Base Data and Look up table
    base_data, base_header, base_affine = load_atlas(Path(args.base_map))
    base_metadata = pd.read_csv(args.base_metadata, sep="\t")
    
    #getting rid of unecessary spaces in the base metadata as it will influence Region replacement
    base_metadata['name'] = base_metadata['name'].str.strip()
    
    #Getting lablels for white matter to replace inaccurate base_data regions
    left_cwm_label = base_metadata.loc[base_metadata['name'] == 'Left-Cerebral-White-Matter', 'index'].iloc[0]
    right_cwm_label = base_metadata.loc[base_metadata['name'] == 'Right-Cerebral-White-Matter', 'index'].iloc[0]
    
    #Regions of interest that will be replaced by overlay atlas. they have to be in the base_metadata 
    roi_list = load_roi_list(args.roi_file)
    
    #Extracts ROI Labels from base data and replaces them with White Matter Labels
    left_roi_indices = base_metadata.loc[base_metadata['name'].apply(
    lambda x: any(roi in x for roi in roi_list) and x.startswith("Left"))]['index'].tolist()
    right_roi_indices = base_metadata.loc[base_metadata['name'].apply(
    lambda x: any(roi in x for roi in roi_list) and x.startswith("Right"))]['index'].tolist()

        # Create a mask for the Regions of Interest in the base atlas
    left_mask = np.isin(base_data, left_roi_indices)
    right_mask = np.isin(base_data, right_roi_indices)

        # Set the Regions of Interest in the base atlas ton white matter
    base_data[left_mask] = left_cwm_label
    base_data[right_mask] = right_cwm_label
    
    
    # Find the Regions of Interest in the base_metadata for Left and Right hemispheres
    left_roi_rows = base_metadata.loc[base_metadata['name'].apply(
        lambda x: any(roi in x for roi in roi_list) and x.startswith("Left"))]
    right_roi_rows = base_metadata.loc[base_metadata['name'].apply(
        lambda x: any(roi in x for roi in roi_list) and x.startswith("Right"))]

    # Concatenate left and right hemisphere regions and convert to a list
    excluded_regions = left_roi_rows.append(right_roi_rows)['name'].tolist()

    # Print the list of excluded regions
    print("Excluded regions from base data:")
    for region in excluded_regions:
        print(region)
    
    #Loading overlay data and Look up Table
    overlay_data, _, _ = load_atlas(Path(args.overlay_map))
    overlay_metadata = pd.read_csv(args.overlay_metadata, sep="\t")
    
    base_masks = split_atlas(base_data, base_metadata, prefix="base_")
    overlay_masks = split_atlas(overlay_data, overlay_metadata, prefix="overlay_")

    merged_map, merged_metadata = merge_atlases(base_masks, overlay_masks, base_data.shape)

    merged_mapz = merged_map.astype(np.uint16)


    new_row = pd.DataFrame({'index': [0], 'name': ['unknown']})
    merged_metadata = pd.concat([new_row, merged_metadata], ignore_index=True)


    merged_metadata["name"] = merged_metadata["name"].str.replace('base', '').str.replace('overlay', '')
    merged_metadata["name"] = merged_metadata["name"].str.replace('_', '').str.replace('_', '')


    merged_img = nib.Nifti1Image(
        dataobj=merged_mapz, affine=base_affine, header=base_header
    )

    nib.save(merged_img, args.out_map_path)
    merged_metadata.to_csv(args.out_metadata_path, sep="\t", index=False)
 

if __name__ == "__main__":
    main()
