import os
import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass

def get_centroids(parcellation_path, output_path):
    # Load the parcellation image
    img = nib.load(parcellation_path)
    data = img.get_fdata()
    
    # Get unique parcel labels, excluding 0 if it's the background
    labels = np.unique(data)
    labels = labels[labels != 0]
    
    centroids = []
    
    for label in labels:
        # Compute the centroid for each label
        mask = (data == label)
        centroid = center_of_mass(mask)
        # Convert voxel coordinates to world coordinates
        centroid_world = nib.affines.apply_affine(img.affine, centroid)
        centroids.append((label, *centroid_world))
    
    # Save centroids to a text file
    with open(output_path, 'w') as f:
        for label, x, y, z in centroids:
            f.write(f"{label}: ({x}, {y}, {z})\n")
    
    print(f"Centroids saved to {output_path}")