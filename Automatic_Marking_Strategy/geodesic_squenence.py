import os
from pathlib import Path
import time
import logging
import numpy as np
import nibabel as nib
import skfmm
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Constants
BETA = 1.0
SIGMA = 1.0
ARCTAN_SHIFT = 0.3
ARCTAN_SCALE = 0.1
a_supper = 0.5
a_lower = 0.75
DX = 1.0  # Grid spacing
MAX_WORKERS = os.cpu_count() - 1 or 1


def compute_speed_map(volume: np.ndarray, beta: float = BETA, sigma: float = SIGMA) -> np.ndarray:
    """
    Precompute the speed map (f) for geodesic computation once per volume.
    """
    # normalize and smooth
    normed = (volume - volume.mean()) / volume.std()
    smooth = gaussian_filter(normed, sigma=sigma)
    # gradient magnitude
    grads = np.gradient(smooth)
    mag = np.sqrt(sum(g**2 for g in grads))
    f = beta * (np.max(smooth) - smooth) * mag**2
    return 1.0 / (f + 1e-3)


def geodesic_distance(seed_mask: np.ndarray, speed: np.ndarray, dx: float = DX) -> np.ndarray:
    """
    Compute normalized geodesic travel time from seed mask.
    """
    raw = skfmm.travel_time(seed_mask - 0.5, speed, dx=dx, order=1)
    return raw / np.max(raw)


def process_volume(volume_path: Path, out_hp: Path, out_hs: Path) -> tuple:
    """
    Process a single NIfTI volume: compute Hp, find isolated regions, compute Hs, save maps and return counts.
    """
    # Load data
    img = nib.load(str(volume_path))
    data = img.get_fdata()
    data_norm = data / data.max()
    affine = img.affine

    # Precompute speed map once
    speed = compute_speed_map(data_norm)

    # Initial geodesic from fixed seed
    seed0 = np.zeros_like(data_norm, dtype=float)
    seed0[10, 10, 10] = 1.0
    T0 = geodesic_distance(seed0, speed)
    Hp = 0.5 + (1.0/np.pi) * np.arctan(-(T0 - ARCTAN_SHIFT)/ARCTAN_SCALE)
    # save Hp
    nib.save(nib.Nifti1Image(Hp, affine), str(out_hp))
    # binarize
    T_p = (1 - Hp) >= a_supper
    m_p = T_p.mean()

    # label connected components in one pass
    labeled = label(T_p)
    props = regionprops(labeled)

    num= 0
    # for each component, compute Hs and save if condition
    for idx, prop in enumerate(props, start=1):
        centroid = tuple(map(int, prop.centroid))
        seed = np.zeros_like(data_norm, dtype=float)
        seed[centroid] = 1.0
        Tn = geodesic_distance(seed, speed)
        Hs = 0.5 + (1.0/np.pi) * np.arctan(-(Tn - ARCTAN_SHIFT)/ARCTAN_SCALE)
        T_s = Hs >= a_lower
        if T_s.mean() < m_p:
            num += 1
            print(f"Region {idx} centroid: {centroid}")
            # normalize and save
            Hs_norm = Hs / Hs.max()

            out_file = out_hs.with_name(f"{volume_path.stem}_{num+1:03d}.nii.gz")
            nib.save(nib.Nifti1Image(Hs_norm, affine), str(out_file))
    return num


def main(base_dir: str, label_dir: str, save_hp: str, save_hs: str, mat_out: str):
    start = time.time()
    base_path = Path(base_dir)
    label_path = Path(label_dir)
    out_hp = Path(save_hp); out_hp.mkdir(parents=True, exist_ok=True)
    out_hs = Path(save_hs); out_hs.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in (label_path).iterdir() if f.suffix == '.gz'])
    nums = []
    accs = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = []
        for lab in files:
            img = base_path / (lab.name.replace('.nii.gz', '_0000.nii.gz'))
            if not img.exists():
                logging.warning("Missing volume for %s", lab.name)
                continue
            hp_file = out_hp / img.name.replace('_0000.nii.gz', '_0001.nii.gz')
            futures.append(exe.submit(process_volume, img, lab, hp_file, out_hs/img.name.replace('_0000.nii.gz', '')))

        for future in as_completed(futures):
            n = future.result()
            nums.append(n)


    # save .mat
    from scipy.io import savemat
    savemat(mat_out, {'Num_center': np.array(nums)[:, None]})
    logging.info("Done in %.2f seconds", time.time() - start)


if __name__ == '__main__':
    main(r'/data/Task001_Hecktor/imagesTs',
         r'/data/Task001_Hecktor/PriorTs', r'/data/Task001_Hecktor/SequencesTs',
         r'/data/Task001_Hecktor/supper_0.5_lower_0.75.mat')

