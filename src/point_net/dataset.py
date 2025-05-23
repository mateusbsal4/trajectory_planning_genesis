import os
import torch
import numpy as np
import pandas as pd
import yaml

from torch_geometric.data import Dataset, Data
from sklearn.model_selection import train_test_split


def load_success_labels(root: str,
                        labels_csv: str = "labels.csv",
                        success_yaml: str = "opt_successfull.yaml") -> pd.DataFrame:
    """
    Loads labels.csv and filters to only those scenes with success=true (according to the YAML).
    Returns a DataFrame indexed by bare 'scene' names (no extension).
    """
    # 1) Load success flags from YAML
    yaml_path = success_yaml if os.path.isabs(success_yaml) else os.path.join(root, success_yaml)
    with open(yaml_path, "r") as f:
        success_map = yaml.safe_load(f)

    # 2) Load labels CSV
    csv_path = labels_csv if os.path.isabs(labels_csv) else os.path.join(root, labels_csv)
    df = pd.read_csv(csv_path)
    #df["scene"] = df["scene"].str.replace(r"\.txt$|\.ply$", "", regex=True)

    # 3) Keep only rows where success_map[scene] is True
    successful_scenes = [scene for scene, ok in success_map.items() if ok]
    df_success = df[df["scene"].isin(successful_scenes)].copy()

    # 4) Index by scene name
    df_success = df_success.set_index("scene")
    return df_success


class GainRegressionDataset(Dataset):
    """
    A PyG Dataset for scene-level regression on point clouds saved as `.txt`.
    Each `…/data/inputs/{scene}.txt` contains N lines of "x y z".
    Only scenes with success=true are included. We sample exactly `npoints` per scene,
    center & normalize, and (optionally) augment.

    Returns a torch_geometric.data.Data with:
        - pos:   [npoints, 3] float32 tensor
        - y:     [36] float32 tensor (detect_shell_rad + five 7-D vectors)
        - x:     None  (no extra features beyond pos)
        - batch: automatically assigned by PyG DataLoader when batching
    """
    def __init__(
        self,
        root: str,
        labels_csv: str = "labels.csv",
        success_yaml: str = "opt_successfull.yaml",
        split: str = "train",
        test_size: float = 0.2,
        random_state: int = 42,
        npoints: int = 2500,
        augment: bool = True
    ):
        # root is expected to be ".../dataset_generator/data"
        super().__init__(root)
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.npoints = npoints
        self.augment = augment if split == "train" else False
        self.root_dir = root

        # 1) Load & filter CSV to only successful scenes
        df = load_success_labels(root, labels_csv, success_yaml)

        # 2) Keep only the 36 output columns: "detect_shell_rad" + all k_* keys
        output_cols = ["detect_shell_rad"] + [
            col for col in df.columns
            if any(col.startswith(p) for p in ["k_a_ee", "k_c_ee", "k_r_ee", "k_d_ee", "k_manip"])
        ]
        self.labels_df = df[output_cols]

        # 3) Identify all ".txt" files under root/inputs
        txt_dir = os.path.join(root, "inputs")
        all_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
        all_scenes = [os.path.splitext(f)[0] for f in all_files]

        # 4) Keep only those scenes present in labels_df
        valid_scenes = sorted(set(all_scenes) & set(self.labels_df.index))

        # 5) Split into train vs val
        train_scenes, val_scenes = train_test_split(
            valid_scenes,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        #train_scenes.sort(key=lambda s: int(s.split('_')[1]))
        #val_scenes.sort(key=lambda s: int(s.split('_')[1]))
        #print("Train scenes: ", train_scenes)
        self.scenes = train_scenes if split == "train" else val_scenes
        self.txt_dir = txt_dir

    def len(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Data:
        """
        Returns a single torch_geometric.data.Data object.

        Steps:
          1.   Read `…/inputs/{scene}.txt` via np.loadtxt → pts (N×3)
          2.   Sample exactly `npoints` rows
          3.   Center & scale to unit sphere
          4.   (Optional) random rotation about Y + jitter
          5.   Build Data(pos=[npoints,3], y=[36], x=None)
        """
        scene = self.scenes[idx]
        txt_path = os.path.join(self.txt_dir, scene + ".txt")
        # 1) Load point cloud as (N,3) float32
        pts = np.loadtxt(txt_path, dtype=np.float32)  # shape: [N, 3]

        # 2) Sample a fixed subset of size self.npoints
        if pts.shape[0] >= self.npoints:
            choice = np.random.choice(pts.shape[0], self.npoints, replace=False)
        else:
            choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        point_set = pts[choice, :]  # shape: [npoints, 3]

        # 3) Center & normalize
        centroid = point_set.mean(axis=0)
        point_set -= centroid
        max_dist = np.max(np.linalg.norm(point_set, axis=1))
        point_set /= max_dist

        # 4) (Optional) augment: random Y‐axis rotation + jitter
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rot)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        # Convert to torch.FloatTensor
        pos = torch.from_numpy(point_set)  # shape: [npoints, 3], dtype=torch.float32

        # 5) Read the 36‐dim scene label from self.labels_df
        label_vec = self.labels_df.loc[scene].values.astype(np.float32)  # [36]
       
        y = torch.from_numpy(label_vec).unsqueeze(0)                     # [1, 36]
        #print("Input shape", pos.shape)
        #print("Label shape:", y.shape)        
        
        # Build a Data object with pos and y
        data = Data(pos=pos, y=y)
        return data
