import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import os


class VideoToNumpy:
    def __init__(self,
                 video_path,
                 camera_name):

        self.video_path = video_path
        self.camera_name = camera_name
        self.frames = None

        if self.camera_name == 'HIKMICRO':
            self.header_rows = 4
            self.matrix_rows = 192
            self.matrix_cols = 256
            self.frame_rate = 25
            self.frames = self._load_all_frames_HIKMICRO()

        if self.camera_name == 'NEC':
            self.frame_rate = 30
            self.frames = self._load_all_frames_NEC()

    def _detect_csv_config(self,
                           sample_file_path):
        """Detect delimiter and decimal once using a sample file."""
        configurations = [
            {"sep": ',', "decimal": '.'},
            {"sep": ';', "decimal": '.'},
            {"sep": ';', "decimal": ','}
        ]
        for config in configurations:
            try:
                pd.read_csv(
                    sample_file_path,
                    header=None,
                    sep=config["sep"],
                    decimal=config["decimal"],
                    engine='python',
                    nrows=5  # Only need a few rows
                )
                return config
            except:
                continue
        raise ValueError(f"Could not detect CSV format for {sample_file_path}")

    def _load_single_frame_nec(self,
                               file_path,
                               csv_config):
        """Load one NEC CSV file efficiently."""
        try:
            df = pd.read_csv(
                file_path,
                header=None,
                sep=csv_config["sep"],
                decimal=csv_config["decimal"],
                engine='python',
                on_bad_lines='warn'
            )
            arr = np.nan_to_num(df.to_numpy(),
                                nan=0.0)
            if arr.ndim == 2 and arr.shape[1] > 2:
                return arr[:, 1:-1]
            else:
                print(f"Warning: {os.path.basename(file_path)} has {arr.shape[1]} cols, keeping as-is.")
                return arr
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None

    def _load_all_frames_NEC(self):
        """Optimized: detect format once, load in parallel."""
        # Get and sort CSV files
        files_name_list = [
            f for f in os.listdir(self.video_path)
            if f.endswith('.csv') and not f.startswith('.')
        ]
        if not files_name_list:
            raise ValueError("No CSV files found.")

        files_name_list.sort()
        file_paths = [os.path.join(self.video_path,
                                   f) for f in files_name_list]

        # Detect CSV format using first file
        print(f"Detecting CSV format from {files_name_list[0]}...")
        csv_config = self._detect_csv_config(file_paths[0])

        print(f"Using config: sep='{csv_config['sep']}', decimal='{csv_config['decimal']}'")
        print(f"Loading {len(file_paths)} frames in parallel...")

        # Parallel loading
        frames = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Correct: single executor.map call
            results = executor.map(
                lambda fp: self._load_single_frame_nec(fp,
                                                       csv_config),
                file_paths
            )
            # Iterate over results as they complete
            for i, arr in enumerate(results):
                if arr is not None:
                    frames.append(arr)
                else:
                    print(f"Skipping invalid frame: {files_name_list[i]}")

        if not frames:
            raise ValueError("No valid frames loaded.")

        print(f"Successfully loaded {len(frames)} frames.")
        return np.stack(frames,
                        axis=0)

    # def _load_all_frames_NEC(self):
    #     """Compute the standard deviation map from the temperature maps in the CSV file."""
    #     files_name_list = os.listdir(self.video_path)
    #     maps_array_list=[]
    #
    #     for filename in files_name_list:
    #         if filename.endswith('.csv') and not filename.startswith('.'):
    #             print(f'loading: {filename}')
    #             file_path = os.path.join(self.video_path, filename)
    #             Map = IntensityMap(file_path=file_path,
    #                 camera_name='NEC')
    #             map_array=Map.data
    #             maps_array_list.append(map_array)
    #
    #     # Stack all frames into a 3D array: (num_frames, height, width)
    #     print(f'Number of frames analyzed: {len(maps_array_list)}')
    #     stacked_maps = np.stack(maps_array_list,
    #                             axis=0)
    #
    #     return stacked_maps

    def _load_all_frames_HIKMICRO(self):
        """
        Parse the CSV file and return a 3-D array of shape
        (nframes, matrix_rows, matrix_cols).  The file is read only once.
        """
        if self.frames is not None:
            return self.frames

        print("Loading full video file into memory ...")
        frames_list = []

        with open(self.video_path,
                  "r") as fh:
            # skip global header
            for _ in range(self.header_rows):
                fh.readline()

            while True:
                line = fh.readline()
                if not line:  # EOF
                    break

                if line.strip().startswith("absolute time:"):
                    # next line is the per-frame time stamp (ignored for now)
                    fh.readline()

                    # read the matrix block (192 rows × 256 cols)
                    matrix = np.full((self.matrix_rows, self.matrix_cols),
                                     np.nan)

                    for row_idx in range(self.matrix_rows):
                        row_line = fh.readline()
                        if not row_line:
                            raise EOFError("Unexpected EOF while reading matrix.")
                        parts = [p.strip() for p in row_line.split(",")]
                        # drop trailing empty strings
                        while parts and not parts[-1]:
                            parts.pop()
                        if len(parts) != self.matrix_cols:
                            raise ValueError(
                                f"Row {row_idx} has {len(parts)} values, expected {self.matrix_cols}"
                            )
                        matrix[row_idx] = [
                            float(p) if p else np.nan for p in parts
                        ]

                    frames_list.append(matrix)

                    # skip the empty separator line (if present)
                    sep = fh.readline()
                    if sep.strip():
                        warnings.warn(
                            f"Expected empty separator, got: {sep.strip()!r}"
                        )

        if not frames_list:
            raise ValueError("No frames found in the CSV file.")

        self.frames = np.stack(frames_list,
                               axis=0)  # (T, H, W)
        print(f"Loaded {self.frames.shape[0]} frames ({self.matrix_rows}×{self.matrix_cols}).")
        return self.frames


if __name__ == "__main__":
    video_path = (
        '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz '
        'Camera/20250616/NEC/video')
    camera_name = 'NEC'
    frames = VideoToNumpy(video_path=video_path,
                          camera_name=camera_name).frames
