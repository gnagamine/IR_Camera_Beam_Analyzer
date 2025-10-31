import pandas as pd
import os


class GlazDataLoader:
    def __init__(self,
                 file_path,):
        self.file_path = file_path
        self.time_trace_matrix_df = pd.read_csv(file_path,
                                                             delimiter='\t',
                                                             header=None)

        delay_file_path = self._get_param_filepath(file_path)
        delay_df = pd.read_csv(delay_file_path,
                               delimiter='\t',
                               header=0)
        self.delays = delay_df.iloc[:, 0].values
        self.comments=self._get_comments()
        self.label = self._extract_series_parameter()


    def _get_param_filepath(self,
                           file_path):
        base, fname = os.path.split(file_path)
        name, ext = os.path.splitext(fname)
        parts = name.split('_')
        # Derive dataset prefix (date_time) and file index
        if len(parts) >= 4:
            dataset_prefix = '_'.join(parts[:2])
            index = parts[-1]
            param_name = f"{dataset_prefix}_param_{index}"
        else:
            # fallback for unexpected filename
            param_name = name.replace(parts[-2],
                                      'param')
        param_file_path = os.path.join(base,
                                       f"{param_name}{ext}")

        return param_file_path
    @staticmethod
    def _is_float(token: str) -> bool:
        try:
            float(token)
            return True
        except ValueError:
            return False

    def _get_commets_filepath(self):
        file_path = self.file_path
        base, fname = os.path.split(file_path)
        name, ext = os.path.splitext(fname)
        parts = name.split('_')
        # Derive dataset prefix (date_time) and file index
        if len(parts) >= 4:
            dataset_prefix = '_'.join(parts[:2])
            index = parts[-1]
            param_name = f"{dataset_prefix}_Param"
        else:
            # fallback for unexpected filename
            param_name = name.replace(parts[-2],
                                      'param')
        comments_file_path = os.path.join(base,
                                          f"{param_name}{ext}")

        return comments_file_path

    def _get_comments(self):
        """
        Parse metadata comments from the parameter file header.
        Returns a dict where keys are the header fields (e.g. 'Scancount', 'Comments')
        and values are the corresponding text (multiline for 'Comments').
        """
        header_lines = []
        path = self._get_commets_filepath()
        with open(path,
                  "r") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue

                # stop only when the *whole line* looks numeric
                tokens = stripped.split()
                if all(self._is_float(t) for t in tokens):
                    break

                header_lines.append(stripped)
        text = "\n".join(header_lines)
        return text

    def _extract_series_parameter(self,
                                 series_parameter_keyword="series keyword:"):  # Default to "series keyword:"
        """
        Extract the text following a specified keyword (e.g., 'series keyword:')
        from the 'Comments' metadata, regardless of its position in the line.

        Args:
            series_parameter_keyword (str, optional): The keyword to search for.
                Defaults to "series keyword:".

        Returns:
            str or None: The extracted string if the keyword is found, otherwise None.
        """
        if not series_parameter_keyword:  # Handle empty keyword if necessary
            return None

        keyword_to_find = series_parameter_keyword.lower()

        for line in self.comments.splitlines():
            line_lower = line.lower()
            if keyword_to_find in line_lower:
                # Find the starting position of the keyword in the original line (case-insensitive)
                start_index = line_lower.find(keyword_to_find)
                # Extract the substring after the keyword
                # Add len(keyword_to_find) to get the part *after* the keyword
                value = line[start_index + len(keyword_to_find):].strip()
                # If the keyword itself ends with a colon, an additional strip(':') might be useful
                # or ensure the keyword includes the colon if that's the delimiter.
                # For "series keyword:", the current strip() is likely sufficient.
                return value
        return None