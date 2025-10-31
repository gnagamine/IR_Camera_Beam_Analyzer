import os

def create_folder_in_cwd_if_not_exist (folder_name: str) -> None:
    """
    Create a folder in the current working directory if it does not already exist.

    Parameters
    ----------
    folder_name : str
        Name of the folder to create (relative to the current working directory).
    """
    current_directory = os.getcwd()
    target_path = os.path.join(current_directory, folder_name)

    # Create folder only if it doesn't exist
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"Created folder: {target_path}")
    else:
        print(f"Folder already exists: {target_path}")