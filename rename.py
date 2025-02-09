import os

directory = "pages"

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath) and "." not in filename:  # Ensure it's a file without an extension
        new_filepath = filepath + ".txt"
        os.rename(filepath, new_filepath)
        print(f"renamed {new_filepath}")