import requests
import os # Import os for path checking, a good debugging step

# Replace this with the EXACT path you verified from Windows Explorer
# Using a raw string (r"...") is best for Windows paths
file_path = r"C:\Users\eluon\OneDrive\Pictures\maulana sidik.png" 

# --- Debugging Step (Optional but Recommended) ---
if not os.path.exists(file_path):
    print(f"ERROR: File not found at path: {file_path}")
    # You could raise an error or exit here if the path is bad
else:
    print(f"File found! Proceeding with request.")
    # --- End Debugging Step ---

url = "http://localhost:5000/predict"
keluhan_text = "Jalan berlubang di depan SMP Negeri 3, mohon segera diperbaiki"

try:
    with open(file_path, "rb") as f:
        files = {"image": f}
        data = {"keluhan": keluhan_text}
        response = requests.post(url, data=data, files=files)

    print(response.status_code)
    print(response.json())
except FileNotFoundError:
    print(f"Fatal Error: The file was not found at {file_path}. Please check your path.")