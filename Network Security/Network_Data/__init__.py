import pandas as pd
import os

# Path to your Excel file
excel_file = r"C:\Users\AMAN\Downloads\data.xlsx"   # change to your file name

# Read Excel file
df = pd.read_excel(excel_file)

# Define project folder path (example: your GitHub project folder)
project_folder = r"C:\Users\AMAN\Downloads\DATA Scientist\GitHub\ML-OR-DL-Endt-to-End-Projects\Network Security"

# Save as CSV inside the project folder
csv_file = os.path.join(project_folder, "data.csv")
df.to_csv(csv_file, index=False)

print(f"File successfully saved at: {csv_file}")
