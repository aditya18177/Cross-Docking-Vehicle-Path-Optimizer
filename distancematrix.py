import pandas as pd
import numpy as np
import os

# --- Configuration ---
# IMPORTANT: Make sure 'synthetic_nodes.csv' is in the same directory as this script,
# or provide the full path to your CSV file.
csv_file_name = 'synthetic_nodes.csv'

# --- Data Loading ---
try:
    df = pd.read_csv(csv_file_name)
    print(f"Successfully loaded data from '{csv_file_name}'")
    print("\nFirst 5 rows of the dataset:")
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
    print("\nColumn information:")
    df.info()
except FileNotFoundError:
    print(f"Error: The file '{csv_file_name}' was not found.")
    print("Please ensure the CSV file is in the same directory as the script,")
    print("or update the 'csv_file_name' variable with the correct path.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    exit()

# --- Distance Matrix Calculation ---
print("\nCalculating the distance matrix...")

# Extract node coordinates
# We are interested in 'x' and 'y' columns for distance calculation
coordinates = df[['x', 'y']].values

# Get the number of nodes
num_nodes = len(coordinates)

# Initialize an empty matrix to store distances
# The matrix will be square, with dimensions num_nodes x num_nodes
distance_matrix = np.zeros((num_nodes, num_nodes))

# Calculate Euclidean distance between all pairs of nodes
# Euclidean distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
for i in range(num_nodes):
    for j in range(num_nodes):
        # Calculate the difference in x and y coordinates
        dx = coordinates[i, 0] - coordinates[j, 0]
        dy = coordinates[i, 1] - coordinates[j, 1]
        # Apply the Euclidean distance formula
        distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)

# Create a Pandas DataFrame for better readability and indexing
# Use node_id as both row and column headers
node_ids = df['node_id'].tolist()
distance_df = pd.DataFrame(distance_matrix, index=node_ids, columns=node_ids)

# --- Displaying the Distance Matrix ---
print("\nDistance Matrix:")
# Display the DataFrame in Markdown format for clear output
print(distance_df.to_markdown(numalign="left", stralign="left"))

# Optional: Save the distance matrix to a CSV file
output_csv_name = 'distance_matrix.csv'
try:
    distance_df.to_csv(output_csv_name)
    print(f"\nDistance matrix successfully saved as '{output_csv_name}' in the current directory.")
except Exception as e:
    print(f"An error occurred while saving the distance matrix to CSV: {e}")
