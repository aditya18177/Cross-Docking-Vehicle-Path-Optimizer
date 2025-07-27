import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# --- Data Visualization ---
print("\nGenerating the node graph...")
plt.figure(figsize=(10, 8)) # Set the figure size for better readability

# Create the scatter plot using seaborn
# x and y define the coordinates of each node
# hue colors the nodes based on their 'type' (crossdock, supplier, customer)
# size scales the nodes based on their 'demand'
# sizes specifies the range of marker sizes
# alpha sets the transparency of the markers
# palette chooses the color scheme
sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='type',
    size='demand',
    sizes=(50, 500), # Adjust min/max size of bubbles
    alpha=0.7,       # Transparency
    palette='deep'   # Color palette
)

# Add labels and title for clarity
plt.title('Graph of Nodes with Type and Demand', fontsize=16)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)

# Add a grid for easier coordinate reading
plt.grid(True, linestyle='--', alpha=0.6)

# Place the legend outside the plot area to avoid overlap
plt.legend(title='Node Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Adjust layout to prevent labels/legend from being cut off
plt.tight_layout()

# --- Saving the Plot ---
output_image_name = 'node_graph.png'
try:
    plt.savefig(output_image_name, dpi=300) # Save with high resolution
    print(f"Graph successfully saved as '{output_image_name}' in the current directory.")
except Exception as e:
    print(f"An error occurred while saving the plot: {e}")

# Display the plot (optional, will open a window if run locally)
# plt.show()
