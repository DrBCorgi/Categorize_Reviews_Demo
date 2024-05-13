import tkinter as tk
from tkinter import filedialog
from transformers import pipeline
import pandas as pd

# Create a tkinter window
window = tk.Tk()
window.withdraw()

# Prompt for the input CSV file
input_file_path = filedialog.askopenfilename(title="Select the input CSV file")

# Prompt for the categories (comma-separated)
categories = tk.simpledialog.askstring("Categories", "Enter the categories (comma-separated):")
categories = [cat.strip() for cat in categories.split(",")]

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(input_file_path)

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Perform the classification
results = df["Review"].apply(lambda x: classifier(x, categories, multi_label=True))

# Extract the categories and scores from the results
df["Categories"] = results.apply(lambda x: ", ".join(x["labels"]))
for i, category in enumerate(categories):
    df[f"Score_{category}"] = results.apply(lambda x: x["scores"][i])

# Prompt for the output file path
output_file_path = filedialog.asksaveasfilename(title="Save the output CSV file", defaultextension=".csv")

# Save the DataFrame to a new CSV file
df.to_csv(output_file_path, index=False)

print("Classification completed. Results saved to:", output_file_path)