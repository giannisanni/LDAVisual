import pandas as pd

file_path = 'combined_sorted_data.xlsx'
data = pd.read_excel(file_path)

unique_dg_types = data['DG'].unique()

for i, dg_type in enumerate(unique_dg_types):
    if i >= 6:  # Limit to 5 types
        break
    filtered_data = data[data['DG'] == dg_type]
    output_file_path = f'filtered_data_{dg_type}.xlsx'
    filtered_data.to_excel(output_file_path, index=False)
