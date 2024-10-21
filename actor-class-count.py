import pandas as pd

# Load the Excel file
file_path = 'inference/pofma_predictions.xlsx'
df = pd.read_excel(file_path)

# Define the columns representing the actors
actors = ['Actor: Media', 'Actor: Political Group or Figure', 'Actor: Civil Society Group or Figure',
          'Actor: Social Media Platform', 'Actor: Internet Access Provider', 'Actor: Private Individual']

# Calculate the percentage of notices for each actor and round to 2 decimal places
percentages = (df[actors].mean() * 100).round(2)

# Create a DataFrame to display the results
percentages_df = pd.DataFrame(percentages, columns=['Percentage of Notices'])

# Display the results
print(percentages_df)

# # Optionally, save the results to a CSV file
# output_path = 'path_to_save_output/pofma_notices_by_actor.csv'
# percentages_df.to_csv(output_path, index=True)
