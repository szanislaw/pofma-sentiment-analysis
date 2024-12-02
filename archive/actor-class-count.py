import pandas as pd

file_path = 'inference/pofma_predictions.xlsx'
df = pd.read_excel(file_path)

actors = ['Actor: Media', 'Actor: Political Group or Figure', 'Actor: Civil Society Group or Figure',
          'Actor: Social Media Platform', 'Actor: Internet Access Provider', 'Actor: Private Individual']

percentages = (df[actors].mean() * 100).round(2)
percentages_df = pd.DataFrame(percentages, columns=['Percentage of Notices'])

print(percentages_df)
