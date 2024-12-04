import pandas as pd

# Assuming you have a DataFrame named df with a column 'Primary genre'
df = pd.read_csv('The Hollywood Insider.csv')  # Load your dataset if needed

df['Primary Genre'] = df['Primary Genre'].str.lower()
# Calculate the count of each genre
genre_counts = df['Primary Genre'].value_counts()

genres_of_interest = ['action', 'comedy', 'drama', 'adventure', 'horror']

# Filter the DataFrame to include only the rows with the specific genres
filtered_df = df[df['Primary Genre'].str.lower().isin(genres_of_interest)]

# Check the first few rows to ensure it's correct
print(filtered_df.head())

# Print the result
print(genre_counts)
