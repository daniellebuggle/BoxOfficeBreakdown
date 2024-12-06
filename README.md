# Box Office Breakdown Dashboard

This dashboard visualises key aspects of movie performance data, including profit trends across different genres, critic scores, and audience feedback over the years. It enables users to explore various movie statistics by genre and compare different attributes through dynamic, interactive visualisations.

## Features

- **Genre-Based Scatter Plot - Small Multiples**: Visualises the average profit per year for different movie genres.
- **Genre-Based Radar Chart - Small Multiples**: Compares critic and audience scores across selected genres using data from Rotten Tomatoes and Metacritic.
- **Dynamic Scatter Plot**: Displays correlations between selected attributes like critic scores and audience feedback.
- **Customizable Controls**: Users can select genres, choose which data to display on the axes, and adjust size attributes to reflect various movie statistics like profit and gross income.

## Requirements

- Python 3.x
- Dash
- Plotly
- pandas
- numpy
- dash-mantine-components
- dash-daq

You can install the required dependencies using `pip`:

```bash
pip install dash plotly pandas numpy dash-mantine-components dash-daq
```

## Project Setup
### 1. Install Dependencies
First, clone the repository and install the required dependencies.

```bash
git clone https://github.com/daniellebuggle/BoxOfficeBreakdown.git
cd box-office-breakdown
pip install dash plotly pandas numpy dash-mantine-components dash-daq
```

### 2. Dataset
The dataset used in this dashboard, The Hollywood Insider, should be saved as The Hollywood Insider.csv in the project directory.
You can modify the file path in the code if the dataset is stored elsewhere.

### 3. Running the Application
To run the dashboard locally, execute the following command in the project directory:
```bash
python app.py
```
This will start a local web server, and you can view the dashboard by navigating to where the terminal output says your server is running on. For example [http://localhost:8050](http://localhost:8050).

## Functions Overview

### `calculate_profit(dataframe)`
Calculates the profit for each movie by subtracting its budget from its worldwide gross.

### `sort_by_genre(dataframe, genre)`
Sorts the dataframe by the year for the specified genre.

### `critic_scores(dataframe, genres)`
Calculates the average critic and audience scores for each genre.

### `adjust_alpha(color, alpha_value)`
Adjusts the alpha (transparency) value of a given RGBA color string.

### `clean_data(dataframe, list_numeric)`
Cleans numeric columns by replacing invalid values (e.g., `-` and other non-numeric symbols) and converting the data to numeric types.

