from plotly.subplots import make_subplots
import numpy as np
from dash import Dash, dcc, callback, Output, Input
import pandas as pd
import dash_mantine_components as dmc
import plotly.graph_objects as go


def calculate_profit(dataframe):
    dataframe.loc[:, 'Budget ($million)'] = dataframe['Budget ($million)'].replace('-', np.nan)
    dataframe.loc[:, 'Worldwide Gross'] = dataframe['Worldwide Gross'].replace({'\$': '', ',': ''},
                                                                               regex=True).astype(
        float)
    dataframe.loc[:, 'Budget ($million)'] = dataframe['Budget ($million)'].replace({'\$': '', ',': ''},
                                                                                   regex=True).astype(
        float)
    dataframe = dataframe.dropna(subset=['Worldwide Gross', 'Budget ($million)'])
    # Now calculate the profit
    dataframe['Profit'] = dataframe['Worldwide Gross'] - dataframe['Budget ($million)']
    return dataframe


def sort_by_genre(dataframe, genre):
    # Rename columns for clarity
    dataframe.columns = ['Year', 'Genre', 'Average Profit']
    genre_df = dataframe[dataframe['Genre'] == genre]
    # Sort by Genre
    sorted_genre_df = genre_df.sort_values(by='Year')
    return sorted_genre_df


def get_genre(dataframe, genre):
    return dataframe[dataframe['Primary Genre'] == genre]


def critic_scores(dataframe, genres):
    genre_scores = {}
    for genre in genres:
        genre_df = dataframe[dataframe['Primary Genre'] == genre]
        averages = []
        for category in critic_categories:
            avg = pd.to_numeric(genre_df[category], errors='coerce').mean()
            averages.append(avg)
        genre_scores[genre] = averages
    return genre_scores


def adjust_alpha(color, alpha_value):
    """
    Adjust the alpha value of an RGBA color string.
    """
    rgba_values = color[5:-1]
    rgba_values = rgba_values.split(',')
    # Replace the last value (original alpha) with the new alpha
    rgba_values[-1] = str(alpha_value)
    return f"rgba({', '.join(rgba_values)})"


df = pd.read_csv('The Hollywood Insider.csv')
df.columns = df.columns.str.strip()

# Convert 'Primary Genre' to lowercase
df['Primary Genre'] = df['Primary Genre'].str.lower()

initial_genres = ['action', 'comedy', 'drama', 'adventure', 'horror']
all_genres = ['action', 'comedy', 'drama', 'adventure', 'horror', 'thriller', 'animation', 'romance', 'sci-fi',
              'biography', 'crime', 'fantasy', 'family', 'musical']

color_map = {
    "action": "rgba(255, 0, 0, 1)",  # red
    "comedy": "rgba(0, 0, 255, 1)",  # blue
    "drama": "rgba(88, 177, 255, 1)",  # #58B1FF
    "adventure": "rgba(255, 215, 0, 1)",  # #FFD700
    "horror": "rgba(0, 128, 0, 1)",  # green
    "thriller": "rgba(255, 165, 0, 1)",  # orange
    "animation": "rgba(255, 105, 180, 1)",  # #FF69B4
    "romance": "rgba(255, 192, 203, 1)",  # pink
    "sci-fi": "rgba(245, 154, 35, 1)",  # #F59A23
    "biography": "rgba(138, 43, 226, 1)",  # #8A2BE2
    "crime": "rgba(105, 105, 105, 1)",  # #696969
    "fantasy": "rgba(128, 0, 128, 1)",  # purple
    "family": "rgba(0, 206, 209, 1)",  # #00CED1
    "musical": "rgba(145, 75, 20, 1)"  # #914b14
}

# Adjust alpha for transparency
alpha_value = 0.3
adjusted_color_map = {genre: adjust_alpha(color, alpha_value) for genre, color in color_map.items()}

critic_categories = ['Rotten Tomatoes Critics', 'Metacritic Critics', 'Rotten Tomatoes Audience',
                     'Metacritic Audience']

app = Dash()
app.layout = dmc.Container([
    dcc.ConfirmDialog(
        id='genre_error_message',
        message=''
    ),
    dmc.Grid([
        dmc.Col([
            dmc.Title('My First App with Data, Graph, and Controls', color="blue", size="h3"),
            dcc.Dropdown(all_genres, id="genre_select", placeholder="Select genres", value=initial_genres, multi=True),
            dcc.Graph(
                id="scatter_multiples",
                figure={},  # Use the Plotly figure generated
                style={'height': '80vh'}  # Adjusting the height of the plot
            ),
            dmc.Container(id="message", style={"marginTop": "20px"})
        ], span=5),
        dmc.Col([
            dcc.Graph(
                id="radar",
                figure=fig1,  # Use the Plotly figure generated
                style={'height': '45vh'}  # Adjusting the height of the plot
            ),
        ], span=5)
    ], style={"width": "100%", "height": "100%"}),
], fluid=True, style={"width": "100%", "maxWidth": "100%", "margin": "0 auto"})


@callback(
    Output(component_id='scatter_multiples', component_property='figure'),
    Input(component_id='genre_select', component_property='value')
)
def update_graph(genres_chosen):
    if not genres_chosen:
        return go.Figure()
    filtered_df = df[df['Primary Genre'].str.lower().isin(genres_chosen)]
    # Calculate profit for the movies and add a column
    filtered_df = calculate_profit(filtered_df)
    # Get the average profit of each category for each year
    average_profit_by_genre_year = filtered_df.groupby(['Year', 'Primary Genre'])['Profit'].mean().reset_index()
    # Rename columns for clarity
    average_profit_by_genre_year.columns = ['Year', 'Genre', 'Average Profit']

    fig = make_subplots(rows=len(genres_chosen), cols=1, subplot_titles=genres_chosen)
    x = 1
    for genre in genres_chosen:
        data = sort_by_genre(average_profit_by_genre_year, genre)
        fig.add_trace(
            go.Scatter(name=genre, x=data['Year'].tolist(), y=data['Average Profit'].tolist(),
                       line=dict(color=color_map.get(genre, "gray"))),
            row=x, col=1
        )
        x = x + 1
    fig.update_layout(height=750, width=600, title_text="Average Profit per Year by Genre", title_x=0.45)
    return fig


@app.callback(
    Output("genre_select", "value"),
    Output("genre_error_message", "displayed"),
    Output("genre_error_message", "message"),
    Input("genre_select", "value"),
)
def limit_selection(selected_genres):
    max_selection = 5
    if len(selected_genres) > max_selection:
        return selected_genres[:max_selection], True, f"Selection limited to {max_selection} genres."
    return selected_genres, False, ""


@app.callback(
    Output("radar", "figure"),
    Input("genre_select", "value")
)
def update_radar(genres_chosen):
    if not genres_chosen:
        return go.Figure()

    scores = critic_scores(df, genres_chosen)

    rows = 2
    cols = 3

    # Define the layout of subplots with merged cells in the second row
    specs = [
        [{"type": "polar"}, {"type": "polar"}, {"type": "polar"}],  # First row with 3 plots
        [{"type": "polar", "colspan": 2}, {"type": "polar", "colspan": 2}, None]
        # Second row with merged cells for centering
    ]

    fig = make_subplots(rows=rows, cols=cols, specs=specs)

    plot_idx = 0
    for genre, averages in scores.items():
        row = (plot_idx // cols) + 1
        col = (plot_idx % cols) + 1
        genre_color = color_map.get(genre.lower(), "gray")  # Default to "gray" if genre is not found
        genre_fillcolor = adjusted_color_map[genre]
        fig.add_trace(
            go.Scatterpolar(
                r=averages,
                theta=critic_categories,
                fill='toself',
                fillcolor=genre_fillcolor,
                line=dict(color=genre_color),
                name=genre
            ), row=row, col=col
        )
        plot_idx += 1

        # Update the layout for each subplot's radial axis
        fig.update_layout(
            height=250 * rows, width=800,
            title_text="Critic Scores", title_x=0.45,
            legend=dict(
                title="Genres",
                orientation="v",
                x=1.05,  # Position the legend outside of the plot area (right side)
                y=0.5,  # Center the legend vertically
            )
        )

        # Update polar radial axis for all subplots
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                # Alternate tick positions based on row and column
                if (row + col) % 2 == 0:  # Condition for alternating ticks
                    tickvals = [1, 2, 3, 4]  # Ticks at 0, 2, and 4 positions
                else:
                    tickvals = [0.5, 1.5, 2.5, 3.5]  # Ticks at 1 and 3 positions
                fig.update_polars(
                    row=row, col=col,
                    radialaxis=dict(
                        range=[0, 100],
                        tickvals=[0, 20, 40, 60, 80, 100],
                        ticktext=["0", "20", "40", "60", "80", "100"],
                        showline=False,
                    ),
                    angularaxis=dict(
                        tickvals=tickvals,
                        ticktext=["RT Critic", "Meta Critic", "RT Audience", "Meta Audience"],  # Custom labels
                        showline=False,
                    )
                )
    return fig


if __name__ == '__main__':
    app.run(debug=True)
