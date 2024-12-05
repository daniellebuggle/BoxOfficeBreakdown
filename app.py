from plotly.subplots import make_subplots
import numpy as np
from dash import Dash, dcc, callback, Output, Input
import pandas as pd
import dash_mantine_components as dmc
import plotly.graph_objects as go
import dash_daq as daq


def calculate_profit(dataframe):
    dataframe.loc[:, 'budget ($million)'] = dataframe['budget ($million)'].replace('-', np.nan)
    dataframe.loc[:, 'worldwide gross'] = dataframe['worldwide gross'].replace({'\$': '', ',': ''},
                                                                               regex=True).astype(
        float)
    dataframe.loc[:, 'budget ($million)'] = dataframe['budget ($million)'].replace({'\$': '', ',': ''},
                                                                                   regex=True).astype(
        float)
    dataframe = dataframe.dropna(subset=['worldwide gross', 'budget ($million)'])
    dataframe['profit'] = dataframe['worldwide gross'] - dataframe['budget ($million)']
    return dataframe


def sort_by_genre(dataframe, genre):
    dataframe.columns = ['year', 'genre', 'average profit']
    genre_df = dataframe[dataframe['genre'] == genre]
    sorted_genre_df = genre_df.sort_values(by='year')
    return sorted_genre_df


def get_genre(dataframe, genre):
    return dataframe[dataframe['primary genre'] == genre]


def critic_scores(dataframe, genres):
    genre_scores = {}
    for genre in genres:
        genre_df = dataframe[dataframe['primary genre'] == genre]
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
    rgba_values[-1] = str(alpha_value)
    return f"rgba({', '.join(rgba_values)})"


def clean_data(dataframe, list_numeric):
    for entry in list_numeric:
        dataframe.loc[:, entry] = dataframe[entry].replace('-', np.nan)
        dataframe.loc[:, entry] = dataframe[entry].replace({'\$': '', ',': ''}, regex=True).astype(float)
        dataframe = dataframe.dropna(subset=[entry])
    return dataframe


numeric_attributes = [
    "rotten tomatoes critics", "metacritic critics", "average critics ",
    "rotten tomatoes audience", "metacritic audience", "average audience",
    "audience vs critics deviance", "opening weekend ($million)",
    "opening weekend", "domestic gross ($million)", "domestic gross",
    "foreign gross ($million)", "foreign gross", "worldwide gross",
    "worldwide gross ($million)", "budget ($million)"
]

size_attributes = ["profit", "opening weekend ($million)",
                   "domestic gross ($million)",
                   "foreign gross ($million)",
                   "worldwide gross ($million)"]

numeric_attributes = [attr.strip().lower() for attr in numeric_attributes]  # Cleaned and lowercase

df = pd.read_csv('The Hollywood Insider.csv')
df.columns = df.columns.str.strip().str.lower()
df['primary genre'] = df['primary genre'].str.lower()

df = clean_data(df, numeric_attributes)
df = calculate_profit(df)
df['profit'] = pd.to_numeric(df['profit'], errors='coerce')
df['worldwide gross ($million)'] = pd.to_numeric(df['worldwide gross ($million)'], errors='coerce')

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

alpha_value = 0.3
adjusted_color_map = {genre: adjust_alpha(color, alpha_value) for genre, color in color_map.items()}
color_map_table = {genre: adjust_alpha(color, 0.6) for genre, color in color_map.items()}
critic_categories = ['rotten tomatoes critics', 'metacritic critics', 'rotten tomatoes audience',
                     'metacritic audience']

dropdown_categories = ["film", "rotten tomatoes critics", "metacritic critics", "average critics ",
                       "rotten tomatoes audience", "metacritic audience", "average audience",
                        "script type", "oscar winners", "oscar detail"]

app = Dash()
app.layout = dmc.Container([
    dcc.ConfirmDialog(
        id='genre_error_message',
        message=''
    ),
    dmc.Grid([
        dmc.Col([
            dmc.Title('Box Office Breakdown', color="blue", size="h3"),
            dcc.Dropdown(all_genres, id="genre_select", placeholder="Select genres", value=initial_genres, multi=True),
            dcc.Graph(
                id="scatter_multiples",
                figure={},
                style={'height': '80vh'},
                config={"responsive": True}
            ),
            dmc.Container(id="message", style={"marginTop": "20px"})
        ], span=6),
        dmc.Col([
            dcc.Graph(
                id="radar",
                figure={},
                style={'height': '45vh', 'marginBottom': '30px'},
                config={"scrollZoom": True, 'showTips': True, "responsive": True}
            ),
            dmc.Grid([
                dmc.Col([
                    dcc.Dropdown(
                        id="x_axis_dropdown",
                        options=[{"label": col, "value": col} for col in dropdown_categories],
                        placeholder="X-axis",
                        value="rotten tomatoes critics",
                        style={'width': '100%'}
                    ),
                ], span=3, style={'display': 'flex', 'align-items': 'right'}),

                dmc.Col([
                    dcc.Dropdown(
                        id="y_axis_dropdown",
                        options=[{"label": col, "value": col} for col in dropdown_categories],
                        placeholder="Y-axis",
                        value="metacritic critics",
                        style={'width': '100%'}
                    )
                ], span=3, style={'display': 'flex', 'align-items': 'right'}),
                dmc.Col([
                    daq.BooleanSwitch(
                        id='genre_sort_switch',
                        on=False,
                        label={
                            "label": "Genres",
                            "style": {
                                "fontFamily": "Arial, sans-serif",
                                "fontSize": "16px",
                                "fontWeight": "600",
                                "color": "#333333",
                                "marginRight": "100px",
                                "paddingBottom": "8px",
                                "paddingLeft": "8px"
                            }
                        },
                        labelPosition='right',
                    )
                ], span=2, style={'display': 'flex', 'align-items': 'right'}),
                dmc.Col([
                    dcc.Dropdown(
                        id="encode_size",
                        options=[{"label": col, "value": col} for col in size_attributes],
                        placeholder="Size Attribute",
                        style={'width': '100%'}
                    )
                ], span=3, style={'display': 'flex', 'align-items': 'left'}),
            ], align="center"),

            # Scatter plot to show the correlation between the two selected attributes
            dcc.Graph(
                id="scatter_plot",
                figure={},
                style={'height': '45vh'},
                config={"scrollZoom": True, 'showTips': True, "responsive": True}
            ),
        ], span=6)
    ], style={"width": "100%", "height": "100%"}),
], style={"width": "100%", "maxWidth": "100%"})


@callback(
    Output(component_id='scatter_multiples', component_property='figure'),
    Input(component_id='genre_select', component_property='value')
)
def update_graph(genres_chosen):
    if not genres_chosen:
        return go.Figure()
    filtered_df = df[df['primary genre'].str.lower().isin(genres_chosen)]
    average_profit_by_genre_year = filtered_df.groupby(['year', 'primary genre'])['profit'].mean().reset_index()
    # Rename columns for clarity
    average_profit_by_genre_year.columns = ['year', 'genre', 'average profit']

    fig = make_subplots(rows=len(genres_chosen), cols=1, subplot_titles=genres_chosen)
    x = 1
    for genre in genres_chosen:
        data = sort_by_genre(average_profit_by_genre_year, genre)
        fig.add_trace(
            go.Scatter(name=genre, x=data['year'].tolist(), y=data['average profit'].tolist(),
                       line=dict(color=color_map.get(genre, "gray"))),
            row=x, col=1
        )
        x = x + 1
    fig.update_layout(
        title_text="Average profit per year by genre",
        title_x=0.45,
        autosize=True,
        margin=dict(l=50, r=50, t=100, b=50)
    )
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
        [{"type": "polar"}, {"type": "polar"}, {"type": "polar"}],
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
        fig.update_layout(
            title_text="Critic Scores", title_x=0.45,
            legend=dict(
                title="genres",
                orientation="v",
                x=1.05,
                y=0.5,
            ),
        )
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                # Alternate rotation positions based on row and column
                if (row + col) % 2 == 0:
                    rotation = 0
                else:
                    rotation = 45
                fig.update_polars(
                    row=row, col=col,
                    radialaxis=dict(
                        range=[0, 100],
                        tickvals=[0, 20, 40, 60, 80, 100],
                        ticktext=["0", "20", "40", "60", "80", "100"],
                        showline=False,
                    ),
                    angularaxis=dict(
                        tickvals=list(range(len(critic_categories))),
                        ticktext=["Meta Critic", "RT Audience", "Meta Audience", "RT Critic"],  # Custom labels
                        showline=False,
                        rotation=rotation
                    )
                )
    return fig


@app.callback(
    Output("scatter_plot", "figure"),
    Input("x_axis_dropdown", "value"),
    Input("y_axis_dropdown", "value"),
    Input("genre_select", "value"),
    Input("genre_sort_switch", "on"),
    Input("encode_size", "value")
)
def update_scatter_plot(x_attr, y_attr, genres_chosen, genre_sort_on, encode_size):
    if not x_attr or not y_attr:
        return go.Figure()  # Return an empty figure if one or both attributes are not selected

    if ((x_attr == "film" and y_attr == "oscar winners") or (x_attr == "oscar winners" and y_attr == "film") or
            (x_attr == "film" and y_attr == "oscar detail") or (x_attr == "oscar detail" and y_attr == "film") or
            (x_attr == "film" and y_attr == "script type") or (x_attr == "script type" and y_attr == "film") or
            (x_attr == "oscar winners" and y_attr == "script type") or (
                    x_attr == "script type" and y_attr == "oscar winners") or
            (x_attr == "oscar winners" and y_attr == "oscar detail") or (
                    x_attr == "oscar detail" and y_attr == "oscar winners") or
            (x_attr == "script type" and y_attr == "oscar detail") or (
                    x_attr == "oscar detail" and y_attr == "script type")
    ):
        filtered_df = df.copy()
        filtered_df = filtered_df.dropna(subset=[x_attr, y_attr])

        if x_attr == "oscar detail" or y_attr == "oscar detail":
            filtered_df = filtered_df.sort_values(by=['oscar detail', 'film'])
        else:
            filtered_df = filtered_df.sort_values(by=['script type', 'film'])

        # Filter by selected genres if applicable
        if genre_sort_on:
            filtered_df = filtered_df[filtered_df['primary genre'].str.lower().isin([g.lower() for g in genres_chosen])]
            if x_attr == "oscar winners" or y_attr == "oscar winners":
                if x_attr == "script type" or y_attr == "script type":
                    filtered_df = filtered_df.sort_values(by=['primary genre', 'script type'])
                elif x_attr == "oscar detail" or y_attr == "oscar detail":
                    filtered_df = filtered_df.sort_values(by=['primary genre', 'oscar detail'])
                else:
                    filtered_df = filtered_df.sort_values(by=['primary genre', 'film'])
            elif x_attr == "oscar detail" or y_attr == "oscar detail":
                filtered_df = filtered_df.sort_values(by=['primary genre', 'oscar detail'])
            else:
                filtered_df = filtered_df.sort_values(by=['primary genre', 'script type'])

        row_colors = [color_map_table.get(genre.lower(), 'white') for genre in filtered_df["primary genre"]]

        # Create the table figure
        table_fig = go.Figure(data=[go.Table(
            header=dict(
                values=[x_attr.title(), "Primary Genre", y_attr.title()],
                fill_color='lightgrey',
                align='left'
            ),
            cells=dict(
                values=[
                    filtered_df[x_attr],
                    filtered_df["primary genre"],
                    filtered_df[y_attr]
                ],
                # Assign row colors dynamically based on the genre
                fill_color=[
                    row_colors,  # Film column colors (row-based color matching to genre)
                    row_colors,  # Primary Genre column colors (row-based color matching to genre)
                    row_colors  # Oscar Winner column with white background
                ],
                align='left'
            )
        )])
        table_fig.update_layout(
            title=f"Table: {x_attr.title()} and {y_attr.title()} Status",  # Adjust height as needed
        )

        return table_fig

    filtered_df = df.copy()

    if x_attr in numeric_attributes:
        filtered_df[x_attr] = pd.to_numeric(filtered_df[x_attr], errors='coerce')
    if y_attr in numeric_attributes:
        filtered_df[y_attr] = pd.to_numeric(filtered_df[y_attr], errors='coerce')
    filtered_df = filtered_df.dropna(subset=[x_attr, y_attr])
    filtered_df = filtered_df.sort_values(by=[x_attr, y_attr], ascending=True)
    fig = go.Figure()

    if genre_sort_on:
        # If genre sorting is on, create a scatter line for each genre
        for genre in filtered_df['primary genre'].unique():
            filtered_df = filtered_df[
                filtered_df['primary genre'].str.lower().isin([genre.lower() for genre in genres_chosen])]
            genre_df = filtered_df[filtered_df['primary genre'] == genre]
            hover_text, size = encoding_size(encode_size, genre_df)
            fig.add_trace(go.Scatter(
                x=genre_df[x_attr],
                y=genre_df[y_attr],
                mode='markers',
                name=genre,
                marker=dict(color=color_map.get(genre, 'gray'), size=size),
                text=hover_text
            ))
    else:
        hover_text, size = encoding_size(encode_size, filtered_df)
        fig.add_trace(go.Scatter(
            x=filtered_df[x_attr],
            y=filtered_df[y_attr],
            mode='markers',
            name=f"{x_attr} vs {y_attr}",
            marker=dict(color=filtered_df['primary genre'].apply(lambda x: color_map.get(x, 'gray')), size=size),
            text=hover_text

        ))
    fig.update_layout(
        title=f"Scatter Plot: {x_attr} vs {y_attr}",
        xaxis_title=x_attr,
        yaxis_title=y_attr,
        yaxis=dict(range=[filtered_df[y_attr].min(), filtered_df[y_attr].max()]),
        xaxis=dict(range=[filtered_df[x_attr].min(), filtered_df[x_attr].max()]),
        hovermode="closest",
        legend=dict(
            tracegroupgap=0,
            itemsizing="constant"  # Set the fixed size of the markers in the legend
        )
    )

    return fig


def encoding_size(encode_size, genre_df):
    if encode_size and encode_size in size_attributes:
        size = genre_df[encode_size]
        hover_text = genre_df.apply(lambda row: f"{row['film']}<br>{encode_size}: ${row[encode_size]:,.2f}",
                                    axis=1)
        if encode_size == 'profit':
            size = size / 10000000
        elif encode_size == 'domestic gross ($million)':
            size = size / 5
        elif encode_size == 'foreign gross ($million)':
            size = size / 8
        elif encode_size == 'worldwide gross ($million)':
            size = size / 10
        else:
            size = size / 2

    else:
        size = 5
        hover_text = genre_df['film']
    return hover_text, size


if __name__ == '__main__':
    app.run(debug=True)
