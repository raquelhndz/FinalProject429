# -*- coding: utf-8 -*-
"""DashInterfaceWork.ipynb

"""

# imports
import dash
from dash import callback, Dash, dcc, html, no_update
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import json
import torch
from sentence_transformers import SentenceTransformer, util
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import logging


# Add logger for error messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Initializing Dash app...")

# loading geopy object
geolocator = Nominatim(user_agent="Community_Matching")

# Create the Dash app
app = Dash(__name__)
server = app.server

logger.info("Loading SentenceTransformer model...")
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
logger.info("Model loaded successfully!")


# List of US states with abbreviations
US_STATES_ABBR = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

app.layout = html.Div(
    children=[
        # Title and Logo
        html.Div(
            children=[
                html.H1("Project Invent", style={"text-align": "center", "font-size": "24px"}),
                html.H2("Organization Matching", style={"text-align": "center", "font-size": "16px"}),
            ]
        ),

        # Input Section
        html.Div(
            children=[
                html.Label("What is your topic of interest?", style={"font-weight": "bold", "color": "#1B1464", "display": "block"}),
                dcc.Input(
                    id="topic-input",
                    type="text",
                    placeholder="Enter a topic",
                    style={"margin": "0 auto", "display": "block", "margin-bottom": "15px", "width": "50%", "padding": "10px", "color": "#333566"},
                ),
                html.Label("What city are you in?", style={"font-weight": "bold", "color": "#1B1464", "display": "block"}),
                dcc.Input(
                    id="city-input",
                    type="text",
                    placeholder="Enter a city",
                    style={"margin": "0 auto", "display": "block", "margin-bottom": "15px", "width": "50%", "padding": "10px", "color": "#333566"},
                ),
                html.Label("Select your state:", style={"font-weight": "bold", "color": "#1B1464", "display": "block"}),
                dcc.Dropdown(
                    id="state-dropdown",
                    options=[{"label": state, "value": state} for state in US_STATES_ABBR],
                    placeholder="Select a state",
                    style={"margin": "0 auto", "display": "block", "margin-bottom": "15px", "width": "50%", "padding": "10px", "color": "#333566"},
                ),
                html.Label("What is your zip code?", style={"font-weight": "bold", "color": "#1B1464", "display": "block"}),
                dcc.Input(
                    id="zipcode-input",
                    type="text",
                    placeholder="Enter a zip code",
                    style={"margin": "0 auto", "display": "block", "margin-bottom": "15px", "width": "50%", "padding": "10px", "color": "#333566"},
                ),
                html.Button("Submit", id="submit-button", n_clicks=0, style={"margin-top": "15px", "color": "#333566"}),
                dcc.Loading(id="loading-bar", children=[html.Div(id="loading-bar-output")], style={"position": "relative", "top": "115px"}),
            ],
            style={"width": "50%", "margin": "0 auto", "text-align": "center", "backgroundColor":"white"},
        ),

        dcc.Loading(
            id="loading-screen",
            type="dot",
            fullscreen=True,
            children=[
                html.Div(id="data-status", style={"display": "none"}),
            ],
        ),

        # Results Section
        html.Div(
            id="results-div",
            children=[
                html.H3("Your top 5 results:", style={"margin-top": "30px", "color": "#1B1464"}),
                html.Div(id="results-list"),
            ],
            style={"text-align": "center", "margin-top": "20px"},
        ),
    ],
    style={"font-family": "Arial, sans-serif", "padding": "20px", "backgroundColor":"white"},
)

# Load the data
@app.callback(
    Output("data-status", "children"),
    Input("data-status", "children"),
    prevent_initial_call=False,  # Ensures the callback runs on app load
)
def load_data(input_data):
    global df_cities, final_df
    logger.info("Loading datasets...")
    df_cities = pd.read_pickle('df_cities.pkl')
    logger.info(f"{df_cities.shape}")
    final_df = pd.read_pickle('final_df.pkl')
    logger.info(f"{final_df.shape}")
    return ""

# Helper function to find nearby cities for a given (latitude, longitude) location
# out of the list of cities, which has tuples of (city, state, coordinates)
# and within a threshold, which defaults to 30 miles (higher threshold makes it take longer to run)
def find_nearby_cities(location, loc_state, cities, threshold=15):
    if not location:
        print(f"location not found.")

    nearby_cities = []

    # Calculate distance to each other city
    for city, state, coordinates in cities:
        if loc_state == state:
            coords = tuple(map(float, coordinates.strip("()").split(',')))
            distance = geodesic(location, coords).miles
            if distance <= threshold:
                nearby_cities.append((city, state))

    return nearby_cities

# Define callback for button click
@app.callback(
    [Output("results-list", "children"),
    Output("topic-input", "value"),
    Output("city-input", "value"),
    Output("state-dropdown", "value"),
    Output("zipcode-input", "value"),
    Output("loading-bar-output", "children")],
    [Input("submit-button", "n_clicks")],
    [State("topic-input", "value"),
    State("city-input", "value"),
    State("state-dropdown", "value"),
    State("zipcode-input", "value")],
)

def update_results(n_clicks, topic, city, state, zipcode):
    if n_clicks > 0 and topic and city and state and zipcode:

        time.sleep(2)

        input_location = f"{city}, {state}"

        # Geocode user location
        user_location = geolocator.geocode(input_location, country_codes="US")
        # error checking
        if user_location is None:
            return [
                html.P("Could not find your location. Please try again.", style={"color": "red"})
            ], no_update, no_update, no_update, no_update, no_update

        # get cities within 30 miles and only look at those entries in the dataframe
        cities_list = list(zip(df_cities['city'], df_cities['state'], df_cities['coordinates']))
        nearby_cities = find_nearby_cities((user_location.latitude, user_location.longitude), state, cities_list)

        nearby_data = {
            'embeddings': [],
            'names': [],
            'cities': [],
            'states': [],
            'descriptions': [],
            'websites': []
        }

        # make all strings uppercase so checking for equality works
        final_df['state_upper'] = final_df['state'].str.upper()
        final_df['city_upper'] = final_df['city'].str.upper()

        # for each nearby city, get rows from the whole dataframe that are in that city and save needed info in lists
        for city, state in nearby_cities:
            df_nearby = final_df[(final_df['city_upper'] == city) & (final_df['state_upper'] == state)]
            #array_embedding = ast.literal_eval(df_nearby['embeddings'])
            nearby_data['embeddings'].extend(df_nearby['embeddings'].apply(json.loads).tolist())
            nearby_data['names'].extend(df_nearby['name'].tolist())
            nearby_data['cities'].extend(df_nearby['city'].tolist())
            nearby_data['states'].extend(df_nearby['state'].tolist())
            nearby_data['descriptions'].extend(df_nearby['Text'].tolist())
            websites = df_nearby['organization_url'].where(pd.notna(df_nearby['organization_url']), None).tolist()
            nearby_data['websites'].extend(websites)

        embeddings_nearby = nearby_data['embeddings']
        names_nearby = nearby_data['names']
        cities_nearby = nearby_data['cities']
        states_nearby = nearby_data['states']
        descriptions_nearby = nearby_data['descriptions']
        websites_nearby = nearby_data['websites']

        device = "cuda" if torch.cuda.is_available() else "cpu"

        #Compute similarity scores
        input_embedding = model.encode(topic, convert_to_tensor=True).to(device)

        embeddings_nearby = np.array(embeddings_nearby, dtype=np.float32)
        embeddings_nearby = torch.tensor(embeddings_nearby).to(device)


        # Compute cosine similarity
        cosine_scores = util.pytorch_cos_sim(input_embedding, embeddings_nearby)

        # Find the most similar descriptions
        most_similar = np.argsort(cosine_scores.numpy()[0])[::-1][:5]


        results = []
        for idx in most_similar:
            org_name = names_nearby[idx]
            org_city = cities_nearby[idx]
            org_state = states_nearby[idx]
            org_description = descriptions_nearby[idx]
            org_website = websites_nearby[idx]
            similarity_score = cosine_scores.numpy()[0][idx]

            # Create a formatted HTML div for each result
            results.append(
                html.Div(
                    children=[
                        html.H4(f"{org_name}"),
                        html.P(f"Location: {org_city}, {org_state}"),
                        html.P(f"About: {org_description}"),
                        html.P(["Website: ", 
                                html.A(org_website, href=f"https://{org_website}" if not org_website.startswith(("http://", "https://")) else org_website, target="_blank")]
                        ) if org_website else None
                    ],
                    style={"margin": "10px", "padding": "10px", "border": "1px solid #ddd"},
                )
            )

        # Return the formatted results for the Dash app
        return results, no_update, no_update, no_update, no_update, no_update
    else:
        return (
            html.P("Please fill in the text boxes and click submit.", style={"color": "gray"}),
            no_update, no_update, no_update, no_update, no_update,
        )


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080)



