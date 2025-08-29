import json

import requests

from pathlib import Path
 
# === Configuration ===

API_KEY = 'AIzaSyAQLqAdKz2yzLD8WpApdM8zBFUA5CoScvE'  # Replace this with your real API key

INPUT_FILE = './processed/agent_5.15.json'

OUTPUT_FILE = './processed/agent_5.15_with_geo.json'

RADIUS_METERS = 3218  # 2 miles in meters

PLACE_TYPES = ['park', 'school', 'supermarket', 'restaurant', 'cafe', 'transit_station', 'hospital']
 
# === Google Maps API Endpoints ===

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

PLACES_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
 
def get_neighborhood(lat, lng):

    params = {

        'latlng': f'{lat},{lng}',

        'result_type': 'neighborhood',

        'key': API_KEY

    }

    res = requests.get(GEOCODE_URL, params=params).json()

    for result in res.get('results', []):

        for comp in result.get('address_components', []):

            if 'neighborhood' in comp.get('types', []):

                return comp['long_name']

    return "Unknown"
 
def get_pois(lat, lng):

    pois = {}

    for place_type in PLACE_TYPES:

        params = {

            'location': f'{lat},{lng}',

            'radius': RADIUS_METERS,

            'type': place_type,

            'key': API_KEY

        }

        res = requests.get(PLACES_URL, params=params).json()

        pois[place_type] = [place['name'] for place in res.get('results', [])[:5]]

    return pois
 
def generate_narrative(neighborhood, pois):

    lines = [f"You live in the {neighborhood} neighborhood."]

    if pois.get("park"):

        lines.append(f"Parks nearby: {', '.join(pois['park'])}.")

    if pois.get("restaurant"):

        lines.append(f"Dining options include: {', '.join(pois['restaurant'])}.")

    if pois.get("cafe"):

        lines.append(f"Cafes nearby: {', '.join(pois['cafe'])}.")

    if pois.get("supermarket"):

        lines.append(f"Grocery stores: {', '.join(pois['supermarket'])}.")

    if pois.get("school"):

        lines.append(f"Schools: {', '.join(pois['school'])}.")

    if pois.get("transit_station"):

        lines.append(f"Public transport: {', '.join(pois['transit_station'])}.")

    if pois.get("hospital"):

        lines.append(f"Hospitals nearby: {', '.join(pois['hospital'])}.")

    return ' '.join(lines)
 
def enrich_agents(input_path, output_path):

    agents = json.loads(Path(input_path).read_text())

    enriched = []

    for agent in agents:

        lat = agent['coordinates']['lat']

        lng = agent['coordinates']['lng']

        neighborhood = get_neighborhood(lat, lng)

        pois = get_pois(lat, lng)

        narrative = generate_narrative(neighborhood, pois)

        agent['geo_content'] = {

            'neighborhood': neighborhood,

            'pois_within_2_miles': pois,

            'narrative': narrative

        }

        enriched.append(agent)

    Path(output_path).write_text(json.dumps(enriched, indent=2))

    print(f"Done. {len(enriched)} agents written to {output_path}")
 
# === Run ===

# Uncomment the line below to execute:

enrich_agents(INPUT_FILE, OUTPUT_FILE)

 