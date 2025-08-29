import requests
import json
from concurrent.futures import ThreadPoolExecutor

# Initialize empty list to store all features
all_features = []

urls = [
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13648595.770600496%2C4559315.863155987%2C-13638811.830979995%2C4569099.802776488&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4559315.863155987%2C-13629027.891359497%2C4569099.802776488&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13629027.891359497%2C4559315.863155987%2C-13619243.951738995%2C4569099.802776488&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13619243.951738995%2C4559315.863155987%2C-13609460.012118496%2C4569099.802776488&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13648595.770600496%2C4549531.923535489%2C-13638811.830979995%2C4559315.863155987&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4549531.923535489%2C-13629027.891359497%2C4559315.863155987&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13629027.891359497%2C4549531.923535489%2C-13619243.951738995%2C4559315.863155987&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13619243.951738995%2C4549531.923535489%2C-13609460.012118496%2C4559315.863155987&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13648595.770600496%2C4539747.983914988%2C-13638811.830979995%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13629027.891359497%2C4539747.983914988%2C-13619243.951738995%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13619243.951738995%2C4539747.983914988%2C-13609460.012118496%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13648595.770600496%2C4529964.044294488%2C-13638811.830979995%2C4539747.983914988&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4529964.044294488%2C-13629027.891359497%2C4539747.983914988&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13629027.891359497%2C4529964.044294488%2C-13619243.951738995%2C4539747.983914988&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13619243.951738995%2C4529964.044294488%2C-13609460.012118496%2C4539747.983914988&maxRecordCountFactor=4&resultOffset=0&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13648595.770600496%2C4539747.983914988%2C-13638811.830979995%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=8000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=16000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=24000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=32000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=40000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=48000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=56000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=64000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=72000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=80000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=88000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=96000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4539747.983914988%2C-13629027.891359497%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=104000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13629027.891359497%2C4549531.923535489%2C-13619243.951738995%2C4559315.863155987&maxRecordCountFactor=4&resultOffset=8000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13629027.891359497%2C4549531.923535489%2C-13619243.951738995%2C4559315.863155987&maxRecordCountFactor=4&resultOffset=16000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13629027.891359497%2C4539747.983914988%2C-13619243.951738995%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=8000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13629027.891359497%2C4539747.983914988%2C-13619243.951738995%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=16000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13629027.891359497%2C4539747.983914988%2C-13619243.951738995%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=24000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13629027.891359497%2C4539747.983914988%2C-13619243.951738995%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=32000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13629027.891359497%2C4539747.983914988%2C-13619243.951738995%2C4549531.923535489&maxRecordCountFactor=4&resultOffset=40000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4529964.044294488%2C-13629027.891359497%2C4539747.983914988&maxRecordCountFactor=4&resultOffset=8000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
    "https://services.arcgis.com/Zs2aNLFN00jrS4gG/arcgis/rest/services/Proposed_Rezoning_Heights_Public/FeatureServer/0/query?f=geojson&geometry=-13638811.830979995%2C4529964.044294488%2C-13629027.891359497%2C4539747.983914988&maxRecordCountFactor=4&resultOffset=16000&resultRecordCount=8000&where=1%3D1&orderByFields=OBJECTID&outFields=DAG213%2COBJECTID&geometryType=esriGeometryEnvelope&defaultSR=102100",
]

# Add additional query parameters to each URL
urls = [
    url + "&outFields=*&outSR=4326" if "?" in url 
    else url + "?outFields=*&outSR=4326" 
    for url in urls
]

# Replace f=pbf with f=geojson in all URLs
urls = [url.replace("f=pbf", "f=geojson") for url in urls]

def fetch_and_parse(url):
    """Download GeoJSON data and filter features where either DAG213 or DAG214 is not null"""
    print(f"Fetching: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            if "features" in data:
                # Filter features where either DAG213 or DAG214 is not null
                filtered_features = [
                    feature for feature in data["features"]
                    if (feature["properties"].get("DAG213") is not None 
                        and feature["properties"]["DAG213"] != "")  # Check DAG213
                    or (feature["properties"].get("DAG214") is not None 
                        and feature["properties"]["DAG214"] != "")  # Check DAG214
                ]
                print(f"Parsed {len(filtered_features)} valid features from {url}")
                return filtered_features
            else:
                print(f"Warning: No features found in response")
                return []
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return []
    else:
        print(f"Failed to fetch {url}, status code: {response.status_code}")
        return []

# Use thread pool to speed up downloads
with ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(fetch_and_parse, urls)

# Merge all features
for result in results:
    all_features.extend(result)

# Save to GeoJSON
geojson_data = {
    "type": "FeatureCollection",
    "crs": {
        "type": "name",
        "properties": {
            "name": "EPSG:4326"
        }
    },
    "features": all_features
}

with open("sf_zoning_2023.geojson", "w") as f:
    json.dump(geojson_data, f, indent=2)

print(f"GeoJSON saved as sf_zoning_2023.geojson with {len(all_features)} features") 