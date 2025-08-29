## Data Specifications

### Technical Parameters
- Format: GeoJSON (RFC 7946)
- Coordinate Reference System: EPSG:4326 (WGS84)
- Geometry Type: Polygon
- Spatial Coverage: San Francisco City Limits

### 2023 Data Format
- Script: get_sf_zoning_map_2023.py
- Output: sf_zoning_2023.geojson

#### Schema (2023)
```json
{
  "type": "Feature",
  "properties": {
    "OBJECTID": "Integer",      // Unique identifier for each feature
    "mapblklot": "String",      // Assessor Parcel Number (APN)
    "DAG213": "String",         // Height limit description (e.g., "65' Height Allowed")
    "DAG214": "Integer",        // Numerical height limit in feet
    "Shape__Area": "Float",     // Polygon area in square feet
    "Shape__Length": "Float"    // Polygon perimeter in feet
  },
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[lon, lat], ...]]  // WGS84 coordinates array
  }
}
```

#### Sample Feature (2023)
```json
{
  "OBJECTID": 7129,
  "mapblklot": "0446002",
  "DAG213": "65' Height Allowed",
  "DAG214": 65,
  "Shape__Area": 78906.2490234375,
  "Shape__Length": 1123.8740290682704
}
```

### 2024 Data Format
- Script: get_sf_zoning_map_2024.py
- Output: sf_zoning_2024.geojson

#### Schema (2024)
```json
{
  "type": "Feature",
  "properties": {
    "OBJECTID": "Integer",      // Unique identifier for each feature
    "NEW_HEIGHT": "String",     // Height limit description (e.g., "65-foot Height")
    "NEW_HEIGHT_NUM": "Integer", // Numerical height limit in feet
    "Shape__Area": "Float",     // Polygon area in square feet
    "Shape__Length": "Float"    // Polygon perimeter in feet
  },
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[lon, lat], ...]]  // WGS84 coordinates array
  }
}
```

#### Sample Feature (2024)
```json
{
  "OBJECTID": 7129,
  "NEW_HEIGHT": "65-foot Height",
  "NEW_HEIGHT_NUM": 65,
  "Shape__Area": 78906.2490234375,
  "Shape__Length": 1123.8740290682704
}
```
