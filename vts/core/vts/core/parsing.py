import math

def parse_distance_to_km(s):
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower()
    if not s:
        return math.nan
    if s.endswith("km"):
        return float(s[:-2])
    if s.endswith("k"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) / 1000
    try:
        return float(s)
    except:
        return math.nan

def parse_speed_to_kmh(s):
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower()
    if not s:
        return math.nan
    if "min/km" in s:
        val = float(s.replace("min/km", "").strip())
        return 60 / val
    try:
        return float(s)
    except:
        return math.nan
