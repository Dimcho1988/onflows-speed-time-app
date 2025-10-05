import math

def parse_distance_km(x):
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().lower().replace(" ", "")
    if not s: return math.nan
    if s in {"marathon","maraton"}: return 42.195
    if s in {"half","halfmarathon","polumaraton"}: return 21.097
    if s.endswith("km"): return float(s[:-2])
    if s.endswith("k"):  return float(s[:-1])
    if s.endswith("m"):  return float(s[:-1]) / 1000.0
    try: return float(s)
    except: return math.nan

def parse_speed_kmh(x):
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().lower().replace(" ", "")
    if not s: return math.nan
    if "min/km" in s or "/km" in s:
        if ":" in s:
            mm, ss = s.split("min")[0].split(":"); pace = float(mm)+float(ss)/60
        else:
            pace = float(s.split("min")[0])
        return 60.0/pace
    try: return float("".join(ch for ch in s if ch.isdigit() or ch=="."))
    except: return math.nan

def parse_time_min(x):
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().lower()
    if ":" in s:
        parts = s.split(":")
        if len(parts)==2: m,s = parts; return float(m)+float(s)/60
        if len(parts)==3: h,m,s = parts; return 60*float(h)+float(m)+float(s)/60
    total=0.0
    for token,coef in (("h",60),("m",1),("s",1/60)):
        if token in s:
            try: total += float(s.split(token)[0].split()[-1])*coef
            except: ...
    return total if total>0 else float("nan")
