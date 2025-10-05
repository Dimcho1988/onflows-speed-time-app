import pandas as pd
def min_to_hms(minutes: float) -> str:
    if pd.isna(minutes) or minutes<=0: return "-"
    tot = int(round(minutes*60)); h=tot//3600; m=(tot%3600)//60; s=tot%60
    return f"{h}:{m:02d}:{s:02d}" if h>0 else f"{m}:{s:02d}"
