# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import random
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Install DEAP if missing
# ---------------------------------------------------------------------
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap", "-q"])
    from deap import base, creator, tools, algorithms

st.set_page_config(page_title="LCA Optimization: Cost vs GWP", layout="wide")
random.seed(42)

# ---------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------
CORE_KEYS = {"Material", "Unit"}
CORE_MIN_INPUTS = {"Material", "Unit", "Amount"}
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

COST_CANDIDATES = [
    "Unit Cost ($)", "Unit Cost", "Cost/Unit", "Cost per Unit", "unit_cost", "cost"
]
GWP_CANDIDATES = [
    "kg CO2-Eq/Unit", "kg CO2 eq / Unit", "kg CO2-eq per unit", "GWP", "Climate change"
]

def norm(s: str) -> str:
    return "".join(c for c in str(s).strip().lower() if c.isalnum())

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # exact normalized match first
    cmap = {norm(c): c for c in df.columns}
    for cand in candidates:
        nc = norm(cand)
        if nc in cmap:
            return cmap[nc]
    # substring fallback
    for c in df.columns:
        nc = norm(c)
        if any(norm(x) in nc for x in candidates):
            return c
    return None

def to_num(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float, np.number)): return float(val)
    s = str(val).replace(",", "").strip()
    try: return float(s)
    except Exception: return np.nan

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    return out

def list_impact_cols(df: pd.DataFrame, cost_col: str) -> List[str]:
    exclude = {"Year", "Material", "Unit", "Amount", cost_col}
    cols = []
    for c in df.columns:
        if c in exclude: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

# ---------------------------------------------------------------------
# Auto-detect helpers for Excel books
# ---------------------------------------------------------------------
def detect_single_merged_sheet(book: Dict[str, pd.DataFrame]) -> Optional[str]:
    for name, df in book.items():
        if CORE_MIN_INPUTS.issubset(set(df.columns)):
            if find_col(df, COST_CANDIDATES) or find_col(df, GWP_CANDIDATES):
                return name
    return None

def detect_inputs_sheet(book: Dict[str, pd.DataFrame]) -> Optional[str]:
    for name, df in book.items():
        if CORE_MIN_INPUTS.issubset(set(df.columns)):
            return name
    return None

def detect_costs_sheet(book: Dict[str, pd.DataFrame]) -> Optional[Tuple[str, str]]:
    for name, df in book.items():
        if CORE_KEYS.issubset(set(df.columns)):
            col = find_col(df, COST_CANDIDATES)
            if col:
                return name, col
    return None

def detect_impacts_sheet(book: Dict[str, pd.DataFrame]) -> Optional[str]:
    for name, df in book.items():
        if CORE_KEYS.issubset(set(df.columns)):
            if find_col(df, GWP_CANDIDATES):
                return name
            # heuristic: has at least one numeric impact-like column
            numericish = sum(
                pd.api.types.is_numeric_dtype(df[c]) or "kg" in c.lower() or "ctu" in c.lower()
                for c in df.columns if c not in CORE_KEYS
            )
            if numericish >= 1:
                return name
    return None

# ---------------------------------------------------------------------
# Load data (ALWAYS returns a 3-tuple; first item is a 7-tuple)
# ---------------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    """
    Returns a triple:
      (
        (merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col),
        diag_df,                      # what detection path was used
        cost_col                      # resolved cost column name
      )
    If no valid file, returns ((None,...), diagnostics, default_cost_name).
    """
    empty7 = (None, None, None, None, None, None, None)
    default_cost = "Unit Cost ($)"
    diag = pd.DataFrame([{"mode": "none", "inputs_sheet": None, "costs_sheet": None, "impacts_sheet": None}])

    if uploaded_file is None:
        return (empty7, diag, default_cost)

    name = uploaded_file.name.lower()

    # ---------- CSV path (assume merged table) ----------
    if name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
        src = {"mode": "csv", "inputs_sheet": None, "costs_sheet": None, "impacts_sheet": None}
    else:
        # ---------- Excel path ----------
        book = pd.read_excel(uploaded_file, sheet_name=None)
        book = {n: sanitize_columns(d) for n, d in book.items()}

        merged_name = detect_single_merged_sheet(book)
        if merged_name:
            df_raw = book[merged_name].copy()
            src = {"mode": "excel-merged", "inputs_sheet": merged_name, "costs_sheet": merged_name, "impacts_sheet": merged_name}
        else:
            inputs_name = detect_inputs_sheet(book)
            costs_info  = detect_costs_sheet(book)
            impacts_name= detect_impacts_sheet(book)

            if not inputs_name:
                return (empty7, diag, default_cost)

            inputs_df = book[inputs_name].copy()
            if "Year" not in inputs_df.columns:
                inputs_df["Year"] = 0
            inputs_df["Amount"] = inputs_df["Amount"].map(to_num)

            df_raw = inputs_df.copy()

            # Costs
            if costs_info:
                costs_name, cost_col = costs_info
                cdf = book[costs_name].copy()
                cdf[cost_col] = cdf[cost_col].map(to_num)
                df_raw = df_raw.merge(cdf[["Material","Unit",cost_col]], on=["Material","Unit"], how="left")
            else:
                cost_col = default_cost
                df_raw[cost_col] = np.nan

            # Impacts
            if impacts_name:
                idf = book[impacts_name].copy()
                for c in idf.columns:
                    if c in ("Material","Unit"): 
                        continue
                    idf[c] = idf[c].map(to_num)
                merge_cols = [c for c in idf.columns if c not in ("Material","Unit")]
                if merge_cols:
                    df_raw = df_raw.merge(idf[["Material","Unit"] + merge_cols], on=["Material","Unit"], how="left")

            src = {"mode": "excel-3sheet", "inputs_sheet": inputs_name, "costs_sheet": costs_info[0] if costs_info else None, "impacts_sheet": impacts_name}

    # ---------- Validate minimum columns ----------
    for col in ("Material","Unit","Amount"):
        if col not in df_raw.columns:
            return (empty7, pd.DataFrame([src]), default_cost)
    if "Year" not in df_raw.columns:
        df_raw["Year"] = 0

    # Save ORIGINAL rows for later year share allocation
    st.session_state["original_rows"] = df_raw.copy()

    # Resolve cost & GWP columns
    cost_col = find_col(df_raw, COST_CANDIDATES) or default_cost
    if cost_col not in df_raw.columns:
        df_raw[cost_col] = np.nan
    df_raw[cost_col] = df_raw[cost_col].map(to_num)

    gwp_col = find_col(df_raw, GWP_CANDIDATES)
    if gwp_col is None:
        st.warning("No explicit GWP column found. Add a 'kg CO2-Eq/Unit' (or similar) column.")
        gwp_col = "kg CO2-Eq/Unit"
        if gwp_col not in df_raw.columns:
            df_raw[gwp_col] = np.nan
    df_raw[gwp_col] = df_raw[gwp_col].map(to_num)

    # ---------- Roll up to totals per Material+Unit ----------
    rolled = (df_raw.groupby(["Material","Unit"], as_index=False)
              .agg(Amount=("Amount","sum")))
    # carry first non-null for other columns
    for c in [x for x in df_raw.columns if x not in ("Material","Unit","Amount","Year")]:
        rolled[c] = (df_raw.groupby(["Material","Unit"])[c]
                     .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
                     .reset_index(drop=True))

    # If you use NPK(19-19-19) proxying to 15-15-15, scale mass only
    m = rolled["Material"].str.contains("19-19-19", case=False, na=False)
    if m.any():
        rolled.loc[m, "Amount"] = rolled.loc[m, "Amount"] * (19.0/15.0)
        rolled.loc[m, "Material"] = "NPK (15-15-15) fertiliser"

    # Build outputs
    merged_df = rolled.fillna(0.0)
    materials = merged_df["Material"].tolist()
    base_amounts = merged_df["Amount"].to_numpy(float)
    costs = merged_df[cost_col].fillna(0.0).to_numpy(float)
    impact_cols = list_impact_cols(merged_df, cost_col)
    impact_df = merged_df[impact_cols].copy().fillna(0.0)

    tuple7 = (merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col)
    return (tuple7, pd.DataFrame([src]), cost_col)

# ---------------------------------------------------------------------
# Manual mapping UI (only used if auto-detect fails)
# ---------------------------------------------------------------------
def manual_map_excel(uploaded) -> Optional[Tuple[tuple, pd.DataFrame, str]]:
    try:
        book = pd.read_excel(uploaded, sheet_name=None)
    except Exception:
        return None
    book = {n: sanitize_columns(d) for n, d in book.items()}
    sheet_names = list(book.keys())

    st.info("Auto-detect failed. Map your sheets/columns below.")

    s_inputs  = st.selectbox("Inputs sheet (must contain quantities)", sheet_names, key="map_inputs")
    s_costs   = st.selectbox("Costs sheet (optional)", ["(none)"] + sheet_names, key="map_costs")
    s_impacts = st.selectbox("Impacts sheet (optional)", ["(none)"] + sheet_names, key="map_impacts")

    df_in = book[s_inputs].copy()
    st.caption("Preview: Inputs sheet")
    st.dataframe(df_in.head(), use_container_width=True)

    col_mat  = st.selectbox("Inputs: Material", df_in.columns)
    col_unit = st.selectbox("Inputs: Unit", df_in.columns, index=min(1, len(df_in.columns)-1))
    col_amt  = st.selectbox("Inputs: Amount", df_in.columns, index=min(2, len(df_in.columns)-1))
    col_year = st.selectbox("Inputs: Year (optional)", ["(none)"] + df_in.columns.tolist())

    inputs_df = pd.DataFrame({
        "Material": df_in[col_mat],
        "Unit": df_in[col_unit],
        "Amount": pd.to_numeric(df_in[col_amt], errors="coerce"),
    })
    if col_year != "(none)":
        inputs_df["Year"] = pd.to_numeric(df_in[col_year], errors="coerce").fillna(0).astype(int)
    else:
        inputs_df["Year"] = 0

    # Costs
    costs_df = None
    cost_col_final = "Unit Cost ($)"
    if s_costs != "(none)":
        df_c = book[s_costs].copy()
        st.caption("Preview: Costs sheet")
        st.dataframe(df_c.head(), use_container_width=True)
        c_mat  = st.selectbox("Costs: Material", df_c.columns, key="c_mat")
        c_unit = st.selectbox("Costs: Unit", df_c.columns, key="c_unit")
        c_val  = st.selectbox("Costs: Cost column", df_c.columns, key="c_val")
        costs_df = pd.DataFrame({
            "Material": df_c[c_mat],
            "Unit": df_c[c_unit],
            cost_col_final: pd.to_numeric(df_c[c_val], errors="coerce"),
        })

    # Impacts
    imp_df = None
    if s_impacts != "(none)":
        df_i = book[s_impacts].copy()
        st.caption("Preview: Impacts sheet")
        st.dataframe(df_i.head(), use_container_width=True)
        i_mat  = st.selectbox("Impacts: Material", df_i.columns, key="i_mat")
        i_unit = st.selectbox("Impacts: Unit", df_i.columns, key="i_unit")
        impact_cols_pick = st.multiselect(
            "Impacts: pick numeric columns (include your GWP column)",
            [c for c in df_i.columns if c not in (i_mat, i_unit)]
        )
        if impact_cols_pick:
            imp_df = df_i[[i_mat, i_unit] + impact_cols_pick].copy()
            imp_df.columns = ["Material", "Unit"] + impact_cols_pick

    # Merge
    merged = inputs_df.copy()
    if costs_df is not None:
        merged = merged.merge(costs_df, on=["Material","Unit"], how="left")
    if imp_df is not None:
        merged = merged.merge(imp_df, on=["Material","Unit"], how="left")

    # Save original rows for year shares
    st.session_state["original_rows"] = merged.copy()

    # Roll-up using the same pipeline as loader
    # (We reuse in-memory CSV to keep code simple)
    buf = StringIO()
    merged.to_csv(buf, index=False)
    buf.seek(0)
    df_raw = pd.read_csv(buf)

    if "Year" not in df_raw.columns:
        df_raw["Year"] = 0

    cost_col = find_col(df_raw, COST_CANDIDATES) or cost_col_final
    if cost_col not in df_raw.columns:
        df_raw[cost_col] = 0.0
    df_raw[cost_col] = df_raw[cost_col].map(to_num)

    gwp_col = find_col(df_raw, GWP_CANDIDATES) or "kg CO2-Eq/Unit"
    if gwp_col not in df_raw.columns:
        df_raw[gwp_col] = 0.0
    df_raw[gwp_col] = df_raw[gwp_col].map(to_num)

    rolled = (df_raw.groupby(["Material","Unit"], as_index=False)
              .agg(Amount=("Amount","sum")))
    for c in [x for x in df_raw.columns if x not in ("Material","Unit","Amount","Year")]:
        rolled[c] = (df_raw.groupby(["Material","Unit"])[c]
                     .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
                     .reset_index(drop=True))

    merged_df = rolled.fillna(0.0)
    materials = merged_df["Material"].tolist()
    base_amounts = merged_df["Amount"].to_numpy(float)
    costs = merged_df[cost_col].fillna(0.0).to_numpy(float)
    impact_cols = list_impact_cols(merged_df, cost_col)
    impact_df = merged_df[impact_cols].copy().fillna(0.0)

    tuple7 = (merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col)
    diag = pd.DataFrame([{"mode":"manual-map", "inputs_sheet": s_inputs, "costs_sheet": s_costs, "impacts_sh_]()
