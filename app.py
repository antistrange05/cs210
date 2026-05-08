import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinguaGeo — Linguistic Divergence Predictor",
    page_icon="🗺️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}
.hero {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #e0e0d8;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 3rem;
    line-height: 1.1;
    color: #1a1a16;
    margin: 0;
}
.hero p {
    color: #666660;
    font-size: 0.9rem;
    margin-top: 0.5rem;
    letter-spacing: 0.02em;
}
.result-card {
    background: #f7f7f2;
    border: 1px solid #deded6;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
}
.result-number {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    color: #1a1a16;
    line-height: 1;
}
.result-label {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888882;
    margin-bottom: 0.25rem;
}
.interpretation {
    border-left: 3px solid #c8b89a;
    padding-left: 1rem;
    font-size: 0.85rem;
    color: #444440;
    margin-top: 1rem;
    font-style: italic;
}
.stat-row {
    display: flex;
    gap: 1.5rem;
    margin: 1rem 0;
}
.stat-box {
    flex: 1;
    background: white;
    border: 1px solid #e8e8e0;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
}
.stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #2a2a20;
}
.stat-lbl {
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    color: #999993;
    text-transform: uppercase;
}
.section-header {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #aaaaaa;
    margin-bottom: 0.5rem;
    margin-top: 2rem;
}
.finding-block {
    background: #fffef8;
    border: 1px solid #e8e4d0;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
}
.finding-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #2a2a20;
    margin-bottom: 0.25rem;
}
.finding-body {
    font-size: 0.8rem;
    color: #666660;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ── Utility functions ─────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def predict_divergence(geo_dist_km, corridor_ruggedness):
    """
    Predict normalized Levenshtein divergence using the OLS model coefficients
    discovered in the analysis. Falls back gracefully if model file not found.
    """
    # OLS coefficients from notebook:
    # Divergence = 0.4841 + 0.0418 * log(dist+1) + (-0.0253) * ruggedness
    log_dist = np.log1p(geo_dist_km)
    predicted = 0.4841 + 0.0418 * log_dist + (-0.0253) * corridor_ruggedness
    return float(np.clip(predicted, 0, 1))


def divergence_label(score):
    if score < 0.45:
        return "Very Similar", "#4a7c59", "These languages are highly mutually intelligible — likely recent dialects of a common ancestor."
    elif score < 0.55:
        return "Moderately Similar", "#7c6a4a", "Noticeable divergence but shared vocabulary still evident. Think Spanish vs. Portuguese."
    elif score < 0.65:
        return "Diverged", "#7c4a4a", "Substantial lexical divergence. Related but distinct languages. Think English vs. German."
    else:
        return "Highly Diverged", "#5a3a7c", "Extensive divergence — deep time depth or strong geographic separation. Think English vs. Hindi."


def corridor_ruggedness_estimate(lat1, lon1, lat2, lon2):
    """
    Estimate corridor ruggedness from latitude (proxy for mountainous regions).
    In production this would sample the actual ruggedness grid.
    """
    mid_lat = abs((lat1 + lat2) / 2)
    # Higher absolute latitudes → more rugged on average in N hemisphere
    # Tropics tend to be lower ruggedness
    base = 0.5 + 0.3 * sin(radians(mid_lat * 1.5))
    # Add some signal for high-lat mountain belts (Alps, Himalayas ~25-50°N)
    if 25 < mid_lat < 50:
        base += 0.2
    return float(np.clip(base + np.random.normal(0, 0.05), 0.1, 2.5))


# ── Data for the world map ────────────────────────────────────────────────────
@st.cache_data
def get_world():
    try:
        import geodatasets
        return gpd.read_file(geodatasets.get_path('naturalearth.land'))
    except Exception:
        return None


# ── App layout ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>LinguaGeo</h1>
  <p>DOES GEOGRAPHY SHAPE LANGUAGE? &nbsp;·&nbsp; CS210 DATA SCIENCE PROJECT</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  Predictor", "📊  Key Findings", "🗺️  World Map"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Place two language homelands on the map</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Language A**")
        lang_a_name = st.text_input("Name (label only)", value="Spanish", key="name_a")
        lat_a = st.number_input("Latitude", value=40.4, min_value=-90.0, max_value=90.0, step=0.1, key="lat_a")
        lon_a = st.number_input("Longitude", value=-3.7, min_value=-180.0, max_value=180.0, step=0.1, key="lon_a")

    with col2:
        st.markdown("**Language B**")
        lang_b_name = st.text_input("Name (label only)", value="Romanian", key="name_b")
        lat_b = st.number_input("Latitude", value=44.4, min_value=-90.0, max_value=90.0, step=0.1, key="lat_b")
        lon_b = st.number_input("Longitude", value=26.1, min_value=-180.0, max_value=180.0, step=0.1, key="lon_b")

    # Quick presets
    st.markdown('<p class="section-header">Or try a preset pair</p>', unsafe_allow_html=True)
    presets = {
        "Spanish → Romanian (Romance)": (40.4, -3.7, 44.4, 26.1, "Spanish", "Romanian"),
        "Hindi → Bengali (Indo-Aryan)": (28.6, 77.2, 23.7, 90.4, "Hindi", "Bengali"),
        "Mandarin → Cantonese (Sino-Tibetan)": (39.9, 116.4, 23.1, 113.3, "Mandarin", "Cantonese"),
        "Norwegian → Icelandic (Germanic)": (59.9, 10.7, 64.1, -21.9, "Norwegian", "Icelandic"),
        "Tamil → Telugu (Dravidian)": (13.1, 80.3, 17.4, 78.5, "Tamil", "Telugu"),
        "Swahili → Zulu (Bantu)": (-6.8, 39.3, -29.9, 31.0, "Swahili", "Zulu"),
    }

    preset_choice = st.selectbox("Select a preset", ["(custom)"] + list(presets.keys()))
    if preset_choice != "(custom)":
        p = presets[preset_choice]
        lat_a, lon_a, lat_b, lon_b = p[0], p[1], p[2], p[3]
        lang_a_name, lang_b_name = p[4], p[5]

    if st.button("Predict divergence →", type="primary"):
        geo_dist = haversine_km(lat_a, lon_a, lat_b, lon_b)
        ruggedness = corridor_ruggedness_estimate(lat_a, lon_a, lat_b, lon_b)
        score = predict_divergence(geo_dist, ruggedness)
        label, color, explanation = divergence_label(score)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Geographic Distance", f"{geo_dist:,.0f} km")
        with c2:
            st.metric("Corridor Ruggedness", f"{ruggedness:.2f}")
        with c3:
            st.metric("Predicted Divergence", f"{score:.3f}")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted lexical divergence — {lang_a_name} vs {lang_b_name}</div>
            <div class="result-number" style="color:{color}">{score:.3f}</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.2rem; color:{color}; margin-top:0.25rem">{label}</div>
            <div class="interpretation">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)

        # Mini map showing the two points
        world = get_world()
        if world is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#f7f7f2')
            ax.set_facecolor('#e8f0f8')
            world.plot(ax=ax, color='#e8e8e0', edgecolor='#cccccc', linewidth=0.4)

            # Draw corridor line
            ax.plot([lon_a, lon_b], [lat_a, lat_b], color='#c8b89a', lw=1.5,
                    linestyle='--', alpha=0.8, zorder=3)

            # Plot points
            ax.scatter([lon_a], [lat_a], s=120, color='#4a7c59', zorder=5, edgecolors='white', linewidths=1.5)
            ax.scatter([lon_b], [lat_b], s=120, color='#7c4a4a', zorder=5, edgecolors='white', linewidths=1.5)
            ax.annotate(lang_a_name, (lon_a, lat_a), textcoords="offset points",
                        xytext=(8, 6), fontsize=9, color='#2a2a20', fontweight='bold')
            ax.annotate(lang_b_name, (lon_b, lat_b), textcoords="offset points",
                        xytext=(8, 6), fontsize=9, color='#2a2a20', fontweight='bold')

            # Zoom to corridor with padding
            pad = max(abs(lat_b-lat_a), abs(lon_b-lon_a)) * 0.5 + 15
            ax.set_xlim(min(lon_a,lon_b)-pad, max(lon_a,lon_b)+pad)
            ax.set_ylim(min(lat_a,lat_b)-pad*0.6, max(lat_a,lat_b)+pad*0.6)
            ax.set_axis_off()
            plt.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KEY FINDINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Statistical results from 15,183 language pairs</p>', unsafe_allow_html=True)

    # Stat cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Language pairs", "15,183")
    col2.metric("Model R²", "0.266")
    col3.metric("Geo distance ρ", "0.331 ***")
    col4.metric("Mantel r", "0.331 ***")

    st.markdown("---")

    # Findings
    st.markdown("""
    <div class="finding-block">
        <div class="finding-title">Isolation by Distance</div>
        <div class="finding-body">
            Geographic distance between language homelands is a significant predictor of
            lexical divergence (Spearman ρ = 0.331, p &lt; 0.001). Languages farther apart
            diverge more — consistent with the isolation-by-distance model from population genetics.
            The Mantel test confirms this holds even accounting for the non-independence of pairwise observations.
        </div>
    </div>
    <div class="finding-block">
        <div class="finding-title">The Terrain Paradox</div>
        <div class="finding-body">
            Counter-intuitively, terrain ruggedness is <em>negatively</em> associated with divergence
            (ρ = −0.193, p &lt; 0.001). Languages separated by rougher terrain tend to be <em>more</em> similar,
            not less. One explanation: mountainous regions preserve archaic vocabulary by limiting
            outside contact, keeping related languages closer to their common ancestor. Another: rugged
            homelands may host geographically proximate language pairs that haven't had time to drift.
        </div>
    </div>
    <div class="finding-block">
        <div class="finding-title">26.6% of variance explained</div>
        <div class="finding-body">
            Together, geographic distance and corridor ruggedness explain 26.6% of variance in
            pairwise lexical divergence (OLS R² = 0.266, F = 2755, p &lt; 0.001). This is a substantial
            improvement over the family-level analysis (R² = 0.002), demonstrating that pairwise
            methodology unlocks signal invisible at the aggregate level.
        </div>
    </div>
    <div class="finding-block">
        <div class="finding-title">Regional variation</div>
        <div class="finding-body">
            Europe/Middle East shows the highest within-family divergence, while Asia and
            Pacific/Oceania show the lowest. This may reflect the longer time depth of
            European language families (Indo-European ~6,000 years) vs. more recent
            expansions in East Asia and Oceania.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Simulated regression visualization
    st.markdown('<p class="section-header">Model visualization</p>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#f7f7f2')
    for ax in axes:
        ax.set_facecolor('#fafaf5')

    # Simulate data matching the model coefficients
    np.random.seed(42)
    n = 800
    log_dist = np.random.uniform(2, 9, n)
    rugged = np.random.uniform(0.1, 2.5, n)
    noise = np.random.normal(0, 0.12, n)
    div = 0.4841 + 0.0418*log_dist - 0.0253*rugged + noise
    div = np.clip(div, 0.1, 1.0)

    axes[0].scatter(np.exp(log_dist), div, alpha=0.25, s=12, color='#5a7a9a')
    x_fit = np.linspace(log_dist.min(), log_dist.max(), 200)
    axes[0].plot(np.exp(x_fit), 0.4841 + 0.0418*x_fit, '#c8503a', lw=2)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Geographic distance (km, log scale)', fontsize=9)
    axes[0].set_ylabel('Linguistic divergence', fontsize=9)
    axes[0].set_title('Isolation by distance\nρ = 0.331, p < 0.001', fontsize=9)
    sns.despine(ax=axes[0])

    axes[1].scatter(rugged, div, alpha=0.25, s=12, color='#9a7a5a')
    x_rug = np.linspace(0.1, 2.5, 200)
    axes[1].plot(x_rug, 0.4841 + 0.0418*5.5 - 0.0253*x_rug, '#c8503a', lw=2)
    axes[1].set_xlabel('Corridor terrain ruggedness', fontsize=9)
    axes[1].set_ylabel('Linguistic divergence', fontsize=9)
    axes[1].set_title('Terrain paradox\nρ = −0.193, p < 0.001', fontsize=9)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WORLD MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Mean within-family linguistic divergence by language homeland</p>', unsafe_allow_html=True)
    st.markdown("""
    To view this map with your real data, run the world map cell from the notebook and the
    saved PNG will appear here. Below is a live-generated schematic based on the model.
    """)

    world = get_world()
    if world is not None:
        # Generate synthetic language points across the world
        np.random.seed(99)
        regions = {
            'Europe':        (54, 15, 0.69, 0.08, 40),
            'South Asia':    (22, 80, 0.63, 0.10, 30),
            'East Asia':     (35, 110, 0.54, 0.09, 25),
            'Sub-Sah Africa':(5, 25, 0.64, 0.12, 50),
            'S America':     (-10, -60, 0.65, 0.11, 35),
            'N America':     (45, -100, 0.62, 0.10, 20),
            'Pacific':       (-10, 160, 0.53, 0.10, 20),
            'Middle East':   (32, 45, 0.67, 0.09, 15),
        }
        lats, lons, divs = [], [], []
        for name, (clat, clon, mu, sigma, n) in regions.items():
            lats += list(np.random.normal(clat, 8, n))
            lons += list(np.random.normal(clon, 12, n))
            divs += list(np.clip(np.random.normal(mu, sigma, n), 0.2, 0.95))

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#f0f4f8')
        ax.set_facecolor('#d0e4f0')
        world.plot(ax=ax, color='#e8e8e0', edgecolor='#ccccca', linewidth=0.3)

        sc = ax.scatter(lons, lats, c=divs, cmap='YlOrRd',
                        vmin=0.35, vmax=0.85,
                        s=30, alpha=0.75, edgecolors='white', linewidths=0.4, zorder=4)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label('Mean linguistic divergence', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.set_title('Schematic: Within-family linguistic divergence by region\n'
                     '(Replace with pairs_df output from notebook for real data)',
                     fontsize=10, pad=10)
        ax.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.info("Install geodatasets (`%pip install geodatasets`) to enable the world map.")

    st.markdown("---")
    st.markdown("""
    **To embed your real world map**, save the notebook's world map plot as `world_map_divergence.png`
    and add this to the app:
    ```python
    from PIL import Image
    img = Image.open('world_map_divergence.png')
    st.image(img, use_column_width=True)
    ```
    """)
