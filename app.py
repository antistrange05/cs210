import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2
from scipy import stats as sp_stats
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinguaGeo",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; font-size: 14px; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

.hero {
    padding: 2.5rem 0 1.8rem;
    border-bottom: 2px solid #1a1a16;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    line-height: 1.0;
    color: #1a1a16;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #999990;
    margin-top: 0.5rem;
}
.section-label {
    font-size: 0.63rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #bbbbbb;
    margin-bottom: 0.3rem;
    margin-top: 1.5rem;
}
.result-card {
    background: #f9f9f4;
    border: 1.5px solid #1a1a16;
    border-radius: 3px;
    padding: 1.6rem 2rem;
    margin: 1.2rem 0;
}
.result-score {
    font-family: 'Playfair Display', serif;
    font-size: 4.2rem;
    line-height: 1;
}
.result-tag {
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.result-meta {
    font-size: 0.75rem;
    color: #888880;
    margin-top: 0.4rem;
}
.result-formula {
    font-size: 0.78rem;
    color: #555550;
    border-top: 1px solid #e0e0d8;
    padding-top: 0.8rem;
    margin-top: 0.8rem;
    line-height: 1.8;
}
.finding-card {
    background: #fffef8;
    border-left: 3px solid #1a1a16;
    padding: 1.1rem 1.4rem;
    margin: 0.7rem 0;
    border-radius: 0 3px 3px 0;
}
.finding-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    margin-bottom: 0.3rem;
    color: #1a1a16;
}
.finding-body { font-size: 0.78rem; color: #555550; line-height: 1.75; }
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin: 0.8rem 0 1.8rem;
}
.stat-box {
    background: #1a1a16;
    color: #f5f5f0;
    padding: 1.1rem 1rem;
    border-radius: 3px;
    text-align: center;
}
.stat-val { font-family: 'Playfair Display', serif; font-size: 1.9rem; line-height: 1; }
.stat-lbl { font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase; color: #aaaaaa; margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_pairs():
    for path in ['Datasets/pairs.csv', 'pairs.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['log_geo_dist'] = np.log1p(df['geo_distance_km'])
            return df
    return None

@st.cache_data
def load_world():
    try:
        import geodatasets
        return gpd.read_file(geodatasets.get_path('naturalearth.land'))
    except Exception:
        return None

pairs_df = load_pairs()
world    = load_world()

if pairs_df is not None:
    families = sorted(pairs_df['family_id'].dropna().unique().tolist())
    # Assign regions by lat/lon for the filter in Tab 2
    def lat_lon_to_region(lat, lon):
        if lat > 60:                          return 'Northern Europe/Asia'
        elif lat > 35 and -15 < lon < 60:     return 'Europe/Middle East'
        elif lat > 20 and lon > 60:           return 'Asia'
        elif lat > 0  and lon < -30:          return 'Central/North America'
        elif lat < 0  and lon < -30:          return 'South America'
        elif lat < 0  and lon > 10:           return 'Sub-Saharan Africa'
        elif lat > 0  and -20 < lon < 55:     return 'Africa'
        elif lon > 100 or lon < -140:         return 'Pacific/Oceania'
        else:                                  return 'Other'

    pairs_df['region_a'] = pairs_df.apply(
        lambda r: lat_lon_to_region(r['lat_a'], r['lon_a']), axis=1)
    regions = ['All regions'] + sorted(pairs_df['region_a'].unique().tolist())


# ── Helpers ───────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def corridor_ruggedness_est(lat1, lon1, lat2, lon2):
    mid = abs((lat1 + lat2) / 2)
    base = 0.5 + 0.3 * sin(radians(mid * 1.5))
    if 25 < mid < 50: base += 0.2
    return float(np.clip(base, 0.1, 2.5))

def predict_div(geo_dist_km, ruggedness):
    return float(np.clip(
        0.4841 + 0.0418 * np.log1p(geo_dist_km) - 0.0253 * ruggedness, 0.1, 1.0))

def div_label(score):
    if score < 0.45:  return "Very Similar",        "#2d6a4f"
    elif score < 0.55: return "Moderately Similar", "#b07d2a"
    elif score < 0.65: return "Diverged",            "#c0392b"
    else:              return "Highly Diverged",     "#6c3483"

def nearest_real_pairs(geo_dist, div, n=5):
    df = pairs_df.copy()
    gs = pairs_df['geo_distance_km'].std() + 1e-9
    ds = pairs_df['ling_divergence'].std()  + 1e-9
    df['_d'] = abs(df['geo_distance_km'] - geo_dist)/gs + abs(df['ling_divergence'] - div)/ds
    top = df.nsmallest(n, '_d')[['lang_a','lang_b','family_id','ling_divergence','geo_distance_km']].copy()
    top.columns = ['Language A','Language B','Family','Divergence','Distance (km)']
    top['Divergence']    = top['Divergence'].round(3)
    top['Distance (km)'] = top['Distance (km)'].round(0).astype(int)
    return top

def draw_corridor_map(lat_a, lon_a, lat_b, lon_b, color):
    if world is None: return None
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor('#f9f9f4')
    ax.set_facecolor('#dce8f0')
    world.plot(ax=ax, color='#e8e8e0', edgecolor='#c8c8c0', linewidth=0.3)
    pad = max(abs(lat_b-lat_a), abs(lon_b-lon_a)) * 0.6 + 16
    ax.set_xlim(min(lon_a,lon_b)-pad, max(lon_a,lon_b)+pad)
    ax.set_ylim(min(lat_a,lat_b)-pad*0.45, max(lat_a,lat_b)+pad*0.45)
    ax.plot([lon_a,lon_b],[lat_a,lat_b], color='#c0392b', lw=1.5,
            linestyle='--', alpha=0.75, zorder=3)
    ax.scatter([lon_a,lon_b],[lat_a,lat_b], s=90, color=color,
               zorder=5, edgecolors='white', linewidths=1.4)
    ax.annotate("A",(lon_a,lat_a), textcoords="offset points",
                xytext=(6,5), fontsize=9, fontweight='bold', color='#1a1a16')
    ax.annotate("B",(lon_b,lat_b), textcoords="offset points",
                xytext=(6,5), fontsize=9, fontweight='bold', color='#1a1a16')
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    return fig


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">LinguaGeo</div>
  <div class="hero-sub">Does geography shape language? &nbsp;·&nbsp; CS210 Data Science &nbsp;·&nbsp; 15,183 real language pairs</div>
</div>
""", unsafe_allow_html=True)

if pairs_df is None:
    st.error("⚠️  pairs.csv not found. Ensure Datasets/pairs.csv is committed to the repo.")
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Predict Divergence",
    "🔍  Explore Pairs",
    "📊  Key Findings",
    "🧬  Family Deep Dive",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    Enter the geographic coordinates of two language homelands. The model predicts
    their lexical divergence using the OLS equation derived from 15,183 real language pairs,
    then anchors the result to the most similar pairs in the actual dataset.
    """)

    PRESETS = {
        "(enter coordinates manually)": None,
        "Spanish → Romanian  (Romance, Europe)":         (40.4, -3.7,  44.4,  26.1),
        "Hindi → Bengali  (Indo-Aryan, South Asia)":     (28.6,  77.2, 23.7,  90.4),
        "Mandarin → Cantonese  (Sino-Tibetan)":          (39.9, 116.4, 23.1, 113.3),
        "Norwegian → Icelandic  (Germanic)":             (59.9,  10.7, 64.1, -21.9),
        "Tamil → Telugu  (Dravidian)":                   (13.1,  80.3, 17.4,  78.5),
        "Swahili → Zulu  (Bantu, Africa)":               (-6.8,  39.3,-29.9,  31.0),
        "French → Italian  (Romance, nearby)":           (48.9,   2.3, 41.9,  12.5),
        "Turkish → Uzbek  (Turkic, Central Asia)":       (39.9,  32.8, 41.3,  69.3),
        "Welsh → Irish  (Celtic, nearby)":               (52.1,  -3.8, 53.4,  -8.2),
    }

    preset = st.selectbox("Quick preset pair", list(PRESETS.keys()))
    defaults = PRESETS[preset] if PRESETS[preset] else (40.4, -3.7, 44.4, 26.1)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Language A — homeland coordinates**")
        lat_a = st.number_input("Latitude A",  value=float(defaults[0]), min_value=-90.0,  max_value=90.0,  step=0.1, format="%.2f", key="lat_a")
        lon_a = st.number_input("Longitude A", value=float(defaults[1]), min_value=-180.0, max_value=180.0, step=0.1, format="%.2f", key="lon_a")
    with col2:
        st.markdown("**Language B — homeland coordinates**")
        lat_b = st.number_input("Latitude B",  value=float(defaults[2]), min_value=-90.0,  max_value=90.0,  step=0.1, format="%.2f", key="lat_b")
        lon_b = st.number_input("Longitude B", value=float(defaults[3]), min_value=-180.0, max_value=180.0, step=0.1, format="%.2f", key="lon_b")

    run = st.button("Predict divergence →", type="primary")

    if run:
        geo_dist   = haversine_km(lat_a, lon_a, lat_b, lon_b)
        ruggedness = corridor_ruggedness_est(lat_a, lon_a, lat_b, lon_b)
        score      = predict_div(geo_dist, ruggedness)
        label, color = div_label(score)

        # ── Result card ──
        st.markdown(f"""
        <div class="result-card">
          <div class="result-tag" style="color:{color}">{label}</div>
          <div class="result-score" style="color:{color}">{score:.3f}</div>
          <div class="result-meta">
            Geographic distance: <strong>{geo_dist:,.0f} km</strong>
            &nbsp;·&nbsp; Corridor ruggedness: <strong>{ruggedness:.2f}</strong>
          </div>
          <div class="result-formula">
            Model: <code>divergence = 0.4841 + 0.0418 × log(distance_km) − 0.0253 × ruggedness</code><br>
            R² = 0.266 &nbsp;·&nbsp; trained on 15,183 language pairs &nbsp;·&nbsp; Mantel r = 0.331 (p = 0.001)
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Nearest real pairs ──
        st.markdown('<p class="section-label">Most similar real pairs from the dataset</p>', unsafe_allow_html=True)
        st.caption(f"Your predicted divergence of **{score:.3f}** is most similar to these real pairs:")
        similar = nearest_real_pairs(geo_dist, score)
        st.dataframe(similar, use_container_width=True, hide_index=True)

        # ── Where this prediction sits on the global scatter ──
        st.markdown('<p class="section-label">Where your prediction sits in the data</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#f9f9f4')
        ax.set_facecolor('#f9f9f4')
        samp = pairs_df.sample(min(2500, len(pairs_df)), random_state=42)
        ax.scatter(samp['geo_distance_km'], samp['ling_divergence'],
                   alpha=0.2, s=10, color='#aaaaaa', label='Real pairs')
        # regression line
        sl, ic, *_ = sp_stats.linregress(np.log1p(pairs_df['geo_distance_km']), pairs_df['ling_divergence'])
        xf = np.linspace(pairs_df['geo_distance_km'].min(), pairs_df['geo_distance_km'].max(), 200)
        ax.plot(xf, ic + sl*np.log1p(xf), '#555555', lw=1.5, alpha=0.6)
        # user's prediction
        ax.scatter([geo_dist], [score], s=180, color=color, zorder=6,
                   edgecolors='white', linewidths=2, label='Your prediction')
        ax.set_xscale('log')
        ax.set_xlabel('Geographic distance (km, log scale)', fontsize=9)
        ax.set_ylabel('Linguistic divergence', fontsize=9)
        ax.set_title('Your prediction vs 15,183 real language pairs', fontsize=9)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Map ──
        st.markdown('<p class="section-label">Geographic corridor</p>', unsafe_allow_html=True)
        fig_map = draw_corridor_map(lat_a, lon_a, lat_b, lon_b, color)
        if fig_map:
            st.pyplot(fig_map, use_container_width=True)
            plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE PAIRS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    Filter, search, and visualize all 15,183 language pairs from the analysis.
    The scatter plot and correlation update live as you change filters.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        family_filter = st.selectbox("Family", ["All families"] + families, key="fam_filter")
    with col2:
        region_filter = st.selectbox("Region", regions, key="reg_filter")
    with col3:
        dist_range = st.slider("Distance (km)", 0, 12000, (0, 12000), step=200)
    with col4:
        div_range  = st.slider("Divergence",    0.0, 1.0,  (0.0, 1.0),  step=0.01)

    search = st.text_input("Search language name (partial match)", placeholder="e.g. Spanish, Hindi, Bantu...")

    filtered = pairs_df.copy()
    if family_filter != "All families":
        filtered = filtered[filtered['family_id'] == family_filter]
    if region_filter != "All regions":
        filtered = filtered[filtered['region_a'] == region_filter]
    filtered = filtered[
        (filtered['geo_distance_km'] >= dist_range[0]) &
        (filtered['geo_distance_km'] <= dist_range[1]) &
        (filtered['ling_divergence'] >= div_range[0]) &
        (filtered['ling_divergence'] <= div_range[1])
    ]
    if search.strip():
        s = search.strip().upper()
        filtered = filtered[
            filtered['lang_a'].str.contains(s, case=False, na=False) |
            filtered['lang_b'].str.contains(s, case=False, na=False)
        ]

    st.markdown(f"**{len(filtered):,} pairs** match your filters")

    if len(filtered) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        fig.patch.set_facecolor('#f9f9f4')
        for ax in axes: ax.set_facecolor('#f9f9f4')

        samp = filtered.sample(min(2500, len(filtered)), random_state=42)

        # Distance vs divergence
        axes[0].scatter(samp['geo_distance_km'], samp['ling_divergence'],
                        alpha=0.3, s=12, color='#2c5f8a')
        if len(filtered) > 10:
            sl, ic, *_ = sp_stats.linregress(np.log1p(filtered['geo_distance_km']),
                                              filtered['ling_divergence'])
            xf = np.linspace(filtered['geo_distance_km'].min(), filtered['geo_distance_km'].max(), 200)
            axes[0].plot(xf, ic + sl*np.log1p(xf), '#c0392b', lw=2)
            r, p = sp_stats.spearmanr(filtered['geo_distance_km'], filtered['ling_divergence'])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            axes[0].set_title(f'Distance vs Divergence  |  ρ = {r:.3f} {sig}  |  n = {len(filtered):,}', fontsize=9)
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Geographic distance (km, log)', fontsize=9)
        axes[0].set_ylabel('Linguistic divergence', fontsize=9)
        sns.despine(ax=axes[0])

        # Ruggedness vs divergence
        axes[1].scatter(samp['corridor_ruggedness'], samp['ling_divergence'],
                        alpha=0.3, s=12, color='#b07d2a')
        if len(filtered) > 10:
            sl2, ic2, *_ = sp_stats.linregress(filtered['corridor_ruggedness'],
                                                filtered['ling_divergence'])
            xr = np.linspace(filtered['corridor_ruggedness'].min(),
                              filtered['corridor_ruggedness'].max(), 200)
            axes[1].plot(xr, ic2 + sl2*xr, '#c0392b', lw=2)
            r2, p2 = sp_stats.spearmanr(filtered['corridor_ruggedness'], filtered['ling_divergence'])
            sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "ns"
            axes[1].set_title(f'Ruggedness vs Divergence  |  ρ = {r2:.3f} {sig2}', fontsize=9)
        axes[1].set_xlabel('Corridor ruggedness', fontsize=9)
        axes[1].set_ylabel('Linguistic divergence', fontsize=9)
        sns.despine(ax=axes[1])

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Table
    st.markdown('<p class="section-label">Pairs table — click a row to map it</p>', unsafe_allow_html=True)
    display = filtered[['lang_a','lang_b','family_id','ling_divergence',
                         'geo_distance_km','corridor_ruggedness']].rename(columns={
        'lang_a':'Language A','lang_b':'Language B','family_id':'Family',
        'ling_divergence':'Divergence','geo_distance_km':'Distance (km)',
        'corridor_ruggedness':'Ruggedness'
    }).round(3)

    selected_row = st.dataframe(
        display.head(300),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Map selected row
    if selected_row and selected_row.selection and selected_row.selection.rows:
        idx = selected_row.selection.rows[0]
        row = filtered.iloc[idx]
        st.markdown('<p class="section-label">Corridor map for selected pair</p>', unsafe_allow_html=True)
        _, color = div_label(row['ling_divergence'])
        fig_sel = draw_corridor_map(row['lat_a'], row['lon_a'], row['lat_b'], row['lon_b'], color)
        if fig_sel:
            st.caption(f"**{row['lang_a']}** → **{row['lang_b']}**  |  "
                       f"Divergence: {row['ling_divergence']:.3f}  |  "
                       f"Distance: {row['geo_distance_km']:,.0f} km  |  "
                       f"Ruggedness: {row['corridor_ruggedness']:.2f}")
            st.pyplot(fig_sel, use_container_width=True)
            plt.close()

    if len(filtered) > 300:
        st.caption(f"Showing first 300 of {len(filtered):,} rows. Narrow filters to see more.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — KEY FINDINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    Summary of the statistical analysis from the notebook.
    All figures are generated live from the loaded dataset.
    """)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-val">15,183</div><div class="stat-lbl">Language pairs</div></div>
        <div class="stat-box"><div class="stat-val">0.266</div><div class="stat-lbl">R² variance explained</div></div>
        <div class="stat-box"><div class="stat-val">0.331</div><div class="stat-lbl">Spearman ρ — distance</div></div>
        <div class="stat-box"><div class="stat-val">−0.193</div><div class="stat-lbl">Spearman ρ — ruggedness</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-card">
        <div class="finding-title">Isolation by Distance is Real</div>
        <div class="finding-body">
            Geographic distance between language homelands is a significant predictor of lexical divergence
            (Spearman ρ = 0.331, p &lt; 0.001). The Mantel test (r = 0.331, p = 0.001) confirms this holds
            even accounting for non-independence of pairwise observations. Languages farther apart diverge
            more — the isolation-by-distance model from population genetics, confirmed for lexical divergence.
        </div>
    </div>
    <div class="finding-card">
        <div class="finding-title">The Terrain Paradox</div>
        <div class="finding-body">
            Counter-intuitively, terrain ruggedness is <em>negatively</em> associated with divergence
            (ρ = −0.193, p &lt; 0.001). Languages separated by rougher terrain tend to be <em>more similar</em>,
            not less. Mountainous regions may preserve archaic vocabulary by limiting outside contact —
            keeping related languages closer to their common ancestor — or rugged homelands may simply
            host geographically proximate pairs that haven't had as much time to drift.
        </div>
    </div>
    <div class="finding-card">
        <div class="finding-title">26.6% of Variance Explained</div>
        <div class="finding-body">
            Geographic distance and corridor ruggedness together explain 26.6% of variance in pairwise
            lexical divergence (OLS R² = 0.266, F = 2755, p &lt; 0.001) — a dramatic improvement over
            the original family-level analysis (R² = 0.002), demonstrating that pairwise methodology
            unlocks signal invisible at the aggregate level.<br><br>
            <code>Divergence = 0.4841 + 0.0418 × log(distance_km) − 0.0253 × ruggedness</code>
        </div>
    </div>
    <div class="finding-card">
        <div class="finding-title">Why Pairwise?</div>
        <div class="finding-body">
            Averaging divergence across entire language families (the original approach) loses enormous
            within-family variation and yields R² = 0.002. Treating every pair of related languages as
            an observation gives 15,183 data points and a much more precise test. Linguistic divergence
            is measured as <strong>normalized Levenshtein distance</strong> across shared ASJP vocabulary —
            normalizing by word length so longer words don't dominate the score.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Live plots from real data
    st.markdown('<p class="section-label">Live regression plots from loaded data</p>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#f9f9f4')
    for ax in axes: ax.set_facecolor('#f9f9f4')

    samp = pairs_df.sample(min(3000, len(pairs_df)), random_state=42)
    r_geo, _ = sp_stats.spearmanr(pairs_df['geo_distance_km'], pairs_df['ling_divergence'])
    sl, ic, *_ = sp_stats.linregress(np.log1p(pairs_df['geo_distance_km']), pairs_df['ling_divergence'])
    xf = np.linspace(pairs_df['geo_distance_km'].min(), pairs_df['geo_distance_km'].max(), 200)
    axes[0].scatter(samp['geo_distance_km'], samp['ling_divergence'],
                    alpha=0.2, s=10, color='#2c5f8a')
    axes[0].plot(xf, ic + sl*np.log1p(xf), '#c0392b', lw=2)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Geographic distance (km, log scale)', fontsize=9)
    axes[0].set_ylabel('Linguistic divergence', fontsize=9)
    axes[0].set_title(f'Isolation by Distance  |  ρ = {r_geo:.3f}, p < 0.001', fontsize=9)
    sns.despine(ax=axes[0])

    r_rug, _ = sp_stats.spearmanr(pairs_df['corridor_ruggedness'], pairs_df['ling_divergence'])
    sl2, ic2, *_ = sp_stats.linregress(pairs_df['corridor_ruggedness'], pairs_df['ling_divergence'])
    xr = np.linspace(pairs_df['corridor_ruggedness'].min(), pairs_df['corridor_ruggedness'].max(), 200)
    axes[1].scatter(samp['corridor_ruggedness'], samp['ling_divergence'],
                    alpha=0.2, s=10, color='#b07d2a')
    axes[1].plot(xr, ic2 + sl2*xr, '#c0392b', lw=2)
    axes[1].set_xlabel('Corridor terrain ruggedness', fontsize=9)
    axes[1].set_ylabel('Linguistic divergence', fontsize=9)
    axes[1].set_title(f'Terrain Paradox  |  ρ = {r_rug:.3f}, p < 0.001', fontsize=9)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Saved notebook PNGs if present
    st.markdown('<p class="section-label">Plots from the notebook</p>', unsafe_allow_html=True)
    for fname, cap in [
        ('predictors_plot.png',      'Geographic predictors of linguistic divergence'),
        ('world_map_divergence.png', 'World map — within-family linguistic divergence'),
        ('divergence_by_region.png', 'Divergence by world region'),
    ]:
        if os.path.exists(fname):
            st.image(fname, caption=cap, use_column_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FAMILY DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    Select a language family to see how it compares to the global dataset —
    where its pairs sit on the distance/divergence curve, how its divergence
    distribution compares, and its most and least divergent pairs.
    """)

    selected_family = st.selectbox("Language family", families, key="fam_dive")
    fam_df = pairs_df[pairs_df['family_id'] == selected_family]

    if len(fam_df) == 0:
        st.warning("No pairs found for this family.")
    else:
        n_langs = len(set(fam_df['lang_a'].tolist() + fam_df['lang_b'].tolist()))
        global_mean = pairs_df['ling_divergence'].mean()
        fam_mean    = fam_df['ling_divergence'].mean()
        delta       = fam_mean - global_mean

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Languages in family", n_langs)
        col2.metric("Pairs", len(fam_df))
        col3.metric("Mean divergence", f"{fam_mean:.3f}",
                    delta=f"{delta:+.3f} vs global avg", delta_color="inverse")
        col4.metric("Mean distance", f"{fam_df['geo_distance_km'].mean():,.0f} km")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor('#f9f9f4')
        for ax in axes: ax.set_facecolor('#f9f9f4')

        # ── Plot 1: this family highlighted on global scatter ──
        axes[0].scatter(pairs_df['geo_distance_km'], pairs_df['ling_divergence'],
                        alpha=0.08, s=8, color='#cccccc', label='All families', zorder=1)
        axes[0].scatter(fam_df['geo_distance_km'], fam_df['ling_divergence'],
                        alpha=0.75, s=35, color='#c0392b', label=selected_family, zorder=4)
        # global trend line
        sl, ic, *_ = sp_stats.linregress(np.log1p(pairs_df['geo_distance_km']),
                                          pairs_df['ling_divergence'])
        xf = np.linspace(pairs_df['geo_distance_km'].min(), pairs_df['geo_distance_km'].max(), 200)
        axes[0].plot(xf, ic + sl*np.log1p(xf), '#555555', lw=1.5,
                     linestyle='--', alpha=0.6, label='Global trend')
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Distance (km, log)', fontsize=8)
        axes[0].set_ylabel('Divergence', fontsize=8)
        axes[0].set_title(f'{selected_family}\nvs all families', fontsize=9)
        axes[0].legend(fontsize=7)
        sns.despine(ax=axes[0])

        # ── Plot 2: divergence distribution ──
        axes[1].hist(pairs_df['ling_divergence'], bins=45, alpha=0.3,
                     color='#aaaaaa', label='All families', density=True)
        axes[1].hist(fam_df['ling_divergence'],
                     bins=max(5, min(30, len(fam_df)//2)),
                     alpha=0.85, color='#c0392b', label=selected_family, density=True)
        axes[1].axvline(fam_mean,    color='#c0392b', lw=2,   linestyle='--', label=f'Family mean ({fam_mean:.3f})')
        axes[1].axvline(global_mean, color='#888888', lw=1.5, linestyle='--', label=f'Global mean ({global_mean:.3f})')
        axes[1].set_xlabel('Divergence', fontsize=8)
        axes[1].set_ylabel('Density', fontsize=8)
        axes[1].set_title('Divergence distribution', fontsize=9)
        axes[1].legend(fontsize=7)
        sns.despine(ax=axes[1])

        # ── Plot 3: ruggedness vs divergence within family ──
        axes[2].scatter(fam_df['corridor_ruggedness'], fam_df['ling_divergence'],
                        alpha=0.65, s=35, color='#2c5f8a')
 try:
    if len(fam_df) > 5 and fam_df['corridor_ruggedness'].nunique() > 1:
        sl3, ic3, *_ = sp_stats.linregress(fam_df['corridor_ruggedness'], fam_df['ling_divergence'])
        xr = np.linspace(fam_df['corridor_ruggedness'].min(), fam_df['corridor_ruggedness'].max(), 100)
        axes[2].plot(xr, ic3 + sl3*xr, '#c0392b', lw=2)
        r3, p3 = sp_stats.spearmanr(fam_df['corridor_ruggedness'], fam_df['ling_divergence'])
        sig3 = "***" if p3 < 0.001 else "**" if p3 < 0.01 else "*" if p3 < 0.05 else "ns"
        axes[2].set_title(f'Ruggedness vs Divergence\nρ = {r3:.3f} {sig3}', fontsize=9)
    else:
        axes[2].set_title('Ruggedness vs Divergence\n(insufficient variation)', fontsize=9)
except ValueError:
    axes[2].set_title('Ruggedness vs Divergence\n(all ruggedness values identical)', fontsize=9)

axes[2].set_xlabel('Corridor ruggedness', fontsize=8)
axes[2].set_ylabel('Divergence', fontsize=8)
sns.despine(ax=axes[2])

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Most / least divergent pairs in this family ──
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-label">Most divergent pairs</p>', unsafe_allow_html=True)
            top5 = fam_df.nlargest(5, 'ling_divergence')[
                ['lang_a','lang_b','ling_divergence','geo_distance_km']].round(3)
            top5.columns = ['Language A','Language B','Divergence','Distance (km)']
            st.dataframe(top5, use_container_width=True, hide_index=True)
        with col2:
            st.markdown('<p class="section-label">Most similar pairs</p>', unsafe_allow_html=True)
            bot5 = fam_df.nsmallest(5, 'ling_divergence')[
                ['lang_a','lang_b','ling_divergence','geo_distance_km']].round(3)
            bot5.columns = ['Language A','Language B','Divergence','Distance (km)']
            st.dataframe(bot5, use_container_width=True, hide_index=True)

        # ── Map all languages in this family ──
        if world is not None and 'lat_a' in fam_df.columns:
            st.markdown('<p class="section-label">Family homelands map</p>', unsafe_allow_html=True)
            all_lats = pd.concat([fam_df['lat_a'], fam_df['lat_b']]).values
            all_lons = pd.concat([fam_df['lon_a'], fam_df['lon_b']]).values
            all_divs = pd.concat([fam_df['ling_divergence'], fam_df['ling_divergence']]).values

            fig_m, ax_m = plt.subplots(figsize=(12, 5))
            fig_m.patch.set_facecolor('#f9f9f4')
            ax_m.set_facecolor('#dce8f0')
            world.plot(ax=ax_m, color='#e8e8e0', edgecolor='#c8c8c0', linewidth=0.3)

            pad = 10
            ax_m.set_xlim(all_lons.min()-pad, all_lons.max()+pad)
            ax_m.set_ylim(all_lats.min()-pad, all_lats.max()+pad)

            sc = ax_m.scatter(all_lons, all_lats, c=all_divs,
                              cmap='YlOrRd', s=40, alpha=0.8,
                              edgecolors='white', linewidths=0.6, zorder=4)
            plt.colorbar(sc, ax=ax_m, shrink=0.6, label='Mean divergence')
            ax_m.set_title(f'{selected_family} — language homelands coloured by divergence', fontsize=9)
            ax_m.set_axis_off()
            plt.tight_layout()
            st.pyplot(fig_m, use_container_width=True)
            plt.close()
