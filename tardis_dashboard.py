import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

#configuratio 
st.set_page_config(
    page_title="TARDIS – Retards SNCF",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

#chargement du model et du fichier csv
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    df["year"] = df["year"].astype("Int64")
    df["month"] = df["month"].astype("Int64")
    return df


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


@st.cache_resource
def build_encoders(_dataframe):
    from sklearn.preprocessing import LabelEncoder

    features = [
        "Service",
        "Gare de départ",
        "Gare d'arrivée",
        "Durée moyenne du trajet",
        "Nombre de circulations prévues",
    ]
    target = "Retard moyen de tous les trains à l'arrivée"
    df_enc = _dataframe[features + [target]].copy()
    df_enc["Durée moyenne du trajet"] = pd.to_numeric(
        df_enc["Durée moyenne du trajet"], errors="coerce"
    )
    df_enc["Nombre de circulations prévues"] = pd.to_numeric(
        df_enc["Nombre de circulations prévues"], errors="coerce"
    )
    df_enc[target] = pd.to_numeric(df_enc[target], errors="coerce")
    df_enc = df_enc.dropna()
    le_s = LabelEncoder().fit(df_enc["Service"])
    le_d = LabelEncoder().fit(df_enc["Gare de départ"])
    le_a = LabelEncoder().fit(df_enc["Gare d'arrivée"])
    return le_s, le_d, le_a


df = load_data()
model = load_model()
le_service, le_dep, le_arr = build_encoders(df)

TARGET = "Retard moyen de tous les trains à l'arrivée"
DELAY_COL_DEP = "Retard moyen de tous les trains au départ"


#filtre
years = sorted(df["year"].dropna().unique())
selected_years = st.sidebar.multiselect(
    "Années",
    options=[int(y) for y in years],
    default=[int(y) for y in years],
)

services = ["Tous"] + sorted(df["Service"].dropna().unique().tolist())
selected_service = st.sidebar.selectbox("Type de service", services)

gares_dep = sorted(df["Gare de départ"].dropna().unique().tolist())
selected_gare = st.sidebar.selectbox(
    "Gare de départ (vue détaillée)", ["Toutes"] + gares_dep
)

# filtrage
mask = df["year"].isin(selected_years)
if selected_service != "Tous":
    mask &= df["Service"] == selected_service
dff = df[mask].copy()

if selected_gare != "Toutes":
    dff_gare = dff[dff["Gare de départ"] == selected_gare]
else:
    dff_gare = dff

# titre principal
st.title("🚄 TARDIS – Tableau de bord des retards SNCF")
st.caption(
    "Analyse des données historiques de ponctualité et prédiction des retards TGV"
)
st.markdown("---")

# indicateur clé
st.subheader("📊 Indicateurs clés")

total_circulations = pd.to_numeric(dff["Nombre de circulations prévues"], errors="coerce").sum()
total_annules = pd.to_numeric(dff["Nombre de trains annulés"], errors="coerce").fillna(0).sum()
retard_moyen = pd.to_numeric(dff[TARGET], errors="coerce").dropna().mean()
ponctualite = 100 * (1 - total_annules / total_circulations) if total_circulations else 0
retard_dep_moyen = pd.to_numeric(dff[DELAY_COL_DEP], errors="coerce").dropna().mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🚆 Trains prévus", f"{int(total_circulations):,}".replace(",", " "))
c2.metric("❌ Trains annulés", f"{int(total_annules):,}".replace(",", " "))
c3.metric("⏱️ Retard moyen arrivée", f"{retard_moyen:.1f} min")
c4.metric("⏳ Retard moyen départ", f"{retard_dep_moyen:.1f} min")
c5.metric("✅ Taux de ponctualité", f"{ponctualite:.1f} %")

st.markdown("---")

# visualisation
st.subheader("📈 Analyse des retards")

tab1, tab2, tab3 = st.tabs(
    ["Distribution des retards", "Évolution temporelle", "Comparaison par gare"]
)

#distribution
with tab1:
    col_l, col_r = st.columns(2)

    with col_l:
        delay_data = pd.to_numeric(dff[TARGET], errors='coerce').dropna()
        delay_data = delay_data[(delay_data >= 0) & (delay_data <= 60)]
        fig_hist = px.histogram(
            delay_data,
            nbins=40,
            title="Distribution du retard moyen à l'arrivée (0-60 min)",
            labels={"value": "Retard (min)", "count": "Fréquence"},
            color_discrete_sequence=["#3B82F6"],
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(
            "La distribution montre que la majorité des trains enregistrent un retard "
            "moyen compris entre 2 et 10 minutes à l'arrivée."
        )

    with col_r:
        cause_cols = {
            "Externes": "Prct retard pour causes externes",
            "Infrastructure": "Prct retard pour cause infrastructure",
            "Gestion trafic": "Prct retard pour cause gestion trafic",
            "Matériel roulant": "Prct retard pour cause matériel roulant",
            "Gestion gare": "Prct retard pour cause gestion en gare et réutilisation de matériel",
            "Voyageurs": "Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)",
        }
        moyennes = {}
        for label, col in cause_cols.items():
            if col in dff.columns:
                moyennes[label] = pd.to_numeric(dff[col], errors="coerce").dropna().mean()

        fig_pie = px.pie(
            values=list(moyennes.values()),
            names=list(moyennes.keys()),
            title="Répartition des causes de retard (%)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption(
            "Les causes externes (météo, incidents réseau) et l'infrastructure "
            "constituent les principales sources de retard."
        )

#evolution temporelle
with tab2:
    dff_plot = dff.copy()
    dff_plot[TARGET] = pd.to_numeric(dff_plot[TARGET], errors='coerce')
    monthly = (
        dff_plot.groupby(["year", "month"])[TARGET]
        .mean()
        .reset_index()
    )
    monthly["periode"] = (
        monthly["year"].astype(str)
        + "-"
        + monthly["month"].astype(str).str.zfill(2)
    )
    monthly = monthly.sort_values("periode")

    fig_line = px.line(
        monthly,
        x="periode",
        y=TARGET,
        title="Évolution mensuelle du retard moyen à l'arrivée",
        labels={TARGET: "Retard moyen (min)", "periode": "Période"},
        color_discrete_sequence=["#EF4444"],
        markers=True,
    )
    fig_line.update_xaxes(tickangle=45)
    st.plotly_chart(fig_line, use_container_width=True)
    st.caption(
        "On observe des pics de retards en hiver (conditions météo) et des améliorations "
        "progressives selon les années. La période COVID (2020) montre une réduction des "
        "retards due à la baisse du trafic."
    )

    # retards > 15 30 60 min
    seuils_15 = pd.to_numeric(dff['Nombre trains en retard > 15min'], errors='coerce').sum()
    seuils_30 = pd.to_numeric(dff['Nombre trains en retard > 30min'], errors='coerce').sum()
    seuils_60 = pd.to_numeric(dff['Nombre trains en retard > 60min'], errors='coerce').sum()
    fig_bar_seuils = go.Figure(
        go.Bar(
            x=["&gt; 15 min", "&gt; 30 min", "&gt; 60 min"],
            y=[seuils_15, seuils_30, seuils_60],
            marker_color=["#FBBF24", "#F97316", "#EF4444"],
        )
    )
    fig_bar_seuils.update_layout(
        title="Nombre total de trains dépassant les seuils de retard",
        xaxis_title="Seuil",
        yaxis_title="Nombre de trains",
    )
    st.plotly_chart(fig_bar_seuils, use_container_width=True)
    st.caption(
        "Les retards supérieurs à 60 minutes restent rares mais significatifs "
        "pour l'expérience voyageur."
    )

#comparaison par gare
with tab3:
    top_n = st.slider("Nombre de gares à afficher", 5, 20, 10)
    dff_gare2 = dff.copy()
    dff_gare2[TARGET] = pd.to_numeric(dff_gare2[TARGET], errors='coerce')
    gare_stats = (
        dff_gare2.groupby("Gare de départ")[TARGET]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    gare_stats.columns = ["Gare", "Retard moyen (min)"]

    fig_bar = px.bar(
        gare_stats,
        x="Retard moyen (min)",
        y="Gare",
        orientation="h",
        title=f"Top {top_n} gares par retard moyen à l'arrivée",
        color="Retard moyen (min)",
        color_continuous_scale="Reds",
    )
    fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        "Certaines gares internationales et lignes transfrontalières affichent "
        "des retards plus élevés en raison de la complexité de la coordination "
        "entre réseaux ferroviaires."
    )

st.markdown("---")

# prediction
st.subheader("🔮 Prédiction de retard")
st.markdown(
    "Renseignez les caractéristiques du trajet pour obtenir une estimation "
    "du retard moyen prévu à l'arrivée."
)

all_gares_dep = sorted(df["Gare de départ"].dropna().unique().tolist())
all_gares_arr = sorted(df["Gare d'arrivée"].dropna().unique().tolist())

col1, col2 = st.columns(2)

with col1:
    pred_service = st.selectbox(
        "Type de service", ["National", "International"], key="pred_service"
    )
    pred_gare_dep = st.selectbox(
        "Gare de départ", all_gares_dep, key="pred_dep"
    )
    pred_gare_arr = st.selectbox(
        "Gare d'arrivée", all_gares_arr, key="pred_arr"
    )

with col2:
    pred_duree = st.slider(
        "Durée moyenne du trajet (min)", 30, 300, 120, step=10
    )
    pred_circulations = st.slider(
        "Nombre de circulations prévues", 10, 600, 200, step=10
    )

if st.button("🚀 Prédire le retard", type="primary"):
    try:
        # Encoder les variables catégorielles 
        if pred_service not in le_service.classes_:
            st.error(f"Service inconnu : {pred_service}")
            st.stop()
        if pred_gare_dep not in le_dep.classes_:
            st.error(f"Gare de départ inconnue du modèle : {pred_gare_dep}")
            st.stop()
        if pred_gare_arr not in le_arr.classes_:
            st.error(f"Gare d'arrivée inconnue du modèle : {pred_gare_arr}")
            st.stop()

        service_enc = int(le_service.transform([pred_service])[0])
        dep_enc = int(le_dep.transform([pred_gare_dep])[0])
        arr_enc = int(le_arr.transform([pred_gare_arr])[0])

        import numpy as np
        input_arr = np.array([[service_enc, dep_enc, arr_enc, float(pred_duree), float(pred_circulations)]])
        prediction = model.predict(input_arr)[0]
        prediction = max(0.0, prediction)

        if prediction < 3:
            couleur = "🟢"
            niveau = "Faible"
        elif prediction < 8:
            couleur = "🟡"
            niveau = "Modéré"
        else:
            couleur = "🔴"
            niveau = "Élevé"

        res1, res2, res3 = st.columns(3)
        res1.metric("⏱️ Retard prédit", f"{prediction:.1f} min")
        res2.metric("Niveau de risque", f"{couleur} {niveau}")
        baseline = pd.to_numeric(df[TARGET], errors='coerce').dropna().mean()
        diff = prediction - baseline
        res3.metric(
            "vs. Moyenne nationale",
            f"{prediction:.1f} min",
            delta=f"{diff:+.1f} min",
            delta_color="inverse",
        )

        st.success(
            f"Pour le trajet **{pred_gare_dep} → {pred_gare_arr}** "
            f"(service {pred_service}, {pred_duree} min de trajet) : "
            f"retard estimé de **{prediction:.1f} minutes**."
        )
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

