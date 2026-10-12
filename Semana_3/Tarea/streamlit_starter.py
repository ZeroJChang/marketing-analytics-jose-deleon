"""
Laboratorio Final — Starbucks Rewards
PUNTO DE PARTIDA para su propia app de Streamlit. Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus
resultados del laboratorio — el Qini de su T-learner y la tabla de políticas —
con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Starbucks Rewards | Uplift", page_icon="☕", layout="wide")
GREEN, ORANGE, GRAY = "#00704A", "#C4512F", "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]

@st.cache_data
def load_data():
    ruta = Path(__file__).resolve().parent / "data" / "uplift_campaign.csv"
    return pd.read_csv(ruta)

@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float = 0.30, seed: int = 42):
    train, test = train_test_split(df, test_size=test_size, random_state=seed,
                                   stratify=df["treatment_group"])
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    modelo_tratado = LogisticRegression(max_iter=1000).fit(train_t[FEATURES], train_t["responded_60d"])
    modelo_control = LogisticRegression(max_iter=1000).fit(train_c[FEATURES], train_c["responded_60d"])
    modelo_riesgo = LogisticRegression(max_iter=1000).fit(train[FEATURES], 1 - train["responded_60d"])
    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    test["prob_churn"] = modelo_riesgo.predict_proba(X_test)[:, 1]
    return train, test

def valor_de_la_politica(test_df: pd.DataFrame, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)

def curva_qini(test_df: pd.DataFrame, score_col: str, steps: int = 40):
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values
    acumulado_t = np.cumsum(np.where(es_tratado, respondio, 0))
    acumulado_c = np.cumsum(np.where(~es_tratado, respondio, 0))
    nt = np.cumsum(es_tratado)
    nc = np.cumsum(~es_tratado)
    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        tasa_t = acumulado_t[i-1] / nt[i-1] if nt[i-1] > 0 else 0
        tasa_c = acumulado_c[i-1] / nc[i-1] if nc[i-1] > 0 else 0
        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)
    return np.array(xs), np.array(ys)

def coeficiente_qini(xs, ys, n):
    area_modelo = np.trapz(ys, xs)
    area_azar = np.trapz(np.linspace(0, ys[-1], len(xs)), xs)
    return (area_modelo - area_azar) / n

st.title("☕ Starbucks Rewards — Decisiones basadas en uplift")
st.caption("¿A qué porcentaje de clientes conviene enviar un cupón para obtener el mayor valor incremental?")
try:
    df = load_data()
except FileNotFoundError:
    st.error("No se encontró data/uplift_campaign.csv. Crea la carpeta data junto a este archivo y agrega allí el CSV.")
    st.stop()
with st.sidebar:
    st.header("Parámetros de campaña")
    pct_contactar = st.slider("Porcentaje de clientes a contactar", min_value=10, max_value=100, value=20, step=10, format="%d%%")
    st.caption("Usamos el 70% para entrenar y el 30% para evaluar. Semilla fija: 42.")

train, test = entrenar_t_learner(df)
xs_uplift, ys_uplift = curva_qini(test, "uplift_estimado")
xs_riesgo, ys_riesgo = curva_qini(test, "prob_churn")
qini_uplift = coeficiente_qini(xs_uplift, ys_uplift, len(test))
qini_riesgo = coeficiente_qini(xs_riesgo, ys_riesgo, len(test))

resultados = []
for pct in range(10, 101, 10):
    n_contactar = int(len(test) * pct / 100)
    indices = test.sort_values("uplift_estimado", ascending=False).head(n_contactar).index
    valor_cliente, n = valor_de_la_politica(test, indices)
    resultados.append({"% a contactar": pct, "Valor por cliente ($)": round(valor_cliente, 2),
                       "Clientes seleccionados": n, "Valor total ($)": round(valor_cliente * n, 2)})
tabla = pd.DataFrame(resultados)
seleccion = tabla.loc[tabla["% a contactar"] == pct_contactar].iloc[0]
optimo = tabla.loc[tabla["Valor total ($)"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Qini del T-learner", f"{qini_uplift:+.3f}")
c2.metric("Qini por riesgo", f"{qini_riesgo:+.3f}")
c3.metric("Clientes seleccionados", f"{int(seleccion['Clientes seleccionados']):,}")
c4.metric("Valor incremental total", f"${seleccion['Valor total ($)']:+,.2f}")

st.subheader("Curva Qini: uplift vs. selección por riesgo")
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(xs_uplift * 100, ys_uplift, color=GREEN, linewidth=2.8, label=f"T-learner (Qini {qini_uplift:+.3f})")
ax.plot(xs_riesgo * 100, ys_riesgo, color=ORANGE, linewidth=2.3, label=f"Riesgo/churn (Qini {qini_riesgo:+.3f})")
ax.plot([0,100],[0,ys_uplift[-1]],color=GRAY,linestyle="--",label="Selección aleatoria")
ax.axvline(pct_contactar,color=GREEN,linestyle=":",alpha=.7,label=f"Porcentaje elegido: {pct_contactar}%")
ax.set(xlabel="Porcentaje de la base de prueba ordenada (%)", ylabel="Compras incrementales estimadas (acumuladas)")
ax.grid(alpha=.18)
ax.legend(loc="best")
st.pyplot(fig, clear_figure=True)

st.subheader("Tabla de políticas comerciales")
st.dataframe(tabla.style.apply(lambda row: ["background-color: #dceee3" if int(row['% a contactar']) == pct_contactar else "" for _ in row], axis=1),hide_index=True,use_container_width=True)
st.caption("El valor por cliente compara la utilidad neta promedio de tratados con el margen promedio de controles en el segmento seleccionado. El valor total se calcula multiplicando esa diferencia por el número seleccionado; es una estimación del experimento, no facturación observada.")

st.success(f"En estos datos, el mayor valor total se obtiene contactando al {int(optimo['% a contactar'])}% de la base: ${optimo['Valor total ($)']:,.2f} estimados. El porcentaje actual del slider es {pct_contactar}%.")
st.download_button("Descargar tabla de políticas (CSV)", tabla.to_csv(index=False).encode("utf-8-sig"), file_name="tabla_politicas.csv", mime="text/csv")
