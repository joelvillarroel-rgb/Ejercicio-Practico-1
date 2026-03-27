import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="M/M/1 Emergencias Médicas", layout="wide")

# ----------------------
# Funciones teóricas M/M/1
# ----------------------
def mm1_metrics(lmbda, mu):
    if lmbda >= mu:
        return None
    rho = lmbda / mu
    P0 = 1 - rho
    Lq = (rho**2) / (1 - rho)
    L = rho / (1 - rho)
    Wq = Lq / lmbda
    W = L / lmbda
    return rho, P0, Lq, L, Wq, W


def prob_n_geq_k(rho, k):
    return rho**k


def prob_wait_more_than_t(mu, rho, t):
    return rho * np.exp(-(mu - rho * mu) * t)

# ----------------------
# Simulación Monte Carlo
# ----------------------
def simulate_mm1(lmbda, mu, n=1000, seed=42):
    np.random.seed(seed)
    arrivals = np.cumsum(np.random.exponential(1/lmbda, n))
    service = np.random.exponential(1/mu, n)

    start_service = np.zeros(n)
    finish_service = np.zeros(n)
    wait = np.zeros(n)

    for i in range(n):
        if i == 0:
            start_service[i] = arrivals[i]
        else:
            start_service[i] = max(arrivals[i], finish_service[i-1])
        wait[i] = start_service[i] - arrivals[i]
        finish_service[i] = start_service[i] + service[i]

    df = pd.DataFrame({
        "Llegada": arrivals,
        "Servicio": service,
        "Inicio": start_service,
        "Fin": finish_service,
        "Espera": wait
    })
    return df

# ----------------------
# Interfaz
# ----------------------
st.title("📞 Simulación Centro de Emergencias (M/M/1)")

col1, col2 = st.columns(2)

with col1:
    lmbda = st.number_input("Tasa de llegada λ (llamadas/hora)", value=18.0)
    mu = st.number_input("Tasa de servicio μ (llamadas/hora)", value=24.0)

with col2:
    t = st.number_input("Tiempo umbral (horas)", value=8/60)
    k = st.number_input("Número mínimo de llamadas (k)", value=4)

# ----------------------
# Resultados
# ----------------------
res = mm1_metrics(lmbda, mu)

if res is None:
    st.error("⚠️ Sistema inestable (λ ≥ μ)")
else:
    rho, P0, Lq, L, Wq, W = res

    st.subheader("📊 Métricas Teóricas")

    metrics_df = pd.DataFrame({
        "Métrica": ["ρ", "P0", "Lq", "L", "Wq (horas)", "W (horas)"],
        "Valor": [rho, P0, Lq, L, Wq, W]
    })

    st.dataframe(metrics_df)

    # Probabilidades
    st.subheader("📈 Probabilidades")
    st.write(f"P(N ≥ {k}) = {prob_n_geq_k(rho, k):.4f}")
    st.write(f"P(Wq > {t*60:.1f} min) = {prob_wait_more_than_t(mu, rho, t):.4f}")

    # ----------------------
    # Gráficos
    # ----------------------
    st.subheader("📉 Visualizaciones")

    fig1, ax1 = plt.subplots()
    ax1.bar(["L", "Lq", "W", "Wq"], [L, Lq, W, Wq])
    st.pyplot(fig1)

    # Distribución de estados
    n_vals = np.arange(0, 10)
    pn = (1 - rho) * (rho ** n_vals)

    fig2, ax2 = plt.subplots()
    ax2.plot(n_vals, pn, marker='o')
    ax2.set_title("Probabilidad de estados Pn")
    st.pyplot(fig2)

    # ----------------------
    # Simulación
    # ----------------------
    st.subheader("🔬 Simulación Monte Carlo")

    df_sim = simulate_mm1(lmbda, mu)

    st.write(df_sim.head())

    st.write("Promedio espera simulada:", df_sim["Espera"].mean())

    fig3, ax3 = plt.subplots()
    ax3.hist(df_sim["Espera"], bins=30)
    ax3.set_title("Distribución de tiempos de espera")
    st.pyplot(fig3)

    # ----------------------
    # Interpretación automática
    # ----------------------
    st.subheader("🧠 Interpretación")

    if rho < 0.7:
        st.success("Sistema eficiente")
    elif rho < 0.9:
        st.warning("Sistema con riesgo moderado")
    else:
        st.error("Sistema crítico, alta congestión")

    st.write("Recomendación: evaluar incremento de capacidad o múltiples operadores.")
