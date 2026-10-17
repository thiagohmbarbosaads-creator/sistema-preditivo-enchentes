import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Sistema Preditivo de Enchentes | FSA",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Sistema Preditivo de Risco de Enchentes")
st.caption("Projeto Integrador de IA e Machine Learning — Fundação Santo André (2026)")
st.write("**Desenvolvido por:** Evellyn Gabrielle & Thiago Henrique de Melo Barbosa")

st.markdown("---")

# 1. Carregar Dados e Treinar o Modelo
@st.cache_resource
def carregar_e_treinar():
    df = pd.read_csv('flood.csv')
    df['Enchente'] = (df['FloodProbability'] >= 0.5).astype(int)
    
    X = df.drop(columns=['FloodProbability', 'Enchente'])
    y = df['Enchente']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    modelo = DecisionTreeClassifier(max_depth=6, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo, X.columns

modelo_arvore, colunas = carregar_e_treinar()

# 2. Painel Lateral para Inserir Medidas Reais
st.sidebar.header("⚙️ Parâmetros da Região (Medidas Reais)")

chuva_mm = st.sidebar.number_input("Volume de Chuva (mm)", min_value=0.0, max_value=300.0, value=50.0, step=5.0)
desmatamento_pct = st.sidebar.slider("Desmatamento na Região (%)", 0.0, 100.0, 30.0)
bueiros_entupidos_pct = st.sidebar.slider("Falha / Entupimento da Drenagem (%)", 0.0, 100.0, 20.0)
nivel_rio_m = st.sidebar.number_input("Elevação do Nível do Rio (m)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

# Botão para simular o caso do Rio Grande do Sul (Maio/2024)
if st.sidebar.button("🚨 Carregar Dados do RS (Maio/2024)"):
    chuva_mm = 300.0
    desmatamento_pct = 65.0
    bueiros_entupidos_pct = 85.0
    nivel_rio_m = 5.35

# 3. Conversão Proporcional de Escala (Mundo Real -> 0 a 10)
nota_chuva = min(10.0, np.interp(chuva_mm, [0, 150], [0, 10]))
nota_desmatamento = np.interp(desmatamento_pct, [0, 100], [0, 10])
nota_drenagem = np.interp(bueiros_entupidos_pct, [0, 100], [0, 10])
nota_rio = np.interp(nivel_rio_m, [0, 8], [0, 10])

media_fatores = float(np.mean([nota_chuva, nota_desmatamento, nota_drenagem, nota_rio]))

dados_convertidos = {col: media_fatores for col in colunas}
dados_convertidos['MonsoonIntensity'] = nota_chuva
dados_convertidos['Deforestation'] = nota_desmatamento
dados_convertidos['DrainageSystems'] = nota_drenagem
dados_convertidos['RiverManagement'] = nota_rio

df_usuario = pd.DataFrame([dados_convertidos])

# 4. Exibição dos Resultados
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Conversão Automática para a IA")
    st.write(f"• **Chuva ({chuva_mm} mm):** Nota {nota_chuva:.1f} / 10")
    st.write(f"• **Desmatamento ({desmatamento_pct}%):** Nota {nota_desmatamento:.1f} / 10")
    st.write(f"• **Falha na Drenagem ({bueiros_entupidos_pct}%):** Nota {nota_drenagem:.1f} / 10")
    st.write(f"• **Elevação do Rio ({nivel_rio_m} m):** Nota {nota_rio:.1f} / 10")

with col2:
    st.subheader("🤖 Diagnóstico do Modelo")
    previsao = modelo_arvore.predict(df_usuario)[0]
    probabilidade = modelo_arvore.predict_proba(df_usuario)[0][1]
    
    if previsao == 1:
        st.error("### 🚨 ALERTA MÁXIMO: ALTO RISCO DE ENCHENTE!")
        st.metric("Probabilidade do Desastre", f"{probabilidade * 100:.1f}%")
        st.warning("**Ações de Emergência:** Acionar sirenes de emergência, iniciar evacuação preventiva e bloquear vias de risco.")
    else:
        st.success("### ✅ SITUAÇÃO ESTÁVEL: BAIXO RISCO DE ENCHENTE")
        st.metric("Probabilidade do Desastre", f"{probabilidade * 100:.1f}%")
        st.info("**Ações Recomendadas:** Manter monitoramento preventivo de rotina.")