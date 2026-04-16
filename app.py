import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd

# ── Credenciales ──────────────────────────────────────────────────────────────
USER     = st.secrets["DB_USER"]
PASSWORD = st.secrets["DB_PASSWORD"]
HOST     = st.secrets["DB_HOST"]
PORT     = st.secrets["DB_PORT"]
DBNAME   = st.secrets["DB_NAME"]

# ── Helpers de BD ─────────────────────────────────────────────────────────────
def get_connection():
    return psycopg2.connect(
        user=USER, password=PASSWORD,
        host=HOST, port=PORT, dbname=DBNAME
    )

def insertar_prediccion(l_p, l_s, a_s, a_p, prediccion):
    sql = """
        INSERT INTO tb_iris (l_p, l_s, a_s, a_p, prediccion)
        VALUES (%s, %s, %s, %s, %s)
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, (l_p, l_s, a_s, a_p, prediccion))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error al guardar: {e}")
        return False

def obtener_historico():
    sql = """
        SELECT id, created_at, l_p, l_s, a_s, a_p, prediccion
        FROM tb_iris
        ORDER BY created_at DESC
    """
    try:
        conn = get_connection()
        cur  = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Error al obtener historial: {e}")
        return []

def badge_especie(nombre):
    """Badge con color por especie."""
    estilos = {
        "setosa":     "background:#E1F5EE;color:#085041;",
        "versicolor": "background:#E6F1FB;color:#0C447C;",
        "virginica":  "background:#EEEDFE;color:#3C3489;",
    }
    key = nombre.lower().replace("iris ", "").strip()
    estilo = estilos.get(key, "background:#F1EFE8;color:#444441;")
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:999px;'
        f'font-size:12px;font-weight:500;{estilo}">{nombre}</span>'
    )

def renderizar_tabla_historico(rows):
    """Genera la tabla HTML estilizada del histórico."""
    total = len(rows)
    filas_html = ""
    for row in rows:
        fecha = pd.to_datetime(row["created_at"]).strftime("%Y-%m-%d %H:%M")
        b = "border-bottom:0.5px solid rgba(0,0,0,0.07);"
        td = f"padding:10px 16px;{b}white-space:nowrap;"
        filas_html += f"""
        <tr>
          <td style="{td}color:#999;font-size:12px;">{row['id']}</td>
          <td style="{td}color:#999;font-size:12px;">{fecha}</td>
          <td style="{td}">{row['l_s']}</td>
          <td style="{td}">{row['a_s']}</td>
          <td style="{td}">{row['l_p']}</td>
          <td style="{td}">{row['a_p']}</td>
          <td style="padding:10px 16px;{b}">{badge_especie(row['prediccion'])}</td>
        </tr>
        """

    th_style = (
        "padding:10px 16px;text-align:left;font-weight:500;font-size:11px;"
        "color:#888;letter-spacing:0.04em;text-transform:uppercase;"
        "border-bottom:0.5px solid rgba(0,0,0,0.1);white-space:nowrap;"
    )
    cabeceras = ["#", "Fecha", "Long. sépalo", "Ancho sépalo",
                 "Long. pétalo", "Ancho pétalo", "Especie predicha"]
    ths = "".join(f'<th style="{th_style}">{c}</th>' for c in cabeceras)

    html = f"""
    <style>
      .hist-wrap tr:last-child td {{ border-bottom: none !important; }}
      .hist-wrap tbody tr:hover td {{ background: rgba(0,0,0,0.025); }}
    </style>

    <div class="hist-wrap" style="margin-top:0.5rem;">
      <p style="font-size:18px;font-weight:500;margin:0 0 4px;">
        Historial de predicciones
      </p>
      <p style="font-size:13px;color:#888;margin:0 0 1rem;">
        Registros ordenados por fecha de creación, del más reciente al más antiguo
      </p>

      <div style="border:0.5px solid rgba(0,0,0,0.12);border-radius:12px;overflow:hidden;">
        <div style="overflow-x:auto;">
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
              <tr style="background:#f5f5f3;">{ths}</tr>
            </thead>
            <tbody>{filas_html}</tbody>
          </table>
        </div>

        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:9px 16px;background:#f5f5f3;
                    border-top:0.5px solid rgba(0,0,0,0.1);">
          <span style="font-size:12px;color:#888;">
            {total} registro{"s" if total != 1 else ""} en total
          </span>
        </div>
      </div>
    </div>
    """
    return html

# ── Config página ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")
st.title("🌸 Predictor de Especies de Iris")

# ── Carga de modelos ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        model      = joblib.load('components/iris_model.pkl')
        scaler     = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en 'components/'")
        return None, None, None

model, scaler, model_info = load_models()

if model is not None:

    # ── Formulario de predicción ───────────────────────────────────────────────
    st.header("Ingresa las características de la flor")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Longitud del Sépalo / l_s (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        sepal_width  = st.number_input("Ancho del Sépalo / a_s (cm)",    min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    with col2:
        petal_length = st.number_input("Longitud del Pétalo / l_p (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
        petal_width  = st.number_input("Ancho del Pétalo / a_p (cm)",    min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    if st.button("🔍 Predecir Especie"):
        features        = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]
        probabilities   = model.predict_proba(features_scaled)[0]
        target_names    = model_info['target_names']
        predicted_species = target_names[prediction]

        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")
        st.write("Probabilidades por especie:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")

        ok = insertar_prediccion(
            l_p=float(petal_length),
            l_s=float(sepal_length),
            a_s=float(sepal_width),
            a_p=float(petal_width),
            prediccion=predicted_species
        )
        if ok:
            st.toast("✅ Predicción guardada en la base de datos")

    # ── Histórico ─────────────────────────────────────────────────────────────
    st.divider()

    col_title, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("🔄 Actualizar"):
            st.rerun()

    rows = obtener_historico()

    if rows:
        st.markdown(renderizar_tabla_historico(rows), unsafe_allow_html=True)
    else:
        st.info("Aún no hay predicciones guardadas.")
