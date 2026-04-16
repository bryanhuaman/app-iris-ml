# ── NUEVO: insertar predicción ────────────────────────────────────────────────
def insertar_prediccion(l_p, l_s, a_s, a_p, prediccion):
    try:
        conn = psycopg2.connect(
            user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME,
            options="-c search_path=public"
        )
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO public.tb_iris (l_p, l_s, a_s, a_p, prediccion) VALUES (%s, %s, %s, %s, %s)",
            (l_p, l_s, a_s, a_p, prediccion)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error al guardar: {e}")
        return False

# ── NUEVO: obtener histórico ──────────────────────────────────────────────────
def obtener_historico():
    try:
        conn = psycopg2.connect(
            user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME,
            options="-c search_path=public"
        )
        cur  = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, created_at, l_p, l_s, a_s, a_p, prediccion FROM public.tb_iris ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Error al obtener historial: {e}")
        return []
