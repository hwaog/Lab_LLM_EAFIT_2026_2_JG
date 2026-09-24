import numpy as np
import pandas as pd
import streamlit as st
import tiktoken
import plotly.express as px
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

st.set_page_config(page_title="LLM Lab con Groq", page_icon="🧠", layout="wide")

# Modelos GPT (open-weight de OpenAI) servidos por Groq; se complementa con la lista en vivo
DEFAULT_GPT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
ENCODINGS = {
    "o200k_base (GPT-4o / gpt-oss)": "o200k_base",
    "cl100k_base (GPT-4 / GPT-3.5)": "cl100k_base",
    "p50k_base (Codex / text-davinci)": "p50k_base",
    "gpt2 (GPT-2)": "gpt2",
}

@st.cache_resource(show_spinner="Cargando tokenizador...")
def get_enc(name):
    try:
        return tiktoken.get_encoding(name)
    except Exception as e:
        st.warning(f"No se pudo cargar el tokenizador '{name}' (requiere internet la primera vez): {e}")
        return None


# ---------------- Sidebar ----------------
st.sidebar.title("🔑 Configuración")
api_key = st.sidebar.text_input("API key de Groq", type="password", placeholder="gsk_...")
client = None
models = DEFAULT_GPT_MODELS

if api_key:
    try:
        client = Groq(api_key=api_key)
        live = sorted(m.id for m in client.models.list().data)
        gpt_live = [m for m in live if "gpt" in m.lower()]
        solo_gpt = st.sidebar.checkbox("Mostrar solo modelos GPT", value=True)
        models = (gpt_live or DEFAULT_GPT_MODELS) if solo_gpt else live
        st.sidebar.success(f"Conectado · {len(live)} modelos disponibles")
    except Exception as e:
        st.sidebar.error(f"API key inválida o error de conexión: {e}")
        client = None
else:
    st.sidebar.info("Ingresa tu API key para habilitar la generación.")

st.sidebar.markdown("---")
st.sidebar.caption("Obtén tu key en console.groq.com")

st.title("🧠 LLM Lab: tokens, representaciones y generación")

tab_models, tab_tok, tab_bow, tab_sim, tab_emb, tab_gen = st.tabs(
    ["🤖 Modelos", "🔤 Tokens", "🧺 Bag of Words", "📏 Similitud", "📍 Embeddings", "✍️ Generación"]
)

# ---------------- Modelos ----------------
with tab_models:
    st.subheader("Modelos disponibles")
    if client:
        try:
            data = client.models.list().data
            df = pd.DataFrame(
                [{"id": m.id, "propietario": getattr(m, "owned_by", ""),
                  "ventana de contexto": getattr(m, "context_window", None),
                  "activo": getattr(m, "active", None)} for m in data]
            ).sort_values("id")
            st.dataframe(df, width="stretch", hide_index=True)
        except Exception as e:
            st.error(e)
    else:
        st.write("Modelos GPT por defecto:", DEFAULT_GPT_MODELS)

# ---------------- Tokens ----------------
with tab_tok:
    st.subheader("Tokenización y Token IDs")
    text_tok = st.text_area("Texto", "Los modelos de lenguaje convierten el texto en tokens antes de procesarlo.")
    enc_names = st.multiselect("Tokenizadores a comparar", list(ENCODINGS), default=list(ENCODINGS)[:2])
    colors = ["#FFD6A5", "#CAFFBF", "#9BF6FF", "#BDB2FF", "#FFC6FF", "#FDFFB6"]
    resumen = []
    for name in enc_names:
        enc = get_enc(ENCODINGS[name])
        if enc is None:
            continue
        ids = enc.encode(text_tok)
        toks = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]
        resumen.append({"tokenizador": name, "tokens": len(ids), "caracteres": len(text_tok),
                        "caracteres/token": round(len(text_tok) / max(len(ids), 1), 2)})
        st.markdown(f"**{name}** — {len(ids)} tokens")
        html = "".join(
            f'<span style="background:{colors[i % len(colors)]};color:#000;padding:2px 3px;'
            f'margin:1px;border-radius:4px;display:inline-block">{t.replace(" ", "␣")}</span>'
            for i, t in enumerate(toks))
        st.markdown(html, unsafe_allow_html=True)
        with st.expander("Ver tabla token → ID"):
            st.dataframe(pd.DataFrame({"token": toks, "token_id": ids}), width="stretch")
    if resumen:
        st.dataframe(pd.DataFrame(resumen), width="stretch", hide_index=True)

# ---------------- Documentos compartidos ----------------
DOCS_DEFAULT = """El gato duerme en el sofá.
El perro duerme en la alfombra.
Los mercados financieros cayeron hoy.
La bolsa de valores tuvo una caída fuerte.
Me gusta programar en Python."""

# ---------------- Bag of Words ----------------
with tab_bow:
    st.subheader("Bag of Words y TF-IDF")
    docs_bow = [d for d in st.text_area("Un documento por línea", DOCS_DEFAULT, key="bow").splitlines() if d.strip()]
    c1, c2 = st.columns(2)
    ngram = c1.slider("n-gramas máximos", 1, 3, 1)
    modo = c2.radio("Representación", ["Conteo (BoW)", "TF-IDF"], horizontal=True)
    if docs_bow:
        vec = CountVectorizer(ngram_range=(1, ngram)) if modo.startswith("Conteo") else TfidfVectorizer(ngram_range=(1, ngram))
        X = vec.fit_transform(docs_bow)
        df_bow = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out(),
                              index=[f"Doc {i+1}" for i in range(len(docs_bow))])
        st.write(f"Vocabulario: **{len(vec.get_feature_names_out())}** términos")
        st.dataframe(df_bow.style.background_gradient(cmap="Blues", axis=None).format(precision=2),
                     width="stretch")
        freq = df_bow.sum().sort_values(ascending=False).head(20)
        st.plotly_chart(px.bar(freq, labels={"index": "término", "value": "peso"}, title="Términos más relevantes"),
                        width="stretch")

# ---------------- Similitud ----------------
def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)

with tab_sim:
    st.subheader("Métricas de similitud")
    c1, c2 = st.columns(2)
    a = c1.text_area("Texto A", "El gato duerme en el sofá.")
    b = c2.text_area("Texto B", "El perro duerme en la alfombra.")
    if a and b:
        tf = TfidfVectorizer().fit([a, b])
        va, vb = tf.transform([a]).toarray(), tf.transform([b]).toarray()
        cv = CountVectorizer().fit([a, b])
        ca, cb = cv.transform([a]).toarray(), cv.transform([b]).toarray()
        enc = get_enc("o200k_base")
        ta, tb = (set(enc.encode(a)), set(enc.encode(b))) if enc else (set(), set())
        m = st.columns(5)
        m[0].metric("Coseno TF-IDF", f"{cosine_similarity(va, vb)[0, 0]:.3f}")
        m[1].metric("Coseno BoW", f"{cosine_similarity(ca, cb)[0, 0]:.3f}")
        m[2].metric("Jaccard palabras", f"{jaccard(a, b):.3f}")
        m[3].metric("Jaccard tokens", f"{len(ta & tb) / max(len(ta | tb), 1):.3f}")
        m[4].metric("Dist. euclidiana TF-IDF", f"{euclidean_distances(va, vb)[0, 0]:.3f}")

    st.markdown("#### Matriz de similitud entre documentos")
    docs_sim = [d for d in st.text_area("Un documento por línea", DOCS_DEFAULT, key="sim").splitlines() if d.strip()]
    if len(docs_sim) > 1:
        S = cosine_similarity(TfidfVectorizer().fit_transform(docs_sim))
        labels = [f"Doc {i+1}" for i in range(len(docs_sim))]
        st.plotly_chart(px.imshow(S, x=labels, y=labels, text_auto=".2f", color_continuous_scale="Viridis",
                                  zmin=0, zmax=1), width="stretch")

# ---------------- Embeddings ----------------
with tab_emb:
    st.subheader("Embeddings (LSA: TF-IDF + SVD) y proyección 2D")
    st.caption("Groq no ofrece endpoint de embeddings; aquí se calculan embeddings densos localmente con LSA.")
    docs_emb = [d for d in st.text_area("Un documento por línea", DOCS_DEFAULT, key="emb").splitlines() if d.strip()]
    if len(docs_emb) >= 3:
        X = TfidfVectorizer().fit_transform(docs_emb)
        max_dim = max(2, min(X.shape) - 1)
        dim = st.slider("Dimensión del embedding", 2, max(max_dim, 2), min(4, max_dim))
        E = TruncatedSVD(n_components=dim, random_state=42).fit_transform(X)
        E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
        st.dataframe(pd.DataFrame(E, index=[f"Doc {i+1}" for i in range(len(docs_emb))],
                                  columns=[f"d{j}" for j in range(dim)]).round(3), width="stretch")
        P = PCA(n_components=2).fit_transform(E)
        fig = px.scatter(x=P[:, 0], y=P[:, 1], text=[d[:40] for d in docs_emb], title="Proyección PCA 2D")
        fig.update_traces(textposition="top center", marker=dict(size=12))
        st.plotly_chart(fig, width="stretch")
        consulta = st.text_input("Búsqueda semántica", "animales durmiendo")
        if consulta:
            tfv = TfidfVectorizer().fit(docs_emb + [consulta])
            sims = cosine_similarity(tfv.transform([consulta]), tfv.transform(docs_emb))[0]
            st.dataframe(pd.DataFrame({"documento": docs_emb, "similitud": sims})
                         .sort_values("similitud", ascending=False), width="stretch", hide_index=True)
    else:
        st.info("Ingresa al menos 3 documentos.")

# ---------------- Generación ----------------
with tab_gen:
    st.subheader("Generación de texto")
    c1, c2 = st.columns([1, 2])
    with c1:
        model = st.selectbox("Modelo", models)
        temperature = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.05)
        top_p = st.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
        max_tokens = st.slider("Máx. tokens", 16, 4096, 512, 16)
        seed = st.number_input("Seed (0 = aleatorio)", 0, 10_000, 0)
        stop = st.text_input("Secuencia de parada (opcional)")
        comparar = st.checkbox("Comparar temperaturas (0.0 / 0.7 / 1.5)")
    with c2:
        system = st.text_area("Prompt de sistema", "Eres un asistente útil que responde en español.")
        prompt = st.text_area("Prompt", "Escribe un párrafo corto sobre cómo funcionan los tokens en un LLM.", height=150)
        run = st.button("🚀 Generar", type="primary", disabled=client is None)

    def generar(temp):
        params = dict(model=model, temperature=temp, top_p=top_p, max_tokens=max_tokens,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        if seed:
            params["seed"] = int(seed)
        if stop:
            params["stop"] = [stop]
        return client.chat.completions.create(**params)

    if run and client:
        temps = [0.0, 0.7, 1.5] if comparar else [temperature]
        cols = st.columns(len(temps))
        for col, t in zip(cols, temps):
            with col, st.spinner(f"Generando (T={t})..."):
                try:
                    r = generar(t)
                    st.markdown(f"**Temperatura {t}**")
                    st.write(r.choices[0].message.content)
                    u = r.usage
                    st.caption(f"Tokens → prompt: {u.prompt_tokens} · respuesta: {u.completion_tokens} · "
                               f"total: {u.total_tokens} · fin: {r.choices[0].finish_reason}")
                except Exception as e:
                    st.error(f"Error: {e}")
    elif client is None:
        st.warning("Ingresa tu API key de Groq en la barra lateral.")
