"""OCR + LLM: extrae texto de una imagen, lo amplía con un modelo GPT y evalúa el texto generado."""
import base64
import io
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageFilter, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import pytesseract
except ImportError:  # la app sigue funcionando con OCR por visión
    pytesseract = None

try:
    import textstat
except ImportError:
    textstat = None

st.set_page_config(page_title="OCR + GPT", page_icon="🖼️", layout="wide")

# ------------------------------------------------------------------ Configuración
PROVEEDORES = {
    "OpenAI": {"base_url": None,
               "modelos": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-4.1-nano"],
               "vision": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]},
    "Groq (GPT-OSS)": {"base_url": "https://api.groq.com/openai/v1",
                       "modelos": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
                       "vision": ["meta-llama/llama-4-scout-17b-16e-instruct"]},
}

ESTILOS = {
    "Formal": ("Redacta en registro formal, con tono cortés, vocabulario cuidado, párrafos bien "
               "cohesionados y sin tecnicismos innecesarios. Evita coloquialismos."),
    "Técnico": ("Redacta en registro técnico y preciso: usa terminología especializada, define los "
                "conceptos clave, organiza la información con encabezados y listas cuando ayude, e "
                "incluye detalles, supuestos y posibles aplicaciones."),
}

IDIOMAS_OCR = {"Español": "spa", "Inglés": "eng", "Español + Inglés": "spa+eng"}

STOPWORDS = set("""a al algo algunas algunos ante antes como con contra cual cuando de del desde donde
durante e el ella ellas ellos en entre era es esa esas ese eso esos esta estaba estado estas este esto
estos fue fueron ha han hasta hay la las le les lo los mas más me mi mis mucho muy nada ni no nos o otra
otros para pero poco por porque que qué quien se ser si sí sin sobre son su sus también tan te tiene
todo todos tu tus un una unas uno unos y ya yo the of and to in is that for it as with was on be by this
are or an at from which its has have not but they their""".split())

CONECTORES = ["además", "asimismo", "por lo tanto", "por consiguiente", "sin embargo", "no obstante",
              "en consecuencia", "por ejemplo", "es decir", "en primer lugar", "finalmente", "en conclusión",
              "por otro lado", "así", "debido a", "puesto que", "ya que", "mientras", "aunque", "en cambio",
              "de hecho", "en resumen", "por último", "igualmente", "dado que"]

# ------------------------------------------------------------------ Sidebar
st.sidebar.title("⚙️ Configuración")
proveedor = st.sidebar.radio("Proveedor", list(PROVEEDORES), horizontal=True)
api_key = st.sidebar.text_input(f"API key de {proveedor.split()[0]}", type="password",
                                placeholder="sk-..." if proveedor == "OpenAI" else "gsk_...")
cfg = PROVEEDORES[proveedor]
client = OpenAI(api_key=api_key, base_url=cfg["base_url"]) if api_key else None

st.sidebar.subheader("🧠 Parámetros del LLM")
modelo = st.sidebar.selectbox("Modelo", cfg["modelos"] + ["Otro (escribir)"])
if modelo == "Otro (escribir)":
    modelo = st.sidebar.text_input("ID del modelo", cfg["modelos"][0])
estilo = st.sidebar.radio("Estilo de respuesta", list(ESTILOS), horizontal=True)
temperature = st.sidebar.slider("Temperatura", 0.0, 2.0, 0.7, 0.05)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
max_tokens = st.sidebar.slider("Máx. tokens", 64, 4096, 800, 32)
extension = st.sidebar.select_slider("Extensión deseada", ["Breve", "Media", "Extensa"], "Media")
if proveedor == "OpenAI":
    freq_pen = st.sidebar.slider("Frequency penalty", -2.0, 2.0, 0.0, 0.1)
    pres_pen = st.sidebar.slider("Presence penalty", -2.0, 2.0, 0.0, 0.1)
else:
    freq_pen = pres_pen = 0.0
idioma_resp = st.sidebar.selectbox("Idioma de la respuesta", ["Español", "Inglés", "Mismo que el texto"])

st.sidebar.subheader("🔍 OCR")
metodos_ocr = ["Tesseract (local)", "Visión con LLM"] if pytesseract else ["Visión con LLM"]
metodo_ocr = st.sidebar.radio("Motor OCR", metodos_ocr)
idioma_ocr = st.sidebar.selectbox("Idioma del OCR", list(IDIOMAS_OCR))
psm = st.sidebar.selectbox("Segmentación de página (PSM)", [3, 4, 6, 11], index=0,
                           help="3 = automática, 4 = columna, 6 = bloque uniforme, 11 = texto disperso")
st.sidebar.markdown("**Preprocesamiento**")
pp_gris = st.sidebar.checkbox("Escala de grises", True)
pp_contraste = st.sidebar.checkbox("Auto-contraste", True)
pp_binarizar = st.sidebar.checkbox("Binarizar", False)
umbral = st.sidebar.slider("Umbral", 0, 255, 150, disabled=not pp_binarizar)
pp_nitidez = st.sidebar.checkbox("Aumentar nitidez", False)
escala = st.sidebar.slider("Escalar imagen", 1.0, 3.0, 1.0, 0.5)


# ------------------------------------------------------------------ Funciones: OCR
def preprocesar(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    if escala != 1.0:
        img = img.resize((int(img.width * escala), int(img.height * escala)), Image.LANCZOS)
    if pp_gris or pp_binarizar:
        img = ImageOps.grayscale(img)
    if pp_contraste:
        img = ImageOps.autocontrast(img)
    if pp_nitidez:
        img = img.filter(ImageFilter.SHARPEN)
    if pp_binarizar:
        img = img.point(lambda p: 255 if p > umbral else 0)
    return img


def ocr_tesseract(img):
    data = pytesseract.image_to_data(img, lang=IDIOMAS_OCR[idioma_ocr], config=f"--psm {psm}",
                                     output_type=pytesseract.Output.DATAFRAME)
    data = data[(data.conf != -1) & data.text.notna()]
    data = data[data.text.astype(str).str.strip() != ""]
    texto = pytesseract.image_to_string(img, lang=IDIOMAS_OCR[idioma_ocr], config=f"--psm {psm}")
    conf = float(data.conf.mean()) if len(data) else 0.0
    return texto.strip(), conf, data


def ocr_vision(img):
    modelo_v = cfg["vision"][0] if modelo not in cfg["vision"] else modelo
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    r = client.chat.completions.create(
        model=modelo_v, temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Transcribe literalmente TODO el texto visible en la imagen, respetando "
                                     "saltos de línea. Devuelve solo el texto, sin comentarios."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
    return (r.choices[0].message.content or "").strip(), None, None


# ------------------------------------------------------------------ Funciones: LLM
def ampliar(texto_ocr):
    largo = {"Breve": "entre 120 y 200 palabras", "Media": "entre 300 y 450 palabras",
             "Extensa": "entre 600 y 900 palabras"}[extension]
    idioma = "" if idioma_resp == "Mismo que el texto" else f"Responde en {idioma_resp.lower()}. "
    system = (f"Eres un experto redactor. {ESTILOS[estilo]} {idioma}"
              "Recibirás texto extraído por OCR (puede contener errores de reconocimiento): corrígelos "
              "de forma implícita, interpreta su sentido y AMPLÍA el contenido con contexto, explicaciones "
              f"y ejemplos pertinentes. Extensión objetivo: {largo}. No inventes datos específicos "
              "(cifras, nombres, fechas) que no estén en el texto.")
    params = dict(model=modelo, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": f"Texto extraído por OCR:\n\"\"\"\n{texto_ocr}\n\"\"\""}])
    if proveedor == "OpenAI":
        params.update(frequency_penalty=freq_pen, presence_penalty=pres_pen)
    r = client.chat.completions.create(**params)
    return r.choices[0].message.content or "", r.usage


def evaluar_con_llm(original, generado):
    prompt = (
        "Evalúa el TEXTO GENERADO como lingüista experto. Devuelve SOLO un JSON con este esquema:\n"
        '{"coherencia":{"puntaje":0-10,"comentario":""},"semantica":{"puntaje":0-10,"comentario":""},'
        '"sintaxis":{"puntaje":0-10,"comentario":""},"gramatica":{"puntaje":0-10,"comentario":""},'
        '"fidelidad_al_original":{"puntaje":0-10,"comentario":""},'
        '"errores_detectados":["..."],"resumen":""}\n\n'
        f"TEXTO ORIGINAL (OCR):\n{original[:3000]}\n\nTEXTO GENERADO:\n{generado[:6000]}")
    r = client.chat.completions.create(model=modelo, temperature=0,
                                       messages=[{"role": "user", "content": prompt}])
    contenido = r.choices[0].message.content or ""
    m = re.search(r"\{.*\}", contenido, re.S)
    return json.loads(m.group(0)) if m else None


# ------------------------------------------------------------------ Funciones: métricas
def oraciones(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", t) if len(s.strip()) > 2]


def palabras(t):
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", t)


def silabas(p):
    return max(1, len(re.findall(r"[aeiouáéíóúü]+", p.lower())))


def metricas_texto(t):
    ws, ss = palabras(t), oraciones(t)
    wl = [w.lower() for w in ws]
    n, ns = len(ws), max(len(ss), 1)
    contenido = [w for w in wl if w not in STOPWORDS]
    sil = sum(silabas(w) for w in ws)
    # Índice Fernández-Huerta (adaptación de Flesch al español): mayor = más fácil
    fh = 206.84 - 0.60 * (100 * sil / max(n, 1)) - 1.02 * (n / ns) if n else 0
    long_or = [len(palabras(s)) for s in ss]
    return {
        "Caracteres": len(t), "Palabras": n, "Oraciones": len(ss),
        "Párrafos": len([p for p in t.split("\n\n") if p.strip()]),
        "Palabras únicas": len(set(wl)),
        "Riqueza léxica (TTR)": round(len(set(wl)) / max(n, 1), 3),
        "Densidad léxica": round(len(contenido) / max(n, 1), 3),
        "Palabras / oración": round(n / ns, 2),
        "Desv. long. oración": round(float(np.std(long_or)) if long_or else 0, 2),
        "Letras / palabra": round(sum(len(w) for w in ws) / max(n, 1), 2),
        "Sílabas / palabra": round(sil / max(n, 1), 2),
        "Palabras largas (>6 letras) %": round(100 * sum(len(w) > 6 for w in ws) / max(n, 1), 1),
        "Legibilidad Fernández-Huerta": round(fh, 1),
        "Tiempo de lectura (min)": round(n / 200, 1),
    }


def metricas_coherencia(t):
    ss = oraciones(t)
    if len(ss) < 2:
        return {"local": None, "global": None, "matriz": None, "oraciones": ss}
    X = TfidfVectorizer().fit_transform(ss + [t])
    S = cosine_similarity(X[:-1])
    local = float(np.mean([S[i, i + 1] for i in range(len(ss) - 1)]))
    glob = float(np.mean(cosine_similarity(X[:-1], X[-1]).ravel()))
    return {"local": local, "global": glob, "matriz": S, "oraciones": ss}


def metricas_semantica(original, generado):
    if not original.strip() or not generado.strip():
        return {}
    tf = TfidfVectorizer().fit([original, generado])
    sim = float(cosine_similarity(tf.transform([original]), tf.transform([generado]))[0, 0])
    claves_o = [w for w, _ in Counter(w.lower() for w in palabras(original)
                                      if w.lower() not in STOPWORDS and len(w) > 3).most_common(15)]
    gen_l = generado.lower()
    cubiertas = [w for w in claves_o if w in gen_l]
    top_gen = Counter(w.lower() for w in palabras(generado) if w.lower() not in STOPWORDS and len(w) > 3)
    return {"similitud": sim, "cobertura": len(cubiertas) / max(len(claves_o), 1),
            "claves_original": claves_o, "cubiertas": cubiertas, "top_generado": top_gen.most_common(15),
            "factor_ampliacion": len(palabras(generado)) / max(len(palabras(original)), 1)}


def metricas_sintaxis(t):
    ss = oraciones(t)
    tl = t.lower()
    conectores = {c: len(re.findall(rf"\b{re.escape(c)}\b", tl)) for c in CONECTORES}
    conectores = {k: v for k, v in conectores.items() if v}
    n_or = max(len(ss), 1)
    return {
        "Comas / oración": round(t.count(",") / n_or, 2),
        "Subordinadas aprox. / oración": round(len(re.findall(r"\b(que|porque|cuando|donde|aunque|si|cual|cuyo)\b",
                                                              tl)) / n_or, 2),
        "Conectores discursivos": sum(conectores.values()),
        "Oraciones interrogativas": t.count("?"),
        "Oraciones muy largas (>35 palabras)": sum(len(palabras(s)) > 35 for s in ss),
        "Voz pasiva aprox.": len(re.findall(r"\b(es|son|fue|fueron|será|serán|ha sido|han sido)\s+\w+(ado|ido|ados|idos|ada|ida)\b", tl)),
        "_conectores": conectores,
    }


def revision_gramatical(t):
    problemas = []
    for m in re.finditer(r"\b(\w+)\s+\1\b", t, re.I):
        problemas.append(("Palabra repetida", m.group(0)))
    for m in re.finditer(r" {2,}", t):
        problemas.append(("Espacios dobles", repr(m.group(0))))
    for m in re.finditer(r"\s+[,.;:]", t):
        problemas.append(("Espacio antes de puntuación", m.group(0).strip()))
    for m in re.finditer(r"[,.;:](?=[A-Za-zÁÉÍÓÚáéíóúñÑ])", t):
        problemas.append(("Falta espacio después de puntuación", t[max(m.start() - 8, 0):m.end() + 8]))
    for s in oraciones(t):
        s2 = s.lstrip("-*#•¿¡\"'“( 0123456789.)")
        if s2 and s2[0].isalpha() and s2[0].islower():
            problemas.append(("Oración inicia en minúscula", s[:50]))
    if t.count("¿") != t.count("?"):
        problemas.append(("Signos de interrogación desbalanceados", f"¿={t.count('¿')} ?={t.count('?')}"))
    if t.count("¡") != t.count("!"):
        problemas.append(("Signos de exclamación desbalanceados", f"¡={t.count('¡')} !={t.count('!')}"))
    if t.count("(") != t.count(")"):
        problemas.append(("Paréntesis desbalanceados", f"(={t.count('(')} )={t.count(')')}"))
    n = max(len(palabras(t)), 1)
    puntaje = max(0.0, 100 - 100 * len(problemas) / n * 5)
    return problemas, round(puntaje, 1)


def gauge(valor, titulo, maximo=1.0):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=valor, title={"text": titulo},
                                 gauge={"axis": {"range": [0, maximo]}, "bar": {"color": "#4C78A8"}}))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ------------------------------------------------------------------ Interfaz
st.title("🖼️ OCR + GPT: extrae, amplía y evalúa")
st.caption("1) Sube una imagen · 2) Extrae el texto con OCR · 3) Amplíalo con un LLM · 4) Revisa las métricas")

if not client:
    st.info("👈 Ingresa tu API key en la barra lateral para usar el LLM.")

archivo = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"])

if archivo:
    original = Image.open(archivo)
    proc = preprocesar(original)
    c1, c2 = st.columns(2)
    c1.image(original, caption="Original", width="stretch")
    c2.image(proc, caption="Preprocesada (entrada del OCR)", width="stretch")

    if st.button("🔍 Extraer texto (OCR)", type="primary"):
        with st.spinner("Ejecutando OCR..."):
            try:
                if metodo_ocr.startswith("Tesseract"):
                    texto, conf, datos = ocr_tesseract(proc)
                else:
                    if not client:
                        st.error("El OCR por visión necesita API key.")
                        st.stop()
                    texto, conf, datos = ocr_vision(original)
                st.session_state.update(ocr_texto=texto, ocr_conf=conf, ocr_datos=datos, generado=None,
                                        eval_llm=None)
            except Exception as e:
                st.error(f"Error en OCR: {e}")

if "ocr_texto" in st.session_state:
    st.subheader("📄 Texto extraído")
    conf = st.session_state.get("ocr_conf")
    datos = st.session_state.get("ocr_datos")
    if conf is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Confianza media OCR", f"{conf:.1f}%")
        m2.metric("Palabras reconocidas", len(datos) if datos is not None else 0)
        m3.metric("Palabras con confianza < 60%", int((datos.conf < 60).sum()) if datos is not None else 0)
        if datos is not None and len(datos):
            with st.expander("Confianza por palabra"):
                st.plotly_chart(px.bar(datos.reset_index(drop=True), y="conf", hover_data=["text"],
                                       labels={"index": "palabra #", "conf": "confianza %"}), width="stretch")
    texto_edit = st.text_area("Puedes corregir el texto antes de enviarlo al LLM",
                              st.session_state["ocr_texto"], height=180)
    st.session_state["ocr_texto"] = texto_edit

    if st.button(f"✨ Ampliar con {modelo} · estilo {estilo.lower()}", type="primary",
                 disabled=not (client and texto_edit.strip())):
        with st.spinner("Generando..."):
            try:
                gen, uso = ampliar(texto_edit)
                st.session_state.update(generado=gen, uso=uso, eval_llm=None)
            except Exception as e:
                st.error(f"Error del LLM: {e}")

gen = st.session_state.get("generado")
if gen:
    st.subheader(f"📝 Respuesta ampliada ({estilo.lower()})")
    st.markdown(gen)
    uso = st.session_state.get("uso")
    if uso:
        st.caption(f"Tokens → entrada: {uso.prompt_tokens} · salida: {uso.completion_tokens} · "
                   f"total: {uso.total_tokens}")
    st.download_button("⬇️ Descargar respuesta (.md)", gen, "respuesta.md")

    st.header("📊 Métricas del texto generado")
    orig = st.session_state["ocr_texto"]
    t_med, t_coh, t_sem, t_sin, t_gra, t_llm = st.tabs(
        ["📏 Medidas", "🔗 Coherencia", "🧩 Semántica", "🌳 Sintaxis", "✅ Gramática", "🤖 Evaluación LLM"])

    with t_med:
        mg, mo = metricas_texto(gen), metricas_texto(orig)
        cols = st.columns(4)
        for i, k in enumerate(["Palabras", "Oraciones", "Riqueza léxica (TTR)", "Legibilidad Fernández-Huerta"]):
            cols[i].metric(k, mg[k], delta=round(mg[k] - mo[k], 2), help="Delta respecto al texto OCR")
        st.dataframe(pd.DataFrame({"Texto OCR": mo, "Texto generado": mg}), width="stretch")
        fh = mg["Legibilidad Fernández-Huerta"]
        nivel = ("Muy fácil" if fh >= 90 else "Fácil" if fh >= 80 else "Algo fácil" if fh >= 70 else
                 "Normal" if fh >= 60 else "Algo difícil" if fh >= 50 else "Difícil" if fh >= 30 else "Muy difícil")
        st.info(f"Legibilidad (Fernández-Huerta): **{fh} → {nivel}**")
        if textstat:
            textstat.set_lang("es")
            st.caption(f"textstat · Szigriszt-Pazos: {textstat.szigriszt_pazos(gen):.1f} · "
                       f"Gutiérrez de Polini: {textstat.gutierrez_polini(gen):.1f} · "
                       f"Crawford (años escolares): {textstat.crawford(gen):.1f}")
        largos = [len(palabras(s)) for s in oraciones(gen)]
        st.plotly_chart(px.bar(x=list(range(1, len(largos) + 1)), y=largos,
                               labels={"x": "oración", "y": "palabras"}, title="Longitud de cada oración"),
                        width="stretch")

    with t_coh:
        c = metricas_coherencia(gen)
        if c["local"] is None:
            st.warning("Se necesitan al menos 2 oraciones.")
        else:
            g1, g2 = st.columns(2)
            g1.plotly_chart(gauge(c["local"], "Coherencia local<br><sub>oraciones consecutivas</sub>"),
                            width="stretch")
            g2.plotly_chart(gauge(c["global"], "Coherencia global<br><sub>oración vs. texto</sub>"),
                            width="stretch")
            st.caption("Similitud coseno TF-IDF. Valores altos indican que las oraciones comparten vocabulario "
                       "y tema; valores muy bajos pueden señalar saltos temáticos.")
            et = [f"O{i+1}" for i in range(len(c["oraciones"]))]
            st.plotly_chart(px.imshow(c["matriz"], x=et, y=et, color_continuous_scale="Blues", zmin=0, zmax=1,
                                      title="Similitud entre oraciones"), width="stretch")

    with t_sem:
        s = metricas_semantica(orig, gen)
        if s:
            a, b, d = st.columns(3)
            a.plotly_chart(gauge(s["similitud"], "Similitud con el OCR"), width="stretch")
            b.plotly_chart(gauge(s["cobertura"], "Cobertura de palabras clave"), width="stretch")
            d.metric("Factor de ampliación", f"×{s['factor_ampliacion']:.1f}")
            st.markdown("**Palabras clave del texto original** (✅ presentes en la respuesta)")
            st.markdown(" ".join(f"`{'✅' if w in s['cubiertas'] else '❌'} {w}`" for w in s["claves_original"]))
            if s["top_generado"]:
                df_top = pd.DataFrame(s["top_generado"], columns=["término", "frecuencia"])
                st.plotly_chart(px.bar(df_top, x="frecuencia", y="término", orientation="h",
                                       title="Términos más frecuentes en la respuesta").update_yaxes(
                    autorange="reversed"), width="stretch")

    with t_sin:
        sx = metricas_sintaxis(gen)
        conect = sx.pop("_conectores")
        cols = st.columns(3)
        for i, (k, v) in enumerate(sx.items()):
            cols[i % 3].metric(k, v)
        if conect:
            st.plotly_chart(px.bar(x=list(conect), y=list(conect.values()),
                                   labels={"x": "conector", "y": "usos"}, title="Conectores discursivos"),
                            width="stretch")

    with t_gra:
        problemas, puntaje = revision_gramatical(gen)
        st.plotly_chart(gauge(puntaje, "Puntaje ortotipográfico", 100), width="stretch")
        if problemas:
            st.dataframe(pd.DataFrame(problemas, columns=["tipo", "fragmento"]), width="stretch", hide_index=True)
        else:
            st.success("No se detectaron problemas ortotipográficos con las reglas automáticas.")
        st.caption("Revisión basada en reglas. Para un análisis gramatical profundo usa la pestaña 'Evaluación LLM'.")

    with t_llm:
        st.write("El modelo actúa como evaluador y puntúa coherencia, semántica, sintaxis, gramática y fidelidad.")
        if st.button("🤖 Evaluar con el LLM", disabled=not client):
            with st.spinner("Evaluando..."):
                try:
                    st.session_state["eval_llm"] = evaluar_con_llm(orig, gen)
                except Exception as e:
                    st.error(f"Error: {e}")
        ev = st.session_state.get("eval_llm")
        if ev:
            dims = [k for k in ["coherencia", "semantica", "sintaxis", "gramatica", "fidelidad_al_original"]
                    if k in ev]
            vals = [float(ev[k].get("puntaje", 0)) for k in dims]
            fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=dims + dims[:1], fill="toself"))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10])), height=380, showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pd.DataFrame([{"dimensión": k, "puntaje": ev[k].get("puntaje"),
                                        "comentario": ev[k].get("comentario")} for k in dims]),
                         width="stretch", hide_index=True)
            if ev.get("errores_detectados"):
                st.markdown("**Errores detectados:**\n" + "\n".join(f"- {e}" for e in ev["errores_detectados"]))
            if ev.get("resumen"):
                st.info(ev["resumen"])
