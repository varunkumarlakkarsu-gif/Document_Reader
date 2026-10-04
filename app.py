"""
app.py
The user interface. Run with:  streamlit run app.py
"""
import streamlit as st

from ocr_engine import draw_boxes, extract_text, join_text, load_image

st.set_page_config(page_title="Document Reader", page_icon="📄", layout="wide")
st.title("📄 Document Reader")
st.caption("Upload an image and the text inside it will be extracted with EasyOCR.")

# ---- Sidebar: options -------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    language_options = {"English": "en", "German": "de", "French": "fr",
                        "Spanish": "es", "Italian": "it"}
    chosen = st.multiselect("Languages in the document",
                            list(language_options), default=["English"])
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.3, 0.05,
                         help="Hide text the model is unsure about.")
    show_boxes = st.checkbox("Show detected text boxes", value=True)

# ---- Upload -----------------------------------------------------------------
uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"])

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

if not chosen:
    st.warning("Pick at least one language in the sidebar.")
    st.stop()

# ---- Process ----------------------------------------------------------------
image = load_image(uploaded.getvalue())
languages = [language_options[name] for name in chosen]

with st.spinner("Reading text... (the first run downloads the model)"):
    lines = extract_text(image, languages=languages, min_confidence=min_conf)

# ---- Display ----------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Image")
    st.image(draw_boxes(image, lines) if show_boxes else image, use_container_width=True)

with right:
    st.subheader("Extracted text")
    if not lines:
        st.warning("No text found. Try a sharper image or lower the confidence.")
    else:
        text = join_text(lines)
        st.text_area("Result", text, height=350)
        st.download_button("Download as .txt", text, file_name="extracted_text.txt")
        with st.expander("Per-line confidence"):
            for line in lines:
                st.write(f"{line['confidence']:.0%} — {line['text']}")
