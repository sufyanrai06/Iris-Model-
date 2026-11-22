from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
from PIL import Image
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Iris Botanical Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. High-Contrast Enterprise Botanical Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #F8FAFC !important;
    }

    .stApp {
        background: linear-gradient(135deg, #090E16 0%, #101926 55%, #0B151D 100%);
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Force crisp high-contrast readability on all native Streamlit widgets */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #F1F5F9;
    }

    [data-testid="stSidebar"] {
        background-color: #0D1520 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }
    [data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }

    /* Radio & File Uploader High Contrast */
    div[role="radiogroup"] label p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.96rem !important;
    }
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px dashed rgba(52, 211, 153, 0.45);
        border-radius: 12px;
        padding: 12px;
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
    }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span {
        color: #E2E8F0 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #10B981 !important;
        color: #052E16 !important;
        font-weight: 700 !important;
        border: none !important;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #6EE7B7 !important;
        border: 1px solid rgba(52, 211, 153, 0.45);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #CBD5E1 !important;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* Result Cards */
    .result-valid {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.22), rgba(6, 78, 59, 0.45));
        border: 1.5px solid #34D399;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .result-invalid {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.22), rgba(127, 29, 29, 0.45));
        border: 1.5px solid #F87171;
        border-radius: 16px;
        padding: 24px;
    }
    .species-tag {
        color: #6EE7B7 !important;
        font-size: 0.86rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .species-name {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 4px 0 10px 0;
    }
    .confidence-pill {
        display: inline-block;
        background: #10B981;
        color: #042F1E !important;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 800;
    }

    /* Botanical Spec Grid */
    .spec-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 12px;
    }
    .spec-label {
        color: #6EE7B7 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .spec-value {
        color: #FFFFFF !important;
        font-size: 0.98rem;
        font-weight: 500;
        margin-top: 4px;
        line-height: 1.45;
    }

    /* Probability Bars */
    .prob-row {
        margin-bottom: 16px;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.98rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 6px;
    }
    .prob-track {
        width: 100%;
        height: 12px;
        background: rgba(255, 255, 255, 0.14);
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill-top {
        height: 100%;
        background: linear-gradient(90deg, #10B981, #6EE7B7);
        border-radius: 999px;
    }
    .prob-fill-other {
        height: 100%;
        background: #64748B;
        border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Botanical Metadata Database
BOTANICAL_INFO = {
    "Iris Setosa": {
        "scientific": "Iris setosa Pall. ex Link",
        "common_name": "Bristle-pointed Iris / Beachhead Iris",
        "morphology": "Three broad, spreading outer falls with tiny, bristle-like upright inner petals (standards), creating a distinct three-petaled star silhouette.",
        "avg_dimensions": "Petal Length: 1.0–1.9 cm | Petal Width: 0.1–0.6 cm",
        "habitat": "Arctic & subarctic coastal meadows, gravel beaches (Alaska, Canada, Siberia).",
    },
    "Iris Versicolor": {
        "scientific": "Iris versicolor L.",
        "common_name": "Harlequin Blueflag / Larger Blue Flag",
        "morphology": "Medium violet-blue falls marked with a bright yellow-to-white central signal patch and intricate dark purple veining; upright inner petals.",
        "avg_dimensions": "Petal Length: 3.0–5.1 cm | Petal Width: 1.0–1.8 cm",
        "habitat": "Wetlands, marshes, sedge meadows, and shorelines across Eastern North America.",
    },
    "Iris Virginica": {
        "scientific": "Iris virginica L.",
        "common_name": "Virginia Iris / Southern Blue Flag",
        "morphology": "Large, gracefully drooping broad violet-blue falls with a bright yellow-orange crest and tall, wavy upright inner petals.",
        "avg_dimensions": "Petal Length: 4.5–6.9 cm | Petal Width: 1.4–2.5 cm",
        "habitat": "Freshwater marshes, swamps, and stream banks of the Southeastern & Central US.",
    },
}


# 4. Load Vision Engine (ONNX Runtime)
@st.cache_resource
def load_vision_engine():
    repo_id = "Xenova/clip-vit-base-patch32"
    vision_path = hf_hub_download(
        repo_id=repo_id, filename="onnx/vision_model.onnx"
    )
    text_path = hf_hub_download(repo_id=repo_id, filename="onnx/text_model.onnx")
    vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.json")
    merges_path = hf_hub_download(repo_id=repo_id, filename="merges.txt")

    vision_sess = ort.InferenceSession(
        vision_path, providers=["CPUExecutionProvider"]
    )
    text_sess = ort.InferenceSession(
        text_path, providers=["CPUExecutionProvider"]
    )
    return vision_sess, text_sess, vocab_path, merges_path


@st.cache_resource
def get_tokenizer(vocab_file, merges_file):
    import json
    import re

    with open(vocab_file, "r", encoding="utf-8") as f:
        encoder = json.load(f)
    with open(merges_file, "r", encoding="utf-8") as f:
        merges = f.read().strip().split("\n")[1 : 49152 - 256 + 1]
    bpe_ranks = dict(zip([tuple(m.split()) for m in merges], range(len(merges))))

    def bytes_to_unicode():
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    byte_encoder = bytes_to_unicode()
    pat = re.compile(
        r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^\sa-zA-Z0-9]+""",
        re.IGNORECASE,
    )

    def bpe(token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = set(zip(word[:-1], word[1:]))
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(
                pairs, key=lambda pair: bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if (
                    word[i] == first
                    and i < len(word) - 1
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = set(zip(word[:-1], word[1:]))
        return " ".join(word)

    def encode_texts(texts):
        sot = encoder["<|startoftext|>"]
        eot = encoder["<|endoftext|>"]
        all_ids = np.zeros((len(texts), 77), dtype=np.int64)
        all_mask = np.zeros((len(texts), 77), dtype=np.int64)
        for idx, text in enumerate(texts):
            text = re.sub(r"\s+", " ", text).strip().lower()
            tokens = [sot]
            for match in re.findall(pat, text):
                encoded = "".join(
                    byte_encoder[b] for b in match.encode("utf-8")
                )
                for bpe_tok in bpe(encoded).split(" "):
                    if bpe_tok in encoder:
                        tokens.append(encoder[bpe_tok])
            tokens.append(eot)
            tokens = tokens[:77]
            all_ids[idx, : len(tokens)] = tokens
            all_mask[idx, : len(tokens)] = 1
        return all_ids, all_mask

    return encode_texts


vision_sess, text_sess, vocab_path, merges_path = load_vision_engine()
encode_texts = get_tokenizer(vocab_path, merges_path)


def preprocess_image(img: Image.Image):
    img = img.resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def get_image_embedding(img: Image.Image):
    pixel_values = preprocess_image(img)
    img_embeds = vision_sess.run(None, {"pixel_values": pixel_values})[0]
    return img_embeds / np.linalg.norm(img_embeds, axis=-1, keepdims=True)


def get_text_embeddings(prompts: list):
    input_ids, attention_mask = encode_texts(prompts)
    txt_inputs = {"input_ids": input_ids}
    if any(inp.name == "attention_mask" for inp in text_sess.get_inputs()):
        txt_inputs["attention_mask"] = attention_mask
    txt_embeds = text_sess.run(None, txt_inputs)[0]
    return txt_embeds / np.linalg.norm(txt_embeds, axis=-1, keepdims=True)


@st.cache_resource
def build_calibrated_prototypes():
    # Balanced 4-prompt ensembles + mean-centering to remove class prior bias
    prompt_groups = {
        "Iris Setosa": [
            "a botanical photo of Iris setosa flower with three wide outer sepals and tiny reduced inner petals",
            "Iris setosa bristle-pointed iris bloom with open three-petal star shape and short compact standards",
            "photo of Iris setosa blue-violet flower with very small upright petals and smooth rounded falls",
            "Iris setosa wildflower specimen showing three broad spreading petals without tall inner standards",
        ],
        "Iris Versicolor": [
            "a botanical photo of Iris versicolor harlequin blueflag flower with yellow-white throat and dark purple veins",
            "Iris versicolor blue flag bloom showing prominent veined purple petals and bright yellow signal patch",
            "photo of Iris versicolor flower with upright blue-violet inner petals and veined lower falls",
            "Iris versicolor wetland iris specimen with variegated purple veining and yellow-white petal base",
        ],
        "Iris Virginica": [
            "a botanical photo of Iris virginica southern blue flag flower with large broad drooping violet petals",
            "Iris virginica bloom with tall wavy upright inner petals and wide cascading deep purple falls",
            "photo of Iris virginica large violet-blue iris flower with broad arching lower sepals",
            "Iris virginica specimen showing large ruffled violet petals and bright yellow crest",
        ],
    }
    species_names = list(prompt_groups.keys())
    raw_prototypes = []
    for name in species_names:
        embeds = get_text_embeddings(prompt_groups[name])
        mean_embed = np.mean(embeds, axis=0)
        mean_embed = mean_embed / np.linalg.norm(mean_embed)
        raw_prototypes.append(mean_embed)

    raw_prototypes = np.array(raw_prototypes)
    # Subtract shared Iris concept centroid so subtle morphological differences stand out cleanly
    shared_centroid = np.mean(raw_prototypes, axis=0, keepdims=True)
    calibrated = raw_prototypes - 0.85 * shared_centroid
    calibrated = calibrated / np.linalg.norm(calibrated, axis=-1, keepdims=True)
    return species_names, calibrated


species_names, species_prototypes = build_calibrated_prototypes()

# 5. Sidebar Specification & Controls
with st.sidebar:
    st.markdown("### 🌿 Specimen Guidelines")
    st.markdown(
        """
        For optimal morphological classification:
        - Ensure the **petal structure** (falls & standards) is clearly visible.
        - Use natural daylight or balanced lighting.
        - Avoid extreme blur or obstructed blooms.
        """
    )
    st.divider()
    st.markdown("### 📊 Supported Taxa")
    st.markdown(
        """
        - **Iris Setosa** *(Bristle-pointed)*
        - **Iris Versicolor** *(Harlequin Blueflag)*
        - **Iris Virginica** *(Southern Blue Flag)*
        """
    )
    st.divider()
    sensitivity = st.slider(
        "Iris Validation Threshold",
        min_value=0.40,
        max_value=0.75,
        value=0.55,
        step=0.05,
        help="Minimum confidence required to confirm the specimen is a valid Iris flower.",
    )

# 6. Hero Header
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">Botanical Vision & Taxonomy System</div>
        <h1 class="hero-title">🌿 Iris Botanical Classifier</h1>
        <p class="hero-subtitle">High-precision automated species identification for <i>Iris setosa</i>, <i>Iris versicolor</i>, and <i>Iris virginica</i></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 7. Workspace Layout
left_col, right_col = st.columns([1, 1.35], gap="large")

with left_col:
    st.markdown("#### 1. Specimen Acquisition")
    input_mode = st.radio(
        "Acquisition Source",
        ["📁 Upload Specimen Image", "📷 Live Optical Capture"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image_source = None
    if input_mode == "📁 Upload Specimen Image":
        uploaded_file = st.file_uploader(
            "Drag and drop or browse a specimen image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            image_source = uploaded_file
    else:
        camera_photo = st.camera_input("Align the Iris bloom within the frame")
        if camera_photo is not None:
            image_source = camera_photo

    if image_source is not None:
        img = Image.open(image_source).convert("RGB")
        st.image(img, caption="Acquired Specimen", use_container_width=True)

with right_col:
    st.markdown("#### 2. Taxonomic Analysis & Results")

    if image_source is None:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 60px 24px;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">🔬</div>
                <h4 style="color: #FFFFFF; margin-bottom: 8px;">Awaiting Specimen Input</h4>
                <p style="color: #CBD5E1; font-size: 0.98rem; max-width: 390px; margin: 0 auto;">
                    Upload a photograph or capture a live image on the left to initiate morphological validation and species classification.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Evaluating petal morphology and venation..."):
            img_embed = get_image_embedding(img)

            # Step A: Domain Validation (Iris vs Non-Iris)
            val_prompts = [
                "a close-up botanical photograph of an iris flower bloom",
                "a photograph of a rose, tulip, sunflower, daisy, or lily flower",
                "a photograph of a human face, person, animal, vehicle, building, or indoor object",
            ]
            val_embeds = get_text_embeddings(val_prompts)
            val_logits = (img_embed @ val_embeds.T)[0] * 100.0
            val_exp = np.exp(val_logits - np.max(val_logits))
            val_probs = val_exp / np.sum(val_exp)
            iris_score = float(val_probs[0])

            if iris_score < sensitivity:
                st.markdown(
                    f"""
                    <div class="result-invalid">
                        <div style="color: #FECACA; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                            Specimen Rejected • Non-Iris Input ({iris_score * 100:.1f}% match)
                        </div>
                        <h2 style="color: #FFFFFF; font-size: 1.8rem; font-weight: 800; margin: 6px 0 10px 0;">
                            ⚠️ Invalid Specimen Image
                        </h2>
                        <p style="color: #FEE2E2; font-size: 1.0rem; margin-bottom: 0;">
                            The uploaded image does not match the botanical structure of an <i>Iris</i> bloom. Please provide a clear photograph of an Iris flower.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Step B: Centroid-Calibrated Multi-Prototype Classification
                spec_logits = (img_embed @ species_prototypes.T)[0] * 18.0
                spec_exp = np.exp(spec_logits - np.max(spec_logits))
                spec_probs = spec_exp / np.sum(spec_exp)

                best_idx = int(np.argmax(spec_probs))
                predicted_species = species_names[best_idx]
                confidence = float(spec_probs[best_idx]) * 100
                info = BOTANICAL_INFO[predicted_species]

                st.markdown(
                    f"""
                    <div class="result-valid">
                        <div class="species-tag">✓ Verified Iris Specimen • {info['scientific']}</div>
                        <div class="species-name">{predicted_species}</div>
                        <div class="confidence-pill">Match Confidence: {confidence:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Probability Distribution Card
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("##### Species Probability Distribution")
                for i, name in enumerate(species_names):
                    pct = float(spec_probs[i]) * 100
                    fill_class = (
                        "prob-fill-top" if i == best_idx else "prob-fill-other"
                    )
                    st.markdown(
                        f"""
                        <div class="prob-row">
                            <div class="prob-header">
                                <span>{name}</span>
                                <span>{pct:.1f}%</span>
                            </div>
                            <div class="prob-track">
                                <div class="{fill_class}" style="width: {pct:.1f}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                # Botanical Reference Card
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <h5 style="margin-top: 0; color: #FFFFFF;">📖 Taxonomic & Morphological Profile</h5>
                        <div class="spec-box">
                            <div class="spec-label">Common Name</div>
                            <div class="spec-value">{info['common_name']}</div>
                        </div>
                        <div class="spec-box">
                            <div class="spec-label">Key Identifying Morphology</div>
                            <div class="spec-value">{info['morphology']}</div>
                        </div>
                        <div class="spec-box">
                            <div class="spec-label">Typical Petal Dimensions</div>
                            <div class="spec-value">{info['avg_dimensions']}</div>
                        </div>
                        <div class="spec-box">
                            <div class="spec-label">Native Habitat</div>
                            <div class="spec-value">{info['habitat']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )