# --- stdlib ---
import json
from pathlib import Path

# --- third-party ---
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

# --- langchain ---
from langchain_core.tools import tool


# ===== 경로 설정 =====
BASE_DIR = Path(__file__).resolve().parent.parent / "ml_models"  # ✅ 한 단계 더 올라가야 함
DEFAULT_MODEL_PATH = BASE_DIR / "models.pkl"
DEFAULT_PERFUME_JSON = BASE_DIR / "perfumes.json"

@tool
def recommend_perfume_simple(
    user_text: str,
    topk_labels: int = 4,
    top_n_perfumes: int = 5,
    use_thresholds: bool = True,
    # model_pkl_path: str = "./models.pkl",
    # perfume_json_path: str = "perfumes.json",
    model_pkl_path: str = str(DEFAULT_MODEL_PATH),      # 수정
    perfume_json_path: str = str(DEFAULT_PERFUME_JSON), # 수정

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    max_len: int = 256,
):
    """
    Minimal one-shot perfume recommender (no caching).
    Loads model & data every call, predicts labels, retrieves with BM25, and returns a JSON-serializable dict.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load ML bundle
    data = joblib.load(model_pkl_path)
    clf = data["classifier"]
    mlb = data["mlb"]
    thresholds = data.get("thresholds", {}) or {}

    # 2) Encoder
    tok = AutoTokenizer.from_pretrained(model_name)
    enc_model = AutoModel.from_pretrained(model_name).to(device)
    enc_model.eval()

    # 3) Load perfumes
    with open(perfume_json_path, "r", encoding="utf-8") as f:
        perfumes = json.load(f)
        if not isinstance(perfumes, list):
            raise ValueError("perfumes.json must contain a list of perfume objects")

    # 4) BM25 index (simple: use 'fragrances' or fallback to description/name/brand)
    def doc_of(p):
        fr = p.get("fragrances")
        if isinstance(fr, list):
            text = " ".join(map(str, fr))
        elif isinstance(fr, str):
            text = fr
        else:
            text = " ".join(
                str(x)
                for x in [
                    p.get("description", ""),
                    p.get("main_accords", ""),
                    p.get("name_perfume") or p.get("name", ""),
                    p.get("brand", ""),
                ]
                if x
            )
        return (text or "unknown").lower()

    tokenized_corpus = [doc_of(p).split() for p in perfumes]
    bm25 = BM25Okapi(tokenized_corpus)

    # 5) Encode text -> vector
    batch = tok(
        [user_text],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = enc_model(**batch)
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy()  # ultra-simple mean

    # 6) Predict labels (supports predict_proba / decision_function / predict)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(emb)[0]
    elif hasattr(clf, "decision_function"):
        logits = np.asarray(clf.decision_function(emb)[0], dtype=float)
        proba = 1.0 / (1.0 + np.exp(-logits))
    else:
        proba = np.asarray(clf.predict(emb)[0], dtype=float)

    classes = list(mlb.classes_)
    if use_thresholds and thresholds:
        picked_idx = [i for i, p in enumerate(proba) if p >= float(thresholds.get(classes[i], 0.5))]
        if not picked_idx:
            picked_idx = np.argsort(-proba)[:topk_labels].tolist()
    else:
        picked_idx = np.argsort(-proba)[:topk_labels].tolist()

    labels = [classes[i] for i in picked_idx]

    # 7) Retrieve with BM25
    scores = bm25.get_scores(" ".join(labels).split())
    top_idx = np.argsort(scores)[-top_n_perfumes:][::-1]

    def _safe(d, *keys, default="N/A"):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return d[k]
        return default

    recs = []
    for rnk, idx in enumerate(top_idx, 1):
        p = perfumes[int(idx)]
        fr = p.get("fragrances")
        if isinstance(fr, list):
            fr_text = ", ".join(map(str, fr))
        else:
            fr_text = fr if isinstance(fr, str) else _safe(p, "main_accords", default="N/A")
        recs.append({
            "rank": int(rnk),
            "index": int(idx),
            "score": float(scores[int(idx)]),
            "brand": _safe(p, "brand"),
            "name": _safe(p, "name_perfume", "name"),
            "fragrances": fr_text,
            "perfume_data": p,  # JSON-native
        })

    return {
        "user_input": user_text,
        "predicted_labels": labels,
        "recommendations": recs,
        "meta": {
            "model_name": model_name,
            "device": device,
            "max_len": int(max_len),
            "db_size": int(len(perfumes)),
        },
    }