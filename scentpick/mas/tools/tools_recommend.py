# --- stdlib ---
import os
import re
from pathlib import Path
from typing import List, Optional
from functools import lru_cache

# --- third-party ---
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from pinecone import Pinecone

# --- langchain ---
from langchain_core.tools import tool


# ======================
# 경로 & 기본 설정
# ======================
BASE_DIR = Path(__file__).resolve().parent.parent / "ml_models"
DEFAULT_MODEL_PATH = BASE_DIR / "models.pkl"

# Pinecone Hosts (환경변수 없으면 기본값 사용)
DEFAULT_PERFUME_HOST = "https://perfume-vectordb2-5h8mu6l.svc.aped-4627-b74a.pinecone.io"
DEFAULT_KEYWORD_HOST = "https://keyword-vectordb-5h8mu6l.svc.aped-4627-b74a.pinecone.io"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# 캐시/싱글톤 유틸
# ======================
@lru_cache()
def get_ml_bundle(model_pkl_path: str):
    return joblib.load(model_pkl_path)

@lru_cache()
def get_hf_encoder(model_name: str):
    """
    메타 텐서 방지:
      - device_map=None, low_cpu_mem_usage=False 로 '실제 가중치' 로드
      - attn_implementation='eager' 로 일부 환경 이슈 회피
      - 로드 후 meta 파라미터 감지 시 즉시 재로드
    """
    tok = AutoTokenizer.from_pretrained(model_name)

    load_kwargs = dict(
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map=None,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    enc = AutoModel.from_pretrained(model_name, **load_kwargs)
    enc.eval()

    # 혹시 meta 파라미터가 섞였으면 재로드
    if any(getattr(p, "is_meta", False) for p in enc.parameters()):
        del enc
        enc = AutoModel.from_pretrained(model_name, **load_kwargs)
        enc.eval()

    if DEVICE == "cuda":
        enc.to(DEVICE, non_blocking=True)

    return tok, enc

@lru_cache()
def get_openai_client(timeout_sec: int = 20):
    return OpenAI(timeout=timeout_sec)  # OPENAI_API_KEY 필요

@lru_cache()
def get_pinecone_index(host: str):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(host=host)


# ======================
# 문자열/토큰 유틸
# ======================
def _normalize_token(s: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", str(s).lower()).strip()

def _split_tokens(text: str) -> List[str]:
    return [t for t in _normalize_token(text).split() if t]

def _unique_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ======================
# 임베딩 유틸
# ======================
def _encode_texts_hf(tokenizer, model, texts: List[str], device: str, max_len: int = 256) -> np.ndarray:
    """HF(mean pooling) -> (n, d) float32"""
    if not texts:
        hidden = getattr(model.config, "hidden_size", 0) or 0
        return np.zeros((0, hidden), dtype=np.float32)
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**batch)
        embs = out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
    return embs

def _embed_openai_1536(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """OpenAI 1536d (cosine 정규화) -> (n,1536)"""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    client = get_openai_client(timeout_sec=20)
    res = client.embeddings.create(model=model, input=texts)
    embs = np.array([d.embedding for d in res.data], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return embs / norms

def _weighted_average(embs: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
    if embs is None or len(embs) == 0:
        return None
    w = np.asarray(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-8)
    v = (embs * w[:, None]).sum(axis=0)
    n = np.linalg.norm(v) + 1e-8
    return (v / n).astype(np.float32)


# ======================
# 키워드 VDB에서 '노트' / '향/재료' 추출
# ======================
_NOTE_PAT = re.compile(r"(?:노트|notes?)\s*[:：]\s*(.+?)(?:\||$)", re.IGNORECASE)
_INGR_PAT = re.compile(r"(?:향/재료|ingredients?)\s*[:：]\s*(.+?)(?:\||$)", re.IGNORECASE)

def _extract_accords_from_keyword_text(text: str, max_terms: int = 4) -> List[str]:
    """
    예: "노트: 파우더리 머스크 | 향/재료: 파우더, 머스크, 화이트머스크, 클린코튼 | ..."
    우선순위: '노트:' → '향/재료:'
    구분자: 콤마/파이프/슬래시/가운뎃점/세미콜론/공백
    """
    accords: List[str] = []

    def _split_candidates(s: str) -> List[str]:
        parts = re.split(r"[,\|/·;]+|\s+", s)
        return [p.strip() for p in parts if p.strip()]

    m1 = _NOTE_PAT.search(text or "")
    if m1:
        accords.extend(_split_candidates(m1.group(1)))

    if not accords:
        m2 = _INGR_PAT.search(text or "")
        if m2:
            accords.extend(_split_candidates(m2.group(1)))

    accords = _unique_preserve(accords)
    return accords[:max_terms] if max_terms > 0 else accords


# ======================
# 추천 도구 (정상 경로 + 키워드 VDB 보완)
# ======================
@tool
def recommend_perfume_vdb(
    user_text: str,
    topk_labels: int = 3,          # 라벨 cap
    top_n_perfumes: int = 3,       # 반환 개수
    use_thresholds: bool = True,   # 임계값 사용
    alpha_labels: float = 0.8,     # (라벨벡터 * alpha + user * (1-alpha))
    model_pkl_path: str = str(DEFAULT_MODEL_PATH),
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    max_len: int = 256,
    index_name: str = "perfume-vectordb2",
    # Fallback 관련
    keyword_index_name: str = "keyword-vectordb",
    keyword_top_k: int = 1,
    fallback_max_terms: int = 4,
):
    """
    정상:
      - HF 임베딩 → 분류기로 라벨/확률 예측(임계값 + topk cap)
      - OpenAI 1536d로 user/labels 임베딩 → 혼합 → perfume-vectordb2 검색
    보완(라벨 실패):
      - user_text 1536d로 keyword-vectordb 질의 → metadata.text에서 어코드 추출
      - 어코드 1536d 평균 → perfume-vectordb2 재검색
    필요 env:
      OPENAI_API_KEY, PINECONE_API_KEY
      (선택) PINECONE_HOST_PERFUME, PINECONE_HOST_KEYWORD
    """
    if not user_text:
        user_text = "(empty)"

    # ===== 1) 분류기/인코더 준비(싱글톤) =====
    data = get_ml_bundle(model_pkl_path)
    clf = data["classifier"]
    mlb = data["mlb"]
    thresholds = data.get("thresholds", {}) or {}

    tok, enc_model = get_hf_encoder(model_name)

    # ===== 2) HF 임베딩 → 라벨 확률 예측 =====
    user_emb_hf = _encode_texts_hf(tok, enc_model, [user_text], DEVICE, max_len=max_len)
    if user_emb_hf.shape[0] == 0:
        raise ValueError("Failed to compute HF embedding for classification.")
    user_vec_hf = user_emb_hf[0]

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(user_vec_hf[None, :])[0]
    elif hasattr(clf, "decision_function"):
        logits = np.asarray(clf.decision_function(user_vec_hf[None, :])[0], dtype=float)
        proba = 1.0 / (1.0 + np.exp(-logits))
    else:
        proba = np.asarray(clf.predict(user_vec_hf[None, :])[0], dtype=float)

    classes = list(mlb.classes_)

    used_fallback = False
    if use_thresholds and thresholds:
        idx_thresh = [i for i, p in enumerate(proba) if p >= float(thresholds.get(classes[i], 0.5))]
        if idx_thresh:
            picked_idx = sorted(idx_thresh, key=lambda i: -proba[i])[:topk_labels]
        else:
            picked_idx = []
            used_fallback = True
    else:
        picked_idx = np.argsort(-proba)[:topk_labels].tolist()

    labels = [classes[i] for i in picked_idx]
    label_probs = [float(proba[i]) for i in picked_idx]

    # ===== 3) OpenAI 1536d 임베딩 생성 =====
    user_emb_vdb = _embed_openai_1536([user_text])[0]  # (1536,)

    PERFUME_HOST = os.environ.get("PINECONE_HOST_PERFUME", DEFAULT_PERFUME_HOST)
    KEYWORD_HOST = os.environ.get("PINECONE_HOST_KEYWORD", DEFAULT_KEYWORD_HOST)
    perfume_index = get_pinecone_index(PERFUME_HOST)
    keyword_index = get_pinecone_index(KEYWORD_HOST)

    # -------------------------------
    # 정상 경로: 라벨 성공
    # -------------------------------
    if not used_fallback and labels:
        label_embs_vdb = _embed_openai_1536(labels) if labels else None
        label_vec = _weighted_average(label_embs_vdb, np.array(label_probs)) if label_embs_vdb is not None else None

        if label_vec is not None:
            v = alpha_labels * label_vec + (1.0 - alpha_labels) * user_emb_vdb
        else:
            v = user_emb_vdb
        v = v / (np.linalg.norm(v) + 1e-8)

        q = perfume_index.query(vector=v.tolist(), top_k=int(top_n_perfumes), include_metadata=True)
        matches = q.get("matches", []) if isinstance(q, dict) else getattr(q, "matches", []) or []
        matches = sorted(matches, key=lambda m: m.get("score", 0.0), reverse=True)[:top_n_perfumes]

        recs = []
        for rnk, m in enumerate(matches, 1):
            meta = m.get("metadata", {}) or {}
            recs.append({
                "rank": int(rnk),
                "id": m.get("id"),
                "score": float(m.get("score", 0.0)),
                "brand": meta.get("brand") or meta.get("Brand") or "N/A",
                "name": meta.get("name_perfume") or meta.get("name") or "N/A",
                "fragrances": (
                    ", ".join(meta.get("fragrances", []))
                    if isinstance(meta.get("fragrances"), list)
                    else (meta.get("fragrances") or meta.get("main_accords") or "N/A")
                ),
                "perfume_data": meta,
            })

        return {
            "user_input": user_text,
            "predicted_labels": labels,
            "label_probs": label_probs,
            "recommendations": recs,
            "meta": {
                "path": "normal",
                "clf_model_name": model_name,
                "device": DEVICE,
                "alpha_labels": float(alpha_labels),
                "topk_labels": int(topk_labels),
                "top_n_perfumes": int(top_n_perfumes),
                "index": index_name,
                "index_metric": "cosine",
                "index_dim": 1536,
                "retrieval": "pinecone_v5(openai-1536, labels+user mix)",
            },
        }

    # -------------------------------
    # Fallback: 라벨 실패 → 키워드 VDB
    # -------------------------------
    kw_q = keyword_index.query(vector=user_emb_vdb.tolist(), top_k=int(keyword_top_k), include_metadata=True)
    kw_matches = kw_q.get("matches", []) if isinstance(kw_q, dict) else getattr(kw_q, "matches", []) or []

    extracted_accords: List[str] = []
    keyword_match_id = None
    keyword_match_score = None
    keyword_match_text = None

    if kw_matches:
        m0 = kw_matches[0]
        keyword_match_id = m0.get("id")
        keyword_match_score = float(m0.get("score", 0.0))
        meta = m0.get("metadata", {}) or {}
        keyword_match_text = meta.get("text") or meta.get("content") or ""
        extracted_accords = _extract_accords_from_keyword_text(keyword_match_text, max_terms=fallback_max_terms)

    if not extracted_accords:
        # 마지막 안전장치: user_text 토큰 일부 사용
        extracted_accords = _split_tokens(user_text)[:fallback_max_terms]

    if extracted_accords:
        acc_embs = _embed_openai_1536(extracted_accords)
        acc_vec = acc_embs.mean(axis=0)
        acc_vec = acc_vec / (np.linalg.norm(acc_vec) + 1e-8)
    else:
        acc_vec = user_emb_vdb

    pq = perfume_index.query(vector=acc_vec.tolist(), top_k=int(top_n_perfumes), include_metadata=True)
    pmatches = pq.get("matches", []) if isinstance(pq, dict) else getattr(pq, "matches", []) or []
    pmatches = sorted(pmatches, key=lambda m: m.get("score", 0.0), reverse=True)[:top_n_perfumes]

    precs = []
    for rnk, m in enumerate(pmatches, 1):
        meta = m.get("metadata", {}) or {}
        precs.append({
            "rank": int(rnk),
            "id": m.get("id"),
            "score": float(m.get("score", 0.0)),
            "brand": meta.get("brand") or meta.get("Brand") or "N/A",
            "name": meta.get("name_perfume") or meta.get("name") or "N/A",
            "fragrances": (
                ", ".join(meta.get("fragrances", []))
                if isinstance(meta.get("fragrances"), list)
                else (meta.get("fragrances") or meta.get("main_accords") or "N/A")
            ),
            "perfume_data": meta,
        })

    return {
        "user_input": user_text,
        "predicted_labels": [],  # 실패
        "label_probs": [],
        "recommendations": precs,
        "meta": {
            "path": "fallback_keyword_vdb",
            "keyword_top_k": int(keyword_top_k),
            "extracted_accords": extracted_accords,
            "keyword_match_id": keyword_match_id,
            "keyword_match_score": keyword_match_score,
            "keyword_match_text": keyword_match_text,
            "clf_model_name": model_name,
            "device": DEVICE,
            "top_n_perfumes": int(top_n_perfumes),
            "index": index_name,
            "index_metric": "cosine",
            "index_dim": 1536,
            "retrieval": "keyword_vdb(notes)->perfume_vdb",
        },
    }


# ======================
# 서버 기동시 웜업(선택)
# ======================
def warmup_recommender():
    """서버 시작 시 한 번 호출해서 모델/클라이언트/인덱스 예열 + meta 방지 재로드"""
    # HF 인코더 캐시 초기화 후 재로드(메타 상태 제거)
    try:
        get_hf_encoder.cache_clear()
    except Exception:
        pass
    _ = get_hf_encoder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    _ = get_ml_bundle(str(DEFAULT_MODEL_PATH))
    _ = get_openai_client(timeout_sec=20)

    perfume_host = os.environ.get("PINECONE_HOST_PERFUME", DEFAULT_PERFUME_HOST)
    keyword_host = os.environ.get("PINECONE_HOST_KEYWORD", DEFAULT_KEYWORD_HOST)
    _ = get_pinecone_index(perfume_host)
    _ = get_pinecone_index(keyword_host)

    try:
        _ = get_openai_client().embeddings.create(model="text-embedding-3-small", input=["warmup"])
    except Exception as e:
        print("[warmup] OpenAI warmup failed:", e)
