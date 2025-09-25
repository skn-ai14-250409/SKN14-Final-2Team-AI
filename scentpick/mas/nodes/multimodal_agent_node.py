from langchain_core.messages import AIMessage, HumanMessage
from ..state import AgentState
from ..tools.tools_rag import query_pinecone, generate_response
from ..tools.tools_metafilters import apply_meta_filters
from ..config import llm, embeddings
from openai import OpenAI
from datetime import datetime, timezone
import json

client = OpenAI()

def multimodal_agent_node(state: AgentState) -> AgentState:
    img_url = state.get("image_url")
    if not img_url:
        return {"messages": [AIMessage(content="이미지를 첨부해주세요.")], "next": None}

    # 1) 최신 user query (텍스트)
    query = ""

    if not query and img_url:
        query = "이미지 기반 추천 요청"

    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            query = m.content or ""
            break

    # 2) GPT-4o-mini로 이미지 분석 (JSON 형식 강제)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """
            Analyze the uploaded perfume-related image. 
            Output STRICT JSON with the following keys if detectable:
            - day_night_score (day or night)
            - gender (Male, Female, Unisex)
            - season_score (spring, summer, fall, winter)
            - free_text (short natural language summary: mood, color impression, style, etc.)
            """},
            {"role": "user", "content": [
                {"type": "text", "text": query or "이 이미지에서 향수 추천에 도움이 되는 단서를 추출해줘."},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]}
        ],
        max_tokens=400,
    )

    raw_analysis = response.choices[0].message.content

    try:
        analysis_json = json.loads(raw_analysis)
    except Exception:
        # JSON 파싱 실패 시 free_text에 전체 메시지 넣기
        analysis_json = {"free_text": raw_analysis}

    # 3) 메타필터 적용
    filtered_json = apply_meta_filters(analysis_json)

    print(f"🔍 multimodal_agent_node 이미지 분석 결과: {json.dumps(analysis_json, ensure_ascii=False)}")

    # 4) Pinecone 검색 (free_text는 벡터 검색, 나머지는 메타필터)
    query_text = analysis_json.get("free_text", query)
    query_vector = embeddings.embed_query(query_text)
    search_results = query_pinecone(query_vector, filtered_json, top_k=3)
    if hasattr(search_results, "to_dict"):
        search_results = search_results.to_dict()

    # 5) LLM으로 최종 추천 답변 생성
    final_response = generate_response(
        original_query=f"{query} + 이미지 분석: {json.dumps(analysis_json, ensure_ascii=False)}",
        search_results=search_results,
        limit=3
    )

    # 6) perfume_list 정규화
    perfume_list = []
    for match in search_results.get("matches", []):
        meta = match.get("metadata", {}) or {}
        pid = meta.get("no")   # ✅ DB용 정수 id
        try:
            pid = int(pid) if pid is not None else None
        except Exception:
            pid = None

        if pid:
            perfume_list.append({
                "id": pid,
                "brand": meta.get("brand"),
                "name": meta.get("name"),
                "score": match.get("score"),
                "text": meta.get("text"),
            })

    # 7) history entry (rec_echo 용)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "multimodal_agent",
        "items": search_results.get("matches", []),
    }

    return {
        "messages": [AIMessage(content=final_response)],
        "search_results": search_results,
        "final_answer": final_response,
        "rec_history": [entry],
        "last_agent": "multimodal_agent",
        "perfume_list": perfume_list,
    }