import json
from ..config import llm
from ..prompts.parser_prompt import parse_prompt

def run_llm_parser(query: str):
    """사용자 쿼리를 JSON으로 파싱"""
    try:
        chain = parse_prompt | llm
        ai_response = chain.invoke({"query": query})
        response_text = ai_response.content.strip()

        # JSON 부분만 추출
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].strip()

        parsed = json.loads(response_text)
        return parsed
    except Exception as e:
        return {"error": f"파싱 오류: {str(e)}"}
