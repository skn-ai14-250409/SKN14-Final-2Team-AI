# scentpick/mas/nodes/utils.py
import re
from typing import Optional, List
from langchain_core.messages import HumanMessage, BaseMessage

## 메타 질문처리를 위한 함수와 변수들

# “직전에 내가 뭐라 했지?” 감지
_MEMORY_PATTERNS = [
    r"(내가\s*)?(방금|지금)\s*뭐(라|라고)?\s*(했|말했)(지|어)\??",
    r"방금\s*내가\s*한\s*(말|질문)\s*뭐(였|였지|였어요)\??",
    r"마지막\s*(질문|말)\s*뭐(였|였지|였어요)\??",
    r"what\s+did\s+i\s+just\s+(ask|say)\??",
    r"what\s+was\s+my\s+last\s+(question|message)\??",
]
# “방금 추천한 향수 뭐였지?” 감지
_REC_PATTERNS = [
    r"(방금|지금)\s*추천(해준|한)\s*(향수|제품)\s*뭐였지\??",
    r"(방금|지금)\s*추천\s*다시\s*보여줘",
    r"추천\s*다시\s*보여줘",
    r"(마지막|직전)\s*추천\s*(목록|리스트)\s*보여줘",
    r"(what|which)\s+(perfumes?|items?)\s+did\s+you\s+(just\s+)?recommend\??",
    r"(last|previous)\s+recommendations?\??",
]

_memory_regex = re.compile("|".join(_MEMORY_PATTERNS), re.IGNORECASE)
_rec_regex = re.compile("|".join(_REC_PATTERNS), re.IGNORECASE)

def is_memory_check(text: str) -> bool:
    return bool(_memory_regex.search(text or ""))

def is_rec_check(text: str) -> bool:
    return bool(_rec_regex.search(text or ""))

def get_prev_user_utterance(messages: List[BaseMessage]) -> Optional[str]:
    humans = [m.content for m in messages if isinstance(m, HumanMessage)]
    if len(humans) >= 2:
        return humans[-2]
    return None
