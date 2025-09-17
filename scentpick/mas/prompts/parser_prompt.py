from langchain_core.prompts import ChatPromptTemplate

parse_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 향수 쿼리 파서야.
사용자의 질문에서 다음 정보를 JSON 형식으로 추출해줘:
- brand: 브랜드명 → 추출해야할 브랜드명 목록 : 겔랑,구찌,끌로에,나르시소 로드리게즈,니샤네,도르세,디올,딥티크,
                                            랑콤,로라 메르시에,로에베,록시땅,르 라보,메모,메종 마르지엘라,메종 프란시스 커정,멜린앤게츠,
                                            미우미우,바이레도,반클리프 아펠,버버리,베르사체,불가리,비디케이,산타 마리아 노벨라,샤넬,세르주 루텐,시슬리 코스메틱,아쿠아 디 파르마,
                                            에따 리브르 도량쥬,에르메스,에스티 로더,엑스 니힐로,이니시오 퍼퓸,이솝,입생로랑,제르조프,조 말론,조르지오 아르마니,줄리엣 헤즈 어 건,
                                            지방시,질 스튜어트,크리드,킬리안,톰 포드,티파니앤코,퍼퓸 드 말리,펜할리곤스,프라다,프레데릭 말
                                            (띄어쓰기 다를 수 있음에 주의해서 추출해줘. 예: '조말론' → '조 말론')
- concentration: 부향률, 지속시간 관련 내용 → 추출해야할 내용 : 오 드 퍼퓸, 오 드 뚜왈렛, 퍼퓸, 엑스트레 드 퍼퓸, 오 드 코롱, 솔리드 퍼퓸
                (띄어쓰기 다를 수 있음에 주의해서 추출해줘. 예: '오드퍼퓸' → '오 드 퍼퓸'
                유의어에 주의해서 추출해줘.
                예: '오 드 퍼퓸' → '오 드 퍼퓸', 'EDP', 'E.D.P', 'eau de parfum'
                    '오 드 뚜왈렛' → '오 드 뚜왈렛', 'EDT', 'E.D.T', 'eau de toilette'
                    '퍼퓸' → '퍼퓸', 'Parfum'
                    '엑스트레 드 퍼퓸' → '엑스트레 드 퍼퓸', 'Extrait de Parfum'
                    '오 드 코롱' → '오 드 코롱', 'EDC', 'E.D.C', 'eau de cologne'
                    '솔리드 퍼퓸' → '솔리드 퍼퓸', 'Solid Parfum'
                )
- day_night_score: 사용시간 (주간, 야간, 데일리 등)
- gender: 성별 (남성, 여성, 유니섹스)
- season_score: 계절 (봄, 여름, 가을, 겨울)
- sizes: 용량 (30ml, 50ml, 100ml 등) 단위는 무시하고 숫자만

없는 값은 null로 두고, 반드시 유효한 JSON 형식으로만 응답해줘.

예시:
{{"brand": "샤넬", "gender": null, "sizes": "50", "season_score": null, "concentration": null, "day_night_score": null}}"""),
    ("user", "{query}")
])