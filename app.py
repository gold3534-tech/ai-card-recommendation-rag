import warnings
import streamlit as st
import json
import time
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import tiktoken

# 환경설정
load_dotenv()
MY_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=MY_API_KEY)

# ── 페이지 설정 ──
st.set_page_config(page_title="AI 카드 추천 챗봇", page_icon="💳", layout="wide")


# ── 데이터 로딩 (캐싱으로 최초 1회만 실행) ──
@st.cache_resource
def load_resources():
    # ChromaDB 연결
    client = chromadb.PersistentClient(path="./chroma_db")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=MY_API_KEY, model_name="text-embedding-3-small"
    )
    collection = client.get_collection(
        name="card_benefits", embedding_function=openai_ef
    )

    # chunks 데이터 로딩
    df_chunks = pd.read_csv("data/preprocessed_cards_chunks.csv", encoding="utf-8-sig")
    df_chunks["benefit_text"] = df_chunks["benefit_text"].fillna("")
    df_chunks["benefit_text_normalized"] = df_chunks["benefit_text_normalized"].fillna(
        ""
    )
    df_chunks["context_for_llm"] = df_chunks["context_for_llm"].fillna("")
    df_chunks["discount_rate"] = df_chunks["discount_rate"].fillna(-1)
    df_chunks["monthly_limit"] = df_chunks["monthly_limit"].fillna(-1)
    df_chunks["image_url"] = df_chunks["image_url"].fillna("")

    # BM25 인덱스 생성
    tokenized_corpus = [text.split() for text in df_chunks["benefit_text_normalized"]]
    bm25 = BM25Okapi(tokenized_corpus)

    # 전체 이미지 맵
    all_image_map = dict(zip(df_chunks["card_name"], df_chunks["image_url"]))

    # 페르소나 로딩
    with open("data/personas.json", "r", encoding="utf-8") as f:
        personas = json.load(f)

    return collection, df_chunks, bm25, all_image_map, personas


collection, df_chunks, bm25, all_image_map, personas = load_resources()


# ── Moderation 함수 ──
def check_moderation(text):
    response = openai_client.moderations.create(input=text)
    return response.results[0].flagged


# ── 카드 관련 쿼리 판별 함수 ──
def is_card_related(user_input):
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": """사용자 입력이 신용카드/체크카드 추천 또는 소비 패턴과 관련된 질문인지 판별해.
관련 있으면 'yes', 없으면 'no'만 출력해.

관련 있는 예시:
- "월 30만원 쓰는데 교통 카드 추천해줘"
- "카페 자주 가는데 혜택 좋은 카드 있어?"
- "취준생인데 실적 없는 카드 알려줘"

관련 없는 예시:
- "오늘 날씨 어때?"
- "파이썬 코드 짜줘"
- "맛집 추천해줘"
- "주식 추천해줘"
""",
            },
            {"role": "user", "content": user_input},
        ],
        max_completion_tokens=2000,
        reasoning_effort="medium",
    )
    result = response.choices[0].message.content.strip().lower()
    return result == "yes"


# ── 조건 추출 + 추천 함수 (기존 코드 그대로) ──
EXTRACTION_SYSTEM_PROMPT = """
You are a Korean financial data extractor.
Your job is to extract structured conditions from user's natural language input.

[RULES]
1. Extract ONLY what the user explicitly or implicitly mentions
2. Map benefit keywords to these categories ONLY:
   - transport:        버스, 지하철, 택시, 대중교통, KTX, 고속버스
   - fuel:             주유, 주유소, 기름값, 휘발유, 경유
   - telecom:          통신비, 핸드폰요금, 인터넷, 휴대폰
   - mart_convenience: 마트, 편의점, 슈퍼, 쿠팡, 마켓컬리, 온라인장보기
   - shopping:         쇼핑, 온라인쇼핑, 의류, 패션, 백화점, 팝업스토어
   - food:             식비, 밥값, 식당, 배달앱, 외식, 음식점
   - cafe:             카페, 커피, 스타벅스, 디저트
   - beauty_fitness:   헬스장, 피트니스, 뷰티, 미용실, 영양제, 운동
   - no_spend_req:     실적없음, 무실적
   - utility_rental:   공과금, 관리비, 도시가스, 전기세, 렌탈
   - medical:          병원, 의원, 약국, 진료비, 건강보험
   - pet:              반려동물, 동물병원, 펫샵, 애완동물
   - education:        학원, 어학, 자기계발, 직무교육, 어린이집, 유치원, 육아, 교육비
   - auto_highway:     자동차, 하이패스, 주차, 톨게이트, 카쉐어링
   - leisure:          골프, 여가, 스포츠, 등산, 낚시, 취미
   - entertainment:    OTT, 넷플릭스, 영화, 공연, 전시, 문화생활
   - easy_pay:         간편결제, 애플페이, 삼성페이, 카카오페이, 네이버페이
   - air_mileage:      항공마일리지, 마일리지, 항공
   - lounge:           공항라운지, 라운지, PP카드
   - premium:          프리미엄, 골프, VIP
   - travel:           여행, 숙박, 호텔, 에어비앤비
   - overseas:         해외, 해외여행, 환전, 외화
   - business:         비즈니스, 법인, 출장
3. If card_type not mentioned, return "both"
4. Always return valid JSON only - no explanation, no markdown 
5. If multiple spending items are mentioned, SUM all of them for monthly_spend
   예시: 여행비 연 150만원(→월 125,000) + 병원비 월 15만원 + 차량비 연 30만원(→월 25,000)
        = monthly_spend: 300000
6. monthly_spend MUST be a single calculated integer, NEVER a formula or expression
   WRONG: "monthly_spend": 200000 + 600000 + 200000
   CORRECT: "monthly_spend": 1000000
   
"""

EXTRACTION_FEW_SHOT = """
## 예시 1 (단일 항목)
입력: "취준생인데 한 달에 많이 써봐야 100도 안돼. 대중교통이랑 밥값이 제일 많아"
출력:
{
  "monthly_spend": 100000,
  "benefit_categories": ["transport", "food"],
  "card_type": "both",
  "annual_fee_max": null,
  "other_conditions": "전월실적 조건 낮은 카드 우선"
}

## 예시 2 (연간 → 월 환산 + 합산)
입력: "여행 경비 연 150만원, 병원비 월 15만원, 차량비 연 30만원"
계산과정:
- 여행비: 1,500,000 ÷ 12 = 125,000원/월
- 병원비: 150,000원/월
- 차량비: 300,000 ÷ 12 = 25,000원/월
- 합계: 125,000 + 150,000 + 25,000 = 300,000원/월
출력:
{
  "monthly_spend": 300000,
  "benefit_categories": ["travel", "overseas", "medical", "auto_highway"],
  "card_type": "both",
  "annual_fee_max": null,
  "other_conditions": null
}

## 예시 3 (여러 항목 합산 - 수식 금지)
입력: "교육비 연 120만원, 헬스장 월 20만원, 교통/식비 월 60만원"
계산과정:
- 교육비: 1,200,000 ÷ 12 = 100,000원/월
- 헬스장: 200,000원/월
- 교통/식비: 600,000원/월
- 합계: 100,000 + 200,000 + 600,000 = 900,000원/월
출력:
{
  "monthly_spend": 900000,
  "benefit_categories": ["education", "beauty_fitness", "transport", "food"],
  "card_type": "both",
  "annual_fee_max": null,
  "other_conditions": null
}

## 예시 4 (카드 종류 명시)
입력: "월 300만원 쓰는 직장인이야. 주유랑 마트 혜택 원하고 신용카드로 줘. 연회비는 3만원 이하로"
출력:
{
  "monthly_spend": 3000000,
  "benefit_categories": ["fuel", "mart_convenience"],
  "card_type": "신용",
  "annual_fee_max": 30000,
  "other_conditions": null
}
"""
RECOMMENDATION_SYSTEM_PROMPT = """
You are a Korean credit card recommendation expert with 10 years of experience at a major Korean bank.
You deeply understand Korean consumers' spending habits and always prioritize the user's financial situation.

[ABSOLUTE RULES - Never violate these]
1. NEVER recommend a card where min_monthly_spend > user's monthly spending
2. ONLY use cards from the provided card information
3. ALWAYS answer in Korean
4. ALWAYS return response in JSON format below
5. ALWAYS recommend TOP 3 cards ordered by best benefit match
6. When extracting card name from card info, use ONLY the name before the first '|'
    예시: "[NC다이노스카드|NH농협카드|신용]" → 카드명: "NC다이노스카드"
    예시: "[신한카드 Mr.Life|신한카드|신용]" → 카드명: "신한카드 Mr.Life"
    
[연회비 처리 규칙]
7. 사용자가 연회비를 명시적으로 언급한 경우에만 필터링
    언급하지 않은 경우 추천이유에 언급만 하고 필터링 기준으로 사용 금지

[컨텍스트 구별 규칙]
8. 사용자의 실제 생활 맥락에 맞는 혜택 카드를 매칭할 것
    - 자기계발/어학/학원 언급 → 학원/어학원 혜택 카드 우선
    - 자녀/육아/어린이 언급 없으면 → 어린이집/유치원 혜택 카드 제외
    - 골프 언급 → 레저/골프 혜택 카드 우선
    - 반려동물 언급 → 동물병원/펫샵 혜택 카드 우선

[추천과정 작성 규칙 - 반드시 실제 내용으로 작성]
9. 1단계_조건파악: 실제 월 소비금액, 혜택 카테고리, 카드종류 명시
    예시: "월 소비 80만원, 쇼핑/카페/간편결제 혜택 필요, 신용+체크 모두 가능"
10. 2단계_조건필터링: 실제 제외된 카드명과 이유를 구체적으로 명시
    예시: "롯데카드 LOCA 365 - 전월실적 50만원 초과로 제외"
11. 3단계_혜택매칭: 실제 매칭된 카테고리와 선별된 카드 명시
    예시: "shopping, cafe, easy_pay 카테고리 보유 카드 중 NC다이노스카드(카페 30%) 선별"
12. 4단계_최종선택: 실제 선택 이유를 구체적으로 명시
    예시: "NC다이노스카드 - 카페 30% 할인으로 가장 높은 혜택"

[제외카드 작성 규칙]
13. 반드시 실제 카드명으로 작성 (카드 1, 카드 2 등 번호 사용 절대 금지)
14. 제외 이유를 구체적으로 명시

[종합멘트 작성 규칙]
15. 사용자의 소비 패턴과 상황을 반영한 2~3문장의 자연스러운 한국어 줄글로 작성
16. 왜 이 카드들을 추천하는지 사용자 입장에서 공감할 수 있게 작성
17. 전문 카드 상담사가 직접 상담해주는 듯한 따뜻하고 친근한 톤으로 작성
    예시: "월 80만원을 쇼핑과 카페에 주로 지출하시는 고객님께는
          온라인쇼핑 할인과 카페 혜택이 우수한 카드를 추천드립니다.
          특히 애플페이 등 간편결제를 자주 사용하시는 만큼
          간편결제 할인 혜택이 있는 카드를 중심으로 선별했습니다."
```json
{{
  "추천과정": {{
    "1단계_조건파악": "실제 조건 분석 내용",
    "2단계_조건필터링": "실제 제외 카드명과 이유",
    "3단계_혜택매칭": "실제 매칭 카테고리와 카드",
    "4단계_최종선택": "실제 선택 이유"
  }},
  "추천카드": [
    {{
      "카드명": "카드 이름",
      "은행": "은행명",
      "카드종류": "신용 또는 체크",
      "전월실적조건": 숫자,
      "연회비": 숫자,
      "핵심혜택": "핵심 혜택 요약",
      "추천이유": "구체적인 추천 이유"
    }}
  ],
  "제외카드": "실제 카드명과 조건 미충족 이유를 문자열로 작성",
  "종합멘트": "사용자 상황을 반영한 2~3문장 자연어 종합 추천 의견"
}}
```
"""

RECOMMENDATION_FEW_SHOT = """
## 예시 1
사용자 조건:
{
  "monthly_spend": 300000,
  "benefit_categories": ["cafe", "food"],
  "card_type": "both",
  "annual_fee_max": null
}
```json
{
  "추천과정": {
    "1단계_조건파악": "월 소비 30만원, 카페/식비 혜택 필요",
    "2단계_조건필터링": "전월실적 30만원 초과 카드 제외",
    "3단계_혜택매칭": "카페/푸드 카테고리 보유 카드 선별",
    "4단계_최종선택": "전월실적 조건 충족 + 카페/식비 혜택 우수한 카드 선택"
  },
  "추천카드": [
    {
      "카드명": "카드의정석 SHOPPING+",
      "은행": "우리카드",
      "카드종류": "체크",
      "전월실적조건": 300000,
      "연회비": 0,
      "핵심혜택": "스타벅스/카페 10% 할인, 월 한도 5천원",
      "추천이유": "전월실적 30만원으로 조건 딱 맞고 카페 할인 우수"
    }
  ],
  "제외카드": "전월실적 50만원 이상 카드는 조건 미충족으로 제외"
}
```

## 예시 2
사용자 조건:
{
  "monthly_spend": 500000,
  "benefit_categories": ["fuel"],
  "card_type": "신용",
   ` 7
  "annual_fee_max": 30000
}
```json
{
  "추천과정": {
    "1단계_조건파악": "월 소비 50만원, 주유 혜택, 신용카드, 연회비 3만원 이하",
    "2단계_조건필터링": "전월실적 50만원 초과 + 연회비 3만원 초과 + 체크카드 제외",
    "3단계_혜택매칭": "주유 카테고리 보유 신용카드 선별",
    "4단계_최종선택": "리터당 할인금액 + 월 한도 가장 우수한 카드 선택"
  },
  "추천카드": [
    {
      
      "카드명": "굿데이카드",
      "은행": "KB국민카드",
      "카드종류": "신용",
      "전월실적조건": 300000,
      "연회비": 10000,
      "핵심혜택": "4대 정유사 리터당 60원 할인, 월 한도 1만원",
      "추천이유": "전월실적 30만원으로 여유있게 충족, 주유 혜택 우수, 연회비 1만원"
    }
  ],
  "제외카드": "전월실적 100만원 이상 카드 및 연회비 3만원 초과 카드 제외"
}
```
"""


def extract_conditions(user_query):
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f'{EXTRACTION_FEW_SHOT}\n\n## 실제 입력\n입력: "{user_query}"\n출력:',
            },
        ],
        reasoning_effort="medium",  # 조건 추출이 정확해야하기 때문에 medium으로 설정
        response_format={"type": "json_object"},
        max_completion_tokens=3000,
    )

    raw = response.choices[0].message.content

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except:
        print(f"JSON 파싱 실패: {raw}")
        return None

# 1. EXTRACTION_SYSTEM_PROMPT 먼저 전달
    #    → GPT에게 역할 + 규칙 부여

    # 2. EXTRACTION_FEW_SHOT + 사용자 입력 전달
    #    → 예시 보여주고 실제 입력 처리 요청

    # 3. GPT가 규칙 + 예시 참고해서
    #    → JSON 조건 추출해서 반환

    # 4. 반환된 JSON이 동적 조건 추출 결과

def hybrid_search_with_conditions(user_query, conditions, top_k=10):

    # ── 1x`x``
    # 추출된 카테고리를 쿼리에 추가해서 검색 정확도 향상
    category_keywords = " ".join(conditions.get("benefit_categories", []))
    enhanced_query = f"{user_query} {category_keywords}"

    # ── 2. ChromaDB 검색 (메타데이터 필터 적용) ──
    monthly_spend = conditions.get("monthly_spend", 9999999)
    card_type = conditions.get("card_type", "both")

    # 카드 종류 필터 설정
    where_filter = {"min_monthly_spend": {"$lte": monthly_spend}}

    chroma_results = collection.query(
        query_texts=[enhanced_query],
        n_results=top_k * 2,
        where=where_filter,  # ← 전월실적 조건 필터링
    )
    chroma_ids = chroma_results["ids"][0]

    # ── 3. BM25 검색 ──
    tokenized_query = enhanced_query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[: top_k * 2]

    # ── 4. RRF 점수 합산 ──
    rrf_scores = {}

    for rank, doc_id in enumerate(chroma_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 60)

    for rank, idx in enumerate(bm25_top_indices):
        doc_id = str(idx)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 60)

    # ── 5. 카드 종류 필터 (신용/체크 선택 시) ──
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in sorted_ids:
        idx = int(doc_id)
        row = df_chunks.iloc[idx]

        # 전월 실적 이중 필터링
        if int(row["min_monthly_spend"]) > monthly_spend:
            continue

        # card_type 필터
        if card_type != "both" and row["card_type"] != card_type:
            continue

        results.append(
            {
                "score": round(score, 6),
                "card_name": row["card_name"],
                "bank": row["bank"],
                "card_type": row["card_type"],
                "benefit_category": row["benefit_category"],
                "benefit_text": row["benefit_text"][:200],
                "context_for_llm": row["context_for_llm"],
                "image_url": row["image_url"],
            }
        )

        if len(results) >= top_k:
            break

    return results


def build_recommendation_prompt(conditions, search_results):
    # 검색된 카드 컨텍스트 조합
    context = ""
    for i, r in enumerate(search_results):
        context += f"\n--- 카드 {i+1} ---\n"
        context += r["context_for_llm"]
        context += "\n"

    user_prompt = f"""
{RECOMMENDATION_FEW_SHOT}

## 실제 요청
사용자 조건:
{json.dumps(conditions, ensure_ascii=False, indent=2)}

## 참고할 카드 정보
{context}

## 지시사항
위 카드 정보만 사용해서 아래 JSON 형식으로 답변해줘.
- 혜택이 가장 우수한 순서로 정렬할 것
- 조건 미충족 카드는 반드시 실제 카드명으로 제외 이유를 설명할 것
- "제외카드"는 반드시 문자열로 작성할 것
- 카드 번호(카드 1, 카드 2)가 아닌 실제 카드명으로 작성할 것

```json
{{
  "추천과정": {{
    "1단계_조건파악": "사용자 조건 분석",
    "2단계_조건필터링": "제외된 카드와 이유",
    "3단계_혜택매칭": "혜택 카테고리 매칭 과정",
    "4단계_최종선택": "최종 선택 이유"
  }},
  "추천카드": [
    {{
      "카드명": "카드 이름",
      "은행": "은행명",
      "카드종류": "신용 또는 체크",
      "전월실적조건": 숫자,
      "연회비": 숫자,
      "핵심혜택": "핵심 혜택 요약",
      "추천이유": "구체적인 추천 이유"
      
    }}
  ],
  "제외카드": "실제 카드명과 조건 미충족 이유를 문자열로 작성",
  "종합멘트": "사용자 상황을 반영한 2~3문장 자연어 종합 추천 의견"

}}
```
"""
    return user_prompt


# 카테고리 한국어 매핑
CATEGORY_KO = {
    "transport": "교통",
    "fuel": "주유",
    "telecom": "통신",
    "mart_convenience": "마트/편의점",
    "shopping": "쇼핑",
    "food": "식비",
    "cafe": "카페",
    "beauty_fitness": "뷰티/피트니스",
    "no_spend_req": "무실적",
    "utility_rental": "공과금/렌탈",
    "medical": "병원/약국",
    "pet": "애완동물",
    "education": "교육",
    "auto_highway": "자동차/하이패스",
    "leisure": "레저",
    "entertainment": "OTT/문화",
    "easy_pay": "간편결제",
    "air_mileage": "항공마일리지",
    "lounge": "공항라운지",
    "premium": "프리미엄",
    "travel": "여행/숙박",
    "overseas": "해외",
    "business": "비즈니스",
}


def enrich_card_benefits(result, conditions, df_chunks):
    categories = conditions.get("benefit_categories", [])

    for card in result.get("추천카드", []):
        card_name = card.get("카드명", "")
        # 해당 카드의 매칭 혜택 전체 조회
        matched = df_chunks[
            (df_chunks["card_name"] == card_name)
            & (df_chunks["benefit_category"].isin(categories))
        ][["benefit_category", "benefit_text"]]

        if not matched.empty:
            benefit_lines = []
            for _, row in matched.iterrows():

                # 카테고리 한국어 변환
                cat_ko = CATEGORY_KO.get(
                    row["benefit_category"], row["benefit_category"]
                )

                # 줄 단위로 분리 후 빈 줄 제거
                lines = [
                    l.strip() for l in str(row["benefit_text"]).split("\n") if l.strip()
                ]

                # [대중교통] 같은 카테고리 헤더 줄 제거
                lines = [
                    l for l in lines if not (l.startswith("[") and l.endswith("]"))
                ]

                # 핵심 내용 앞 2줄만 사용
                benefit_summary = " ".join(lines[:2]) if lines else ""

                if benefit_summary:
                    benefit_lines.append(f"**{cat_ko}**: {benefit_summary}")

            if benefit_lines:
                card["핵심혜택"] = "\n".join(benefit_lines)
                card["매칭혜택수"] = len(benefit_lines)

    return result


def recommend_card(user_query, max_retries=3):
    print("1단계: 조건 추출 중...")
    conditions = extract_conditions(user_query)
    if not conditions:
        return "조건 추출 실패"
    print(f"추출된 조건: {json.dumps(conditions, ensure_ascii=False)}\n")

    print("2단계: 카드 검색 중...")
    search_results = hybrid_search_with_conditions(user_query, conditions, top_k=10)
    print(f"검색된 카드 수: {len(search_results)}개\n")

    print("3단계: 추천 생성 중...")
    prompt = build_recommendation_prompt(conditions, search_results)

    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            clean = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)

            # 이미지 매핑
            for card in result.get("추천카드", []):
                card_name = card.get("카드명", "")
                matched_url = all_image_map.get(card_name, "")
                card["이미지URL"] = matched_url

                if matched_url:
                    print(f"✅ 이미지 매핑 성공: {card_name}")
                else:
                    print(f"❌ 이미지 매핑 실패: {card_name}")

            result = enrich_card_benefits(result, conditions, df_chunks)

            return result

        except Exception as e:
            print(f"⚠️ 시도 {attempt+1}/{max_retries} 실패: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print("최대 재시도 횟수 초과")
                return raw


# ── 카드형 레이아웃 출력 함수 ──
def display_recommendation(result):
    if isinstance(result, str):
        st.error("추천 결과를 가져오는데 실패했습니다. 다시 시도해주세요.")
        return

    # 추천 과정 (접을 수 있게)
    with st.expander("📊 추천 과정 보기"):
        process = result.get("추천과정", {})
        for step, content in process.items():
            st.markdown(f"**{step}**: {content}")

    # 추천 카드 (카드형 레이아웃)
    st.subheader("💳 추천 카드")
    cards = result.get("추천카드", [])

    if not cards:
        st.warning("조건에 맞는 카드를 찾지 못했습니다.")
        return

    cols = st.columns(len(cards))

    for i, (col, card) in enumerate(zip(cols, cards)):
        with col:
            # 카드 이미지
            image_url = card.get("이미지URL", "")
            if image_url:
                st.image(image_url, width="stretch")
            else:
                st.image(
                    "https://via.placeholder.com/300x180?text=No+Image",
                    width="stretch",
                )

            # 카드 정보
            st.markdown(f"### {card['카드명']}")
            st.caption(f"{card['은행']} | {card['카드종류']}")

            # 조건 배지
            col1, col2 = st.columns(2)
            with col1:
                st.metric("전월실적", f"{card['전월실적조건']:,}원")
            with col2:
                st.metric("연회비", f"{card['연회비']:,}원")

            # 핵심 혜택
            st.markdown("**✨ 핵심혜택**")

            with st.container(border=True):
                benefits = card["핵심혜택"].split("\n")
                for benefit in benefits:
                    if benefit.strip():
                        st.markdown(f"• {benefit.strip()}")

            # 추천 이유
            st.markdown("**💡 추천이유**")
            st.success(card["추천이유"])

    종합멘트 = result.get("종합멘트", "")
    if 종합멘트:
        st.markdown("---")
        st.markdown("**💬 종합 추천 의견**")
        st.info(종합멘트)
    # 제외 카드
    excluded = result.get("제외카드", "")
    if excluded:
        with st.expander("❌ 제외된 카드 보기"):
            st.write(excluded)


# ── 세션 상태 초기화 ──
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = None

# ── 사이드바 ──
with st.sidebar:
    st.title("💳 AI 카드 추천")
    st.markdown("---")

    st.subheader("👤 페르소나 선택")
    st.caption("페르소나를 선택하면 자동으로 질문이 입력돼요")

    for persona in personas:
        if st.button(
            f"{persona['name']}",
            key=f"persona_{persona['id']}",
            width="stretch",
        ):
            st.session_state.selected_persona = persona
            st.session_state.messages = []  # 채팅 초기화
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ 채팅 초기화", width="stretch"):
        st.session_state.messages = []
        st.session_state.selected_persona = None
        st.rerun()

# ── 메인 화면 ──
st.title("💳 AI 카드 추천 챗봇")
st.caption("소비 패턴을 알려주시면 최적의 카드를 추천해드립니다")

# 채팅 히스토리 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            # 추천 결과 카드형 레이아웃으로 출력
            display_recommendation(message["content"])

# 페르소나 선택 시 자동 입력
if st.session_state.selected_persona:
    persona = st.session_state.selected_persona
    # 이미 메시지가 없을 때만 자동 실행
    if not st.session_state.messages:
        query = persona["query"]
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("카드를 추천하는 중..."):
                result = recommend_card(query)
            display_recommendation(result)

        st.session_state.messages.append({"role": "assistant", "content": result})
        st.session_state.selected_persona = None

# 사용자 입력창
if user_input := st.chat_input("소비 패턴을 입력해주세요..."):

    # Moderation 체크
    if check_moderation(user_input):
        st.error("⚠️ 부적절한 내용이 감지되었습니다. 다시 입력해주세요.")

    elif not is_card_related(user_input):
        with st.chat_message("assistant"):
            st.warning(
                "💳 저는 신용카드/체크카드 추천 전문 챗봇이에요!\n\n"
                "소비 패턴을 알려주시면 최적의 카드를 추천해드릴게요.\n\n"
                "**예시:**\n"
                "- '월 50만원 쓰는 직장인인데 카페랑 교통 혜택 원해'\n"
                "- '주유비가 부담돼, 연회비 없는 카드 추천해줘'"
            )
    else:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # 추천 실행
        with st.chat_message("assistant"):
            with st.spinner("카드를 추천하는 중..."):
                result = recommend_card(user_input)
            display_recommendation(result)

        st.session_state.messages.append({"role": "assistant", "content": result})
