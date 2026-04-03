import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
import re
from typing import Callable
from urllib.parse import urlparse

from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from tavily import TavilyClient

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

ProgressCallback = Callable[[dict], None]

UNDERSTAND_PROMPT = """You are the routing brain for a consumer decision agent.

Return ONLY valid JSON:
{
  "intent": "evaluate" | "recommend" | "compare",
  "user_goal": "<what the user is actually trying to solve, in their own terms>",
  "search_target": "<best search phrase to use once clarified>",
  "category_hint": "<electronics|beauty|clothing|food|fitness|other>",
  "is_specific_product": true | false,
  "comparison_targets": ["<product A>", "<product B>"],
  "ambiguity_level": "clear" | "moderate" | "vague",
  "needs_budget": true | false,
  "needs_use_case": true | false,
  "needs_skin_type": true | false,
  "needs_occupation_context": true | false,
  "known_constraints": ["<constraint 1>", "<constraint 2>"],
  "inferred_profile_signals": {
    "occupation": "<if detectable from input, else null>",
    "age_group": "<student|young_professional|other|null>",
    "lifestyle_hints": ["<hint 1>", "<hint 2>"]
  }
}

Rules:
- Use "evaluate" when the user asks whether one product is worth buying.
- Use "recommend" when the user wants options, shortlists, or uses "帮我推荐".
- Use "compare" when the user names multiple products or asks which is better.
- Set ambiguity_level:
    * "clear" — product is named, intent is obvious, no critical missing context.
    * "moderate" — product is named but key use scenario or budget is missing.
    * "vague" — no specific product, or the stated need is too broad to search well.
- Set needs_occupation_context = true when:
    * The user is comparing close-spec professional tools (laptops, cameras, tablets).
    * The right answer depends significantly on whether the user is a student, designer,
      developer, video editor, casual user, etc.
    * intent = "compare" AND category = "electronics".
- known_constraints: extract any hard constraints the user already stated
  (e.g., "Apple生态", "预算3000以内", "敏感肌", "通勤用"). Leave empty if none stated.
- inferred_profile_signals: extract what you can infer about the person from their
  phrasing, vocabulary, and context. Use null if unsure — do not hallucinate.
- Mark needs_use_case = true only if the answer would change significantly by scenario.
- Mark needs_budget = true only if price tolerance is necessary for a good answer.
- Mark needs_skin_type = true only for skincare/beauty where skin type changes suitability.
- If a product screenshot clearly shows one product, treat it as evaluate unless the
  user explicitly asks for recommendations.
- Match the user's language. If the user writes in Chinese, any free-text fields stay Chinese.
"""

SCOPING_PROMPT = """You are the early-stage reasoning layer for a consumer decision agent.

Your job:
- Look at the product/need plus a small amount of web evidence.
- Identify which decision dimensions actually matter for a good buy/skip decision.
- Decide whether a key missing user-specific input is still needed before a full search.
- If a follow-up is needed, generate a question that is SPECIFIC to this exact product
  and situation — never use generic buckets unless they are genuinely the right level.

Return ONLY valid JSON:
{
  "preliminary_take": "<one sentence on what this item/category seems to optimize for>",
  "decision_dimensions": [
    {
      "title": "<dimension name>",
      "status": "strong" | "mixed" | "unclear" | "risk",
      "detail": "<what matters on this dimension and why>"
    }
  ],
  "missing_dimension": "none" | "use_case" | "budget" | "skin_type" | "health_context" | "owned_items" | "occupation_context" | "pet_context" | "preference",
  "needs_followup": true | false,
  "followup_reason": "<why this missing input materially affects the verdict — be specific>",
  "followup_questions": [
    {
      "question": "<the question to ask>",
      "options": ["<option A>", "<option B>", "<option C>"],
      "reason": "<one sentence: why this input changes the verdict>",
      "question_type": "multiple_choice" | "open_text",
      "open_text_placeholder": "<hint shown inside text field if open_text>"
    }
  ],
  "question_type": "multiple_choice" | "open_text",
  "followup_question": "<same as followup_questions[0].question>",
  "followup_options": ["<same as followup_questions[0].options>"],
  "open_text_placeholder": "<same as followup_questions[0].open_text_placeholder>",
  "search_focus": ["<search angle 1>", "<search angle 2>"]
}

Rules for followup_questions:
- Put ONLY the single highest-impact question in followup_questions.
- Maximum 1 question total at this stage.
- For each question, set question_type and fill options or open_text_placeholder accordingly.
- Also mirror the first question into followup_question / followup_options / question_type /
  open_text_placeholder fields for backward compatibility.
- If needs_followup is false, set followup_questions to [].

Rules — when to ask:
- CRITICAL: Read USER CONTEXT carefully. If information is already stored there (occupation, pets,
  skin type, use cases, budget, etc.), do NOT ask for it again. Treat stored context as ground truth.
- Only ask a follow-up if the missing input could materially change whether this is worth buying or suitable.
- The question must be about a specific, nameable dimension that your early web evidence shows matters
  for THIS product — not a generic clarification you'd ask for any product.
- Never assume the user has a problem (health issue, digestive problem, budget constraint) they haven't
  mentioned. If it's not in USER CONTEXT and not stated, ask first — never invent.

Rules — how to ask (question format):
- Default to multiple_choice. Only use open_text when the answer space is truly continuous
  (e.g., full workflow description where 3 options couldn't cover the range).
- Ask ONE dimension at a time. Never combine age + size + allergy, or budget + use case,
  or any other multi-axis bundle into the same question.
- For multiple_choice: generate exactly 3 options. Each option must cover ONE scenario only —
  never "A or B" in a single option. Options must be mutually exclusive and concrete.
  BAD option: "日常通勤 / 轻度运动"  GOOD option: "上下班通勤为主"
- For open_text: write a warm, open invitation. Set followup_options to [].

Category-specific rules:
- For supplements / nutrition: health suitability must be checked before product quality.
  Ask if they have a specific deficiency or health signal vs. just general supplementing.
- For pet products: if the product is age-specific, size-specific, or sensitivity-specific,
  verify those dimensions one by one. Never ask multiple pet dimensions in one question.
- For skincare: if skin type is not in USER CONTEXT, ask it first.
- For electronics close-spec comparisons: if occupation/workflow not in USER CONTEXT, ask with
  multiple_choice covering the 3 most distinct user types for that product category.
- For any product where the right answer depends on a specific user attribute NOT in USER CONTEXT
  (pet species/breed, health condition, household composition, professional workflow), ask for it.
  Use multiple_choice if 3 options cover the realistic space; use open_text if they don't.
- Match the user's language. If the user writes in Chinese, ask and respond in Chinese.
"""

FINAL_PROMPT = """You are "Buy or Not" — a sharp, warm consumer decision assistant for people aged 18–30.
You’ve done the research. Now give them a real, direct answer they can actually act on.

Tone:
- Be direct. Give an actual opinion, not a hedge.
- Sound like a knowledgeable friend who genuinely cares — not a product review robot.
- Light personality is welcome: a touch of wit, a dash of warmth. No cringe slang.
- When advising against buying, be candid but not preachy — explain what would serve them better.
- Chinese contexts: use natural 普通话. Target: how a sharp 25-year-old would explain this
  to a friend over coffee. Avoid 综上所述, 不得不说, and other filler phrases.
- No markdown in string fields. No em-dashes as filler.

Return ONLY valid JSON in this schema:
{
  "result_type": "decision" | "recommendation",
  "intent": "evaluate" | "recommend" | "compare",
  "headline": "<sharp, useful one-liner — the verdict in plain language>",
  "product_name": "<main product if specific, else empty string>",
  "category": "<electronics|beauty|clothing|food|fitness|other>",
  "price_range": "<best current price hint or realistic budget band>",
  "verdict": "worth_buying" | "cautious" | "not_recommended" | null,
  "summary": "<2–3 sentences. State the conclusion first, then the key reason. Warm but decisive.>",
  "decision_dimensions": [
    {
      "title": "<dimension name>",
      "status": "strong" | "mixed" | "weak" | "risk",
      "detail": "<how this dimension shaped the verdict — be concrete>"
    }
  ],
  "reasons": [
    "<reason with a specific fact, number, named source, or concrete comparison>",
    "<reason with a specific fact, number, named source, or concrete comparison>",
    "<optional third reason — only include if materially different from above>"
  ],
  "key_specs": {
    "<spec name>": "<value>"
  },
  "alternatives": [
    {"name": "<alternative>", "reason": "<why it fits better for a specific type of user>"}
  ],
  "recommendations": [
    {
      "name": "<product name>",
      "price_hint": "<price or range>",
      "best_for": "<which user profile or scenario this fits>",
      "reason": "<1–2 sentences — cite a concrete differentiator>",
      "tradeoffs": ["<honest tradeoff 1>", "<honest tradeoff 2>"]
    }
  ],
  "followup": null | {
    "question": "<one last clarifying question if a detail would materially change the answer>",
    "options": ["<A>", "<B>", "<C>"],
    "question_type": "multiple_choice" | "open_text",
    "open_text_placeholder": "<hint text>"
  },
  "fit_summary": "<for evaluate/compare: why this product fits or doesn’t fit this specific user>",
  "caution_check": "<the single most important thing to verify before buying>",
  "buy_if": "<the scenario where buying still makes sense — be specific>",
  "skip_if": "<the scenario where skipping is clearly better — be specific>",
  "primary_recommendation": null | {
    "name": "<better option>",
    "reason": "<why it beats the original for this user>",
    "better_points": ["<concrete point 1>", "<concrete point 2>"]
  },
  "budget_alternative": null | {
    "name": "<budget option>",
    "reason": "<what you give up and why it might still be enough>"
  },
  "better_direction": "<if the original item is off-track, what kind of solution actually fits>",
  "scores": {
    "cost_value": <1-5 integer: price-to-value ratio>,
    "quality": <1-5 integer: build quality and reliability>,
    "brand": <1-5 integer: brand reputation and trustworthiness>,
    "fit": <1-5 integer: how well it fits THIS specific user's stated needs>,
    "longevity": <1-5 integer: expected long-term value and durability>,
    "reviews": <1-5 integer: overall user satisfaction from real reviews>
  }
}

Rules for scores:
- Fill all 6 score dimensions for every result (evaluate, compare, recommend).
- For recommend/compare: base scores on the first/primary recommendation or the winning product.
- Scores must reflect real evidence: a product with thin reviews should score 2-3 on "reviews",
  not a confident 4. Don't default to middle scores — spread them based on actual evidence.
- "fit" score is personal: the same product may score 5 for one user and 2 for another.

Rules for evidence quality:
- Each reason MUST contain at least one of: a price figure, a spec value, a named source
  (e.g., "rtings.com", "官网标称", "某测评"), a direct comparison, or a concrete complaint pattern.
- Vague reasons like "口碑不错" or "性价比高" are NOT acceptable alone — pair them with evidence.
- For compare intent: assess each product on key dimensions before declaring a winner.
  State which product wins on which dimension, then give the overall call.
- For recommend intent: order recommendations from best fit to weakest. The first pick
  should be clearly differentiated from the others.
- verdict must always be set for evaluate/compare results. Never leave it null for these.
- If web evidence is thin, say so in summary and mark affected dimensions as "mixed" or "unclear"
  rather than fabricating confidence.
- Use the user’s occupation/lifestyle context (if available) to personalize fit_summary.
- Respect known_constraints — if the user stated a hard constraint, the answer must honor it.
- Keep strings concise and UI-ready. No markdown.
- Match the user’s language. If the user writes in Chinese, all string fields are in Chinese.
- In fit_summary and summary, explicitly connect product traits to user traits:
  price sensitivity, buying style, life stage, family setup, pet stage, skin type, or high-frequency categories.
  Don't stop at specs alone — explain why those specs matter for THIS user.

Reasoning discipline — the foundation of trust:
- ONLY reference concerns grounded in: (a) what the user explicitly stated, (b) verified product
  issues from web evidence (documented complaints, named sources), or (c) the user’s stored profile.
- Do NOT invent a negative scenario for the user unless they said so. Invented risks destroy trust.
- Do NOT make assumptions about a user’s actual practices from their job title alone.
  BAD: "你作为产品经理，可能需要 Photoshop" (they never said this)
  GOOD: "你提到平时做原型设计，所以..." (they explicitly said this)
- "buy_if" and "skip_if": write as OBJECTIVE CONDITIONS the user can self-apply.
  BAD: "如果你的狗狗有肠胃问题，不建议购买" (asserts they have the problem)
  GOOD: "如果你的狗狗本来消化比较敏感，值得谨慎" (neutral, self-applicable)
- Alternatives: only include when there is a concrete user signal that the original might not fit.
  Do NOT add alternatives as a generic hedge. If the product is fine for the user, say so clearly.
- When stored profile data is available (pet breed, occupation, lifestyle), reference it by name.
  "因为你养的是[breed]..." / "考虑到你说的[workflow]..." — specificity is what creates trust.

Writing for trust — how to sound like a person, not a report:
- Lead with your actual conclusion, then explain why. Don’t build up to it.
- Write summary as if you’re texting a friend: "这款其实挺适合你的，主要原因是..."
- When you’re confident, be confident. Don’t hedge every sentence.
- When evidence is thin, say so honestly: "网上关于这块的信息不多，但从成分表来看..."
- If the user’s situation clearly fits the product, say "买吧" in your own words.
  If it clearly doesn’t, say why plainly without padding.
- When explaining the verdict, always include at least one "because of you" link:
  for example "考虑到你更在意耐用性..." / "你家这只是幼年阶段..." / "你平时买这类东西更看重性价比...".
"""

SEARCH_PLANS = {
    "evaluate": [
        ("fit_signal", "Fit & key constraints", "{target} best for who use case fit issues"),
        ("price_specs", "Price & specs", "{target} official specs price"),
        ("reviews", "Reviews & complaints", "{target} review pros cons complaints worth buying alternatives"),
    ],
    "recommend": [
        ("needs_landscape", "Best options", "{target} best options buying guide"),
        ("fit_signal", "Scene fit", "{target} best for commuting office daily running sensitive skin tradeoffs"),
        ("marketplaces", "Market pricing", "{target} amazon tmall jd taobao price"),
        ("reviews", "User sentiment", "{target} review complaints best value comparison"),
    ],
    "compare": [
        ("comparisons", "Head-to-head comparisons", "{target} comparison review vs"),
        ("fit_signal", "Fit by scenario", "{target} which is better for commuting office travel running"),
        ("reviews", "User sentiment", "{target} review pros cons complaints alternatives"),
    ],
}

SEARCH_PLANS_ZH = {
    "evaluate": [
        ("fit_signal", "适用性分析", "{target} 适合什么人 使用场景 问题反馈"),
        ("price_specs", "价格与规格", "{target} 官方规格 价格 参数"),
        ("reviews", "用户评价与问题", "{target} 评测 优缺点 用户反馈 值得买 替代品"),
    ],
    "recommend": [
        ("needs_landscape", "最佳选择", "{target} 最佳选择 购买指南"),
        ("fit_signal", "场景匹配", "{target} 哪款更适合 不同场景 对比"),
        ("marketplaces", "市场价格", "{target} 价格 淘宝 京东 天猫"),
        ("reviews", "用户口碑", "{target} 用户评价 投诉 性价比 对比"),
    ],
    "compare": [
        ("comparisons", "对比评测", "{target} 对比 评测 哪个好"),
        ("fit_signal", "场景适配", "{target} 哪款更适合不同使用场景"),
        ("reviews", "用户反馈", "{target} 评测 优缺点 投诉 替代品"),
    ],
}

SCOPING_PLANS = {
    "evaluate": [
        ("product_fit", "Product fit snapshot", "{target} best for who common issues who should buy"),
        ("suitability", "Suitability check", "{target} suitability contraindications use case complaints"),
    ],
    "recommend": [
        ("decision_lens", "Decision factors", "{target} buying guide what to consider"),
        ("fit_signal", "User-fit clues", "{target} best for who common tradeoffs"),
    ],
    "compare": [
        ("decision_lens", "Decision factors", "{target} comparison what matters most"),
        ("fit_signal", "User-fit clues", "{target} which is better for different use cases"),
    ],
}

SCOPING_PLANS_ZH = {
    "evaluate": [
        ("product_fit", "产品适用性", "{target} 适合什么人 常见问题 值得买吗"),
        ("suitability", "适用性检查", "{target} 使用禁忌 适用场景 用户投诉"),
    ],
    "recommend": [
        ("decision_lens", "决策维度", "{target} 购买指南 选购要点"),
        ("fit_signal", "用户适配", "{target} 最适合什么人 常见权衡"),
    ],
    "compare": [
        ("decision_lens", "对比维度", "{target} 对比 最重要的考量"),
        ("fit_signal", "场景适配", "{target} 哪款更适合不同使用场景"),
    ],
}

CATEGORY_KEYWORDS = {
    "beauty": [
        "防晒", "面霜", "精华", "乳液", "粉底", "口红", "爽肤水", "洗面奶", "香水", "护肤", "美妆",
        "sunscreen", "serum", "moisturizer", "cleanser", "foundation", "lipstick", "skincare", "beauty",
    ],
    "electronics": [
        "耳机", "手机", "电脑", "相机", "平板", "显示器", "键盘", "鼠标", "音箱", "吹风机",
        "headphone", "earbuds", "phone", "laptop", "camera", "tablet", "monitor", "keyboard", "mouse", "speaker",
    ],
    "clothing": [
        "鞋", "跑鞋", "外套", "衣服", "裤子", "包", "手表", "项链", "裙", "靴",
        "shoe", "sneaker", "shirt", "jacket", "coat", "pants", "bag", "watch", "necklace", "dress", "boot",
    ],
    "food": [
        "咖啡", "零食", "麦片", "牛奶", "蛋白粉", "补剂", "维生素", "营养", "保健品", "鱼油", "omega", "dha", "epa",
        "coffee", "snack", "cereal", "milk", "protein", "supplement", "vitamin", "nutrition", "fish oil", "omega-3",
    ],
    "fitness": [
        "瑜伽垫", "哑铃", "跑步机", "自行车", "户外", "徒步", "登山", "运动手表", "训练",
        "dumbbell", "treadmill", "bike", "hiking", "trail", "outdoor", "training", "fitness",
    ],
}

RELATED_CATEGORY_MAP = {
    "beauty": {"beauty"},
    "electronics": {"electronics"},
    "clothing": {"clothing", "fitness"},
    "food": {"food"},
    "fitness": {"fitness", "clothing"},
    "other": {"other"},
}

FAMILY_KEYWORDS = {
    "footwear": ["鞋", "跑鞋", "球鞋", "sneaker", "shoe", "boot", "loafer", "sandal", "trainer"],
    "jewelry": ["项链", "戒指", "耳环", "手链", "necklace", "ring", "earring", "bracelet"],
    "apparel": ["外套", "衣服", "裤", "裙", "jacket", "coat", "shirt", "pants", "dress"],
    "bag": ["包", "背包", "手袋", "bag", "backpack", "tote"],
    "audio": ["耳机", "音箱", "headphone", "earbud", "speaker"],
    "phone": ["手机", "phone", "iphone"],
    "computer": ["电脑", "笔记本", "平板", "显示器", "laptop", "computer", "tablet", "monitor"],
    "skincare": ["防晒", "面霜", "精华", "洗面奶", "乳液", "sunscreen", "serum", "cleanser", "moisturizer"],
    "makeup": ["粉底", "口红", "眼影", "foundation", "lipstick", "makeup"],
    "fragrance": ["香水", "perfume", "fragrance"],
    "supplement": ["蛋白粉", "补剂", "维生素", "鱼油", "omega", "dha", "epa", "fish oil", "protein", "supplement", "vitamin"],
    "snack": ["零食", "咖啡", "麦片", "snack", "coffee", "cereal"],
    "outdoor_gear": ["徒步", "登山", "露营", "hiking", "camping", "trail"],
    "fitness_gear": ["瑜伽垫", "哑铃", "跑步机", "自行车", "dumbbell", "treadmill", "bike"],
}

USE_CASE_KEYWORDS = [
    "通勤", "日常", "办公", "运动", "跑步", "旅行", "送礼", "学生", "居家", "通学", "徒步",
    "commute", "daily", "office", "work", "running", "travel", "gift", "gym", "home", "school", "hiking",
]

BUDGET_KEYWORDS = [
    "预算", "贵", "便宜", "性价比", "price", "budget", "cheap", "expensive", "value", "afford",
    "元", "块", "rmb", "usd", "$", "¥",
]

SKIN_KEYWORDS = [
    "干皮", "油皮", "混干", "混油", "敏感肌", "肤质",
    "dry skin", "oily skin", "combination skin", "sensitive skin", "skin type",
]


def _json_model(system_instruction: str):
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
        generation_config={"response_mime_type": "application/json"},
    )


def _fast_json_model(system_instruction: str):
    """Fast model for latency-sensitive structured outputs (routing, clarification, extraction).
    Uses gemini-2.5-flash-lite which has no thinking overhead."""
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        system_instruction=system_instruction,
        generation_config={"response_mime_type": "application/json"},
    )


def emit_progress(callback: ProgressCallback | None, **event):
    if callback:
        callback(event)


def clean_text(value: str, limit: int = 220) -> str:
    if not value:
        return ""
    collapsed = " ".join(str(value).split())
    return collapsed[:limit].rstrip()


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]

    return json.loads(raw)


def extract_site(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def joined_followup_text(followup_qa: list | None) -> str:
    if not followup_qa:
        return ""
    return " ".join(
        clean_text(f"{item.get('question', '')} {item.get('answer', '')}", 120)
        for item in followup_qa
    )


def infer_category_hint(*parts: str) -> str:
    text = " ".join([part for part in parts if part]).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            return category
    return "other"


def contains_any(text: str, keywords: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def infer_product_family(*parts: str) -> str:
    text = " ".join([part for part in parts if part]).lower()
    for family, keywords in FAMILY_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            return family
    return "generic"


def prefers_chinese(*parts: str) -> bool:
    text = " ".join([part for part in parts if part])
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def has_budget_signal(text: str) -> bool:
    lowered = (text or "").lower()
    if contains_any(lowered, BUDGET_KEYWORDS):
        return True
    return bool(re.search(r"(\$|¥|￥|usd|rmb)?\s?\d{2,5}\s*(元|块|usd|rmb|dollars?)?", lowered))


def has_use_case_signal(text: str) -> bool:
    return contains_any(text, USE_CASE_KEYWORDS)


def has_skin_signal(text: str) -> bool:
    return contains_any(text, SKIN_KEYWORDS)


OCCUPATION_KEYWORDS = [
    "设计师", "程序员", "工程师", "学生", "老师", "医生", "律师", "剪辑", "视频",
    "摄影", "上班族", "研究生", "大学", "专业",
    "designer", "developer", "programmer", "student", "engineer", "teacher",
    "video", "editing", "photography", "professional",
]


def has_occupation_signal(text: str) -> bool:
    return contains_any(text, OCCUPATION_KEYWORDS)


def question_kind(question: str) -> str:
    text = (question or "").lower()
    if "肤质" in question or "skin" in text:
        return "skin_type"
    if "职业" in question or "工作" in question or "workflow" in text:
        return "occupation_context"
    if (
        "适合吃" in question
        or "考虑吃" in question
        or "吃它" in question
        or "补" in question
        or "身体" in question
        or "医生" in question
        or "药" in question
        or "阿司匹林" in question
        or "抗凝" in question
        or "health" in text
        or "contra" in text
        or "aspirin" in text
        or "medication" in text
    ):
        return "health_context"
    if ("宠物" in question or "狗" in question or "猫" in question or "pet" in text or "dog" in text or "cat" in text) and (
        "年龄" in question or "几岁" in question or "life stage" in text or "puppy" in text or "kitten" in text or "阶段" in question
    ):
        return "pet_life_stage"
    if ("宠物" in question or "狗" in question or "猫" in question or "pet" in text or "dog" in text or "cat" in text) and (
        "敏感" in question or "过敏" in question or "allergy" in text or "sensitive" in text
    ):
        return "pet_sensitivity"
    if ("宠物" in question or "狗" in question or "猫" in question or "pet" in text or "dog" in text or "cat" in text) and (
        "体重" in question or "体型" in question or "size" in text or "breed" in text or "品种" in question
    ):
        return "pet_size"
    if "预算" in question or "budget" in text or "price" in text:
        return "budget"
    if "场景" in question or "use" in text or "做什么" in question or "怎么用" in question:
        return "use_case"
    if "已有" in question or "already" in text:
        return "owned_items"
    if "自己住" in question or "跟家人住" in question or "家庭" in question:
        return "family_context"
    if (
        "宠物" in question or "狗" in question or "猫" in question or "品种" in question
        or "pet" in text or "dog" in text or "cat" in text or "breed" in text
    ):
        return "pet_context"
    return "other"


def build_search_blob(search_data: dict) -> str:
    parts = []
    for bucket in (search_data or {}).values():
        parts.append(bucket.get("answer", ""))
        parts.append(bucket.get("query", ""))
        for item in bucket.get("results", [])[:3]:
            parts.append(item.get("title", ""))
            parts.append(item.get("snippet", ""))
    return " ".join([part for part in parts if part])


PET_KEYWORDS = ["狗", "狗粮", "犬", "猫", "猫粮", "pet", "dog", "cat", "puppy", "kitten"]
PET_AGE_KEYWORDS = ["幼犬", "幼猫", "成犬", "成猫", "senior", "adult", "puppy", "kitten", "一岁", "1岁", "1 year", "life stage", "stage"]
PET_SENSITIVITY_KEYWORDS = ["敏感", "过敏", "肠胃", "软便", "allergy", "sensitive", "digestive", "grain free", "hypoallergenic"]
PET_SIZE_KEYWORDS = ["大型犬", "中型犬", "小型犬", "体重", "25kg", "large breed", "medium breed", "small breed", "size", "breed"]


def infer_pet_species(*parts: str) -> str:
    text = " ".join([part for part in parts if part]).lower()
    if "cat" in text or "猫" in text:
        return "cat"
    if "dog" in text or "狗" in text or "犬" in text:
        return "dog"
    return "pet"


def is_pet_related(*parts: str) -> bool:
    return contains_any(" ".join([part for part in parts if part]), PET_KEYWORDS)


def pet_profile_has_age(profile: dict) -> bool:
    pets = profile.get("pets") or []
    return any(isinstance(pet, dict) and pet.get("age") for pet in pets)


def pet_profile_has_sensitivity(profile: dict) -> bool:
    pets = profile.get("pets") or []
    return any(isinstance(pet, dict) and pet.get("health_notes") for pet in pets)


def detect_evidence_constraints(search_data: dict, product_hint: str, understanding: dict) -> dict:
    blob = f"{product_hint or ''} {understanding.get('search_target', '')} {understanding.get('user_goal', '')} {build_search_blob(search_data)}"
    lowered = blob.lower()
    return {
        "pet_related": is_pet_related(blob),
        "pet_age_specific": contains_any(lowered, PET_AGE_KEYWORDS),
        "pet_sensitivity_specific": contains_any(lowered, PET_SENSITIVITY_KEYWORDS),
        "pet_size_specific": contains_any(lowered, PET_SIZE_KEYWORDS),
    }


def use_case_option_set(family: str) -> list[str]:
    family_map = {
        "bag": ["日常通勤，要装电脑/随身物", "短途出行 / 周末外出", "户外 / 高频重装使用"],
        "footwear": ["日常通勤 / 城市走路", "轻运动 / 周末出门", "户外 / 长时间高频使用"],
        "apparel": ["通勤 / 日常穿着", "周末出门 / 风格搭配", "运动 / 户外场景"],
        "outdoor_gear": ["城市通勤 / 偶尔用", "周末轻户外", "长线户外 / 高频高强度"],
        "fitness_gear": ["轻量入门使用", "一周用几次", "高频训练 / 强度很高"],
    }
    return family_map.get(family, ["通勤 / 日常", "居家 / 办公", "高频使用 / 户外"])


def build_framework_followup(
    understanding: dict,
    product_hint: str,
    user_profile: dict,
    followup_qa: list | None,
    followup_count: int,
    search_data: dict,
) -> dict | None:
    if followup_count >= 3:
        return None

    combined = " ".join(
        [
            product_hint or "",
            understanding.get("search_target", ""),
            understanding.get("user_goal", ""),
            joined_followup_text(followup_qa),
            build_search_blob(search_data),
        ]
    )
    family = infer_product_family(combined)
    category = understanding.get("category_hint", "other")
    asked_kinds = {question_kind(item.get("question", "")) for item in (followup_qa or [])}
    evidence = detect_evidence_constraints(search_data, product_hint, understanding)
    species = infer_pet_species(combined)

    if evidence["pet_related"]:
        if evidence["pet_age_specific"] and "pet_life_stage" not in asked_kinds and not pet_profile_has_age(user_profile or {}):
            species_label = "狗狗" if species == "dog" else "猫咪" if species == "cat" else "宠物"
            return {
                "missing_dimension": "pet_life_stage",
                "needs_followup": True,
                "followup_reason": "这类商品对年龄 / 阶段很敏感，得先确认它是不是刚好适配你家这只。",
                "question_type": "multiple_choice",
                "followup_question": f"先确认一下，你家这只{species_label}现在更接近哪个阶段？",
                "followup_options": ["还在幼年 / 1岁以内", "成年阶段", "高龄 / 7岁以上"],
                "search_focus": ["life stage fit", "age suitability"],
            }
        if evidence["pet_sensitivity_specific"] and "pet_sensitivity" not in asked_kinds and not pet_profile_has_sensitivity(user_profile or {}):
            species_label = "它" if species == "pet" else "它"
            return {
                "missing_dimension": "pet_sensitivity",
                "needs_followup": True,
                "followup_reason": "这类商品的核心分歧通常在敏感度和耐受性，不先确认这一点很容易判断偏。",
                "question_type": "multiple_choice",
                "followup_question": f"{species_label}平时有比较明确的食物敏感或肠胃敏感吗？",
                "followup_options": ["没有明显问题", "有点敏感，偶尔会软便 / 挑食", "有明确过敏或需要避开某些成分"],
                "search_focus": ["sensitivity fit", "ingredient tolerance"],
            }
        if evidence["pet_size_specific"] and "pet_size" not in asked_kinds:
            species_label = "狗狗" if species == "dog" else "宠物"
            return {
                "missing_dimension": "pet_size",
                "needs_followup": True,
                "followup_reason": "这类配方和颗粒大小往往会按体型分层，不先确认体型会直接影响适配性。",
                "question_type": "multiple_choice",
                "followup_question": f"它的体型大概更接近哪边？",
                "followup_options": ["小型", "中型", "大型"],
                "search_focus": ["size fit", "breed size guidance"],
            }

    if category == "food" and infer_product_family(combined) == "supplement" and "health_context" not in asked_kinds:
        return {
            "missing_dimension": "health_context",
            "needs_followup": True,
            "followup_reason": "这类补剂先看你适不适合，再看品牌和剂量，不然很容易从一开始就跑偏。",
            "question_type": "multiple_choice",
            "followup_question": "你这次考虑吃它，更接近哪一种情况？",
            "followup_options": ["日常保健 / 想补一补", "有明确不适或化验指标", "医生 / 营养师建议我补"],
            "search_focus": ["benefit suitability", "contraindications side effects"],
        }

    if understanding.get("needs_skin_type") and "skin_type" not in asked_kinds and not user_profile.get("skin_type"):
        return {
            "missing_dimension": "skin_type",
            "needs_followup": True,
            "followup_reason": "这个品类对肤质很敏感，先把肤质确认清楚，后面的结论才像是为你做的。",
            "question_type": "multiple_choice",
            "followup_question": "你的肤质更接近哪一种？",
            "followup_options": ["干皮 / 混干", "油皮 / 混油", "敏感肌"],
            "search_focus": ["skin type fit", "sensitive skin issues"],
        }

    if understanding.get("needs_occupation_context") and "occupation_context" not in asked_kinds and not user_profile.get("occupation"):
        return {
            "missing_dimension": "occupation_context",
            "needs_followup": True,
            "followup_reason": "这类产品的差异主要体现在工作流里，不先知道你主要拿它干什么，很难给出稳的结论。",
            "question_type": "multiple_choice",
            "followup_question": "你更接近哪种使用方式？",
            "followup_options": ["学习 / 文档 / 轻办公", "开发 / 数据 / 多任务", "设计 / 视频 / 重内容处理"],
            "search_focus": ["workflow fit", "performance by workload"],
        }

    if understanding.get("needs_use_case") and "use_case" not in asked_kinds and not has_use_case_signal(combined):
        return {
            "missing_dimension": "use_case",
            "needs_followup": True,
            "followup_reason": "这类商品要先对准场景，不然就算参数好看，也可能买回来并不适合你。",
            "question_type": "multiple_choice",
            "followup_question": "你最主要会把它放在哪种场景里？",
            "followup_options": use_case_option_set(family),
            "search_focus": ["fit by scenario", "comfort durability complaints"],
        }

    if understanding.get("needs_budget") and "budget" not in asked_kinds and not has_budget_signal(combined):
        return {
            "missing_dimension": "budget",
            "needs_followup": True,
            "followup_reason": "预算会直接改变候选集合和判断标准，先卡个大概范围会快很多。",
            "question_type": "multiple_choice",
            "followup_question": "你更接近哪种预算心态？",
            "followup_options": ["先把价格控稳", "愿意多花一点换明显提升", "预算不是这次第一优先级"],
            "search_focus": ["price tier", "best value options"],
        }

    return None




def identify_product_from_image(pil_image: Image.Image) -> str:
    response = genai.GenerativeModel(model_name="gemini-2.5-flash-lite").generate_content(
        [
            pil_image,
            """Look at this image carefully. Identify the product shown.
Return ONLY a JSON object like:
{"product_name": "<brand + model name>", "extra_info": "<visible pack size, color, price, or other useful text>"}
No other text.""",
        ]
    )
    try:
        data = parse_json(response.text)
        return clean_text(f"{data.get('product_name', '')} {data.get('extra_info', '')}".strip(), 140)
    except Exception:
        return clean_text(response.text.strip().split("\n")[0], 140)


def heuristic_understanding(input_text: str, product_hint: str, followup_qa: list | None = None) -> dict:
    followup_text = joined_followup_text(followup_qa)
    combined = f"{input_text or ''} {product_hint or ''} {followup_text}".strip()
    lowered = combined.lower()

    if any(word in combined for word in ["推荐", "求推荐", "帮我找", "推荐个", "有什么适合", "哪些值得买"]):
        intent = "recommend"
    elif any(word in combined for word in ["vs", "对比", "哪个好", "怎么选", "区别"]):
        intent = "compare"
    else:
        intent = "evaluate" if product_hint or input_text else "recommend"

    category = infer_category_hint(combined)
    is_close_spec_compare = (
        intent == "compare"
        and category == "electronics"
        and not has_occupation_signal(combined)
    )
    return {
        "intent": intent,
        "user_goal": clean_text(combined or "Help me make a buying decision.", 180),
        "search_target": clean_text(product_hint or combined, 140),
        "category_hint": category,
        "is_specific_product": bool(product_hint),
        "comparison_targets": [],
        "ambiguity_level": "vague" if not product_hint else "moderate",
        "needs_budget": intent == "recommend" and not has_budget_signal(combined),
        "needs_use_case": (
            intent == "recommend"
            or (category in {"beauty", "clothing", "fitness"} and intent in {"evaluate", "compare"})
        ) and not has_use_case_signal(combined),
        "needs_skin_type": category == "beauty" and not has_skin_signal(combined),
        "needs_occupation_context": is_close_spec_compare,
        "known_constraints": [],
        "inferred_profile_signals": {},
    }


def format_pets(pets_data) -> str:
    """Format pets list into a readable string."""
    if not pets_data or not isinstance(pets_data, list):
        return ""
    parts = []
    for pet in pets_data[:3]:
        if isinstance(pet, dict):
            name = pet.get("breed") or pet.get("species") or "pet"
            age = pet.get("age")
            health = pet.get("health_notes")
            desc = name
            if age:
                desc += f"（{age}）"
            if health:
                desc += f"，{health}"
            parts.append(desc)
        elif isinstance(pet, str):
            parts.append(pet)
    return "、".join(parts)


def format_seed_list(items, limit: int = 4) -> str:
    if not isinstance(items, list):
        return ""
    return "、".join([str(item) for item in items[:limit] if item])


def summarize_memory_seed(memory_seed: dict) -> list[str]:
    if not isinstance(memory_seed, dict) or not memory_seed:
        return []

    sections = []
    identity = memory_seed.get("identity_label")
    identity_detail = memory_seed.get("identity_detail_label")
    if identity:
        if identity_detail:
            sections.append(f"【基础状态】{identity}｜{identity_detail}")
        else:
            sections.append(f"【基础状态】{identity}")

    household = memory_seed.get("household_label")
    household_detail = memory_seed.get("household_detail_label")
    if household:
        if household_detail:
            sections.append(f"【居住 / 家庭】{household}｜{household_detail}")
        else:
            sections.append(f"【居住 / 家庭】{household}")

    value_drivers = format_seed_list(memory_seed.get("value_drivers"), limit=3)
    if value_drivers:
        sections.append(f"【买东西更看重】{value_drivers}")

    shopping_for = format_seed_list(memory_seed.get("shopping_for"), limit=3)
    if shopping_for:
        sections.append(f"【常为谁买】{shopping_for}")

    categories = format_seed_list(memory_seed.get("high_frequency_categories"), limit=3)
    if categories:
        sections.append(f"【高频消费品类】{categories}")

    spend_profile = memory_seed.get("high_frequency_spend_profile_label")
    if spend_profile:
        sections.append(f"【这些高频品类你通常会这样花】{spend_profile}")

    return sections


def build_profile_glimpse(profile: dict) -> str:
    """Person-centric profile summary. Passed to LLM as structured context."""
    sections = []

    # Who they are
    who = []
    if profile.get("occupation"):
        who.append(f"职业：{profile['occupation']}")
    if profile.get("age_group"):
        age_label = {"student": "学生", "young_professional": "职场新人", "other": "其他"}.get(profile["age_group"], profile["age_group"])
        who.append(f"阶段：{age_label}")
    if profile.get("family_context"):
        who.append(f"家庭：{profile['family_context']}")
    if who:
        sections.append("【关于这个人】" + "  |  ".join(who))

    # Pets
    pets_str = format_pets(profile.get("pets"))
    if pets_str:
        sections.append(f"【宠物】{pets_str}")

    # Skin
    if profile.get("skin_type"):
        sections.append(f"【肤质】{profile['skin_type']}")

    # Budget
    if profile.get("budget_sensitivity"):
        budget_label = {"tight": "预算有限", "moderate": "适中", "flexible": "预算充裕"}.get(profile["budget_sensitivity"], profile["budget_sensitivity"])
        sections.append(f"【预算敏感度】{budget_label}")

    # Use patterns
    uses = profile.get("primary_use_cases") or profile.get("use_cases") or []
    if uses:
        sections.append(f"【使用习惯】{', '.join(uses[:4])}")

    # Lifestyle
    if profile.get("lifestyle_hints"):
        sections.append(f"【生活方式】{', '.join(profile['lifestyle_hints'][:4])}")

    sections.extend(summarize_memory_seed(profile.get("memory_seed") or {}))

    return "\n".join(sections) if sections else "No stored profile data."


def understand_request(
    input_text: str,
    product_hint: str,
    has_image: bool,
    user_context: str,
    followup_qa: list | None = None,
    callback: ProgressCallback | None = None,
) -> dict:
    emit_progress(
        callback,
        type="status",
        step="understand",
        label="Understanding your goal",
        detail="Figuring out what you really want solved before deciding how to search.",
    )

    followup_text = joined_followup_text(followup_qa)
    prompt = f"""USER TEXT:
{input_text or ""}

IMAGE PRODUCT HINT:
{product_hint or ""}

HAS IMAGE:
{has_image}

FOLLOW-UP CONTEXT:
{followup_text or "None"}

PROFILE GLIMPSE:
{user_context}
"""

    try:
        response = _fast_json_model(UNDERSTAND_PROMPT).generate_content(prompt)
        data = parse_json(response.text)
    except Exception:
        data = heuristic_understanding(input_text, product_hint, followup_qa)

    combined_text = f"{input_text or ''} {product_hint or ''} {followup_text}".strip()
    inferred_category = infer_category_hint(input_text or "", product_hint or "", followup_text)
    data.setdefault("intent", "evaluate" if product_hint else "recommend")
    data.setdefault("user_goal", clean_text(combined_text or "Help me decide what to buy.", 180))
    data.setdefault("search_target", clean_text(product_hint or combined_text, 140))
    data.setdefault("category_hint", inferred_category)
    data.setdefault("is_specific_product", bool(product_hint))
    data.setdefault("comparison_targets", [])
    data.setdefault("ambiguity_level", "moderate")
    data.setdefault("needs_budget", False)
    data.setdefault("needs_use_case", False)
    data.setdefault("needs_skin_type", False)
    data.setdefault("needs_occupation_context", False)
    data.setdefault("known_constraints", [])
    data.setdefault("inferred_profile_signals", {})

    if not data.get("category_hint") or data.get("category_hint") not in RELATED_CATEGORY_MAP:
        data["category_hint"] = inferred_category
    if not data.get("search_target"):
        data["search_target"] = clean_text(product_hint or combined_text or data["user_goal"], 140)

    if has_budget_signal(combined_text):
        data["needs_budget"] = False
    if has_use_case_signal(combined_text):
        data["needs_use_case"] = False
    if has_skin_signal(combined_text):
        data["needs_skin_type"] = False
    if has_occupation_signal(combined_text):
        data["needs_occupation_context"] = False

    emit_progress(
        callback,
        type="status",
        step="understand",
        label="Goal understood",
        detail=f"Intent: {data['intent']} · {clean_text(data['user_goal'], 120)}",
    )
    return data


def filter_relevant_history(history: list, understanding: dict) -> tuple[list, list]:
    category = understanding.get("category_hint", "other") or "other"
    related_categories = RELATED_CATEGORY_MAP.get(category, {category})
    target = (understanding.get("search_target") or understanding.get("user_goal") or "").lower()
    target_words = [word for word in re.findall(r"[a-z0-9\u4e00-\u9fff]+", target) if len(word) >= 2]
    current_family = infer_product_family(target)

    relevant = []
    ignored = []
    for session in history or []:
        session_category = (session.get("category") or "other").lower() or "other"
        session_name = (session.get("product_name") or "").lower()
        session_family = infer_product_family(session_name)
        same_family = current_family != "generic" and session_family == current_family
        weak_name_match = target_words and any(word in session_name for word in target_words[:4])

        if same_family:
            relevant.append(session)
        elif current_family == "generic" and session_category in related_categories:
            relevant.append(session)
        elif weak_name_match and session_category in related_categories:
            relevant.append(session)
        else:
            ignored.append(session)
    return relevant[:4], ignored


def build_user_context(profile: dict, history: list, understanding: dict) -> tuple[str, dict]:
    relevant_history, ignored_history = filter_relevant_history(history, understanding)
    category = understanding.get("category_hint", "other") or "other"
    memory_seed = profile.get("memory_seed") or {}

    parts = [
        "Memory policy: only use memory that is relevant to this category or current goal. Ignore unrelated past purchases.",
    ]

    if profile.get("occupation"):
        parts.append(f"User occupation: {profile['occupation']}.")
    if profile.get("age_group"):
        parts.append(f"Life stage: {profile['age_group']}.")

    if category == "beauty" and profile.get("skin_type"):
        parts.append(f"Relevant profile signal: skin type is {profile['skin_type']}.")

    if profile.get("primary_use_cases"):
        parts.append(
            f"Primary use patterns (from past context): {', '.join(profile['primary_use_cases'][:3])}."
        )
    elif profile.get("use_cases"):
        parts.append(
            f"Saved lifestyle hints (soft only, current request wins): {', '.join(profile['use_cases'][:3])}."
        )

    if profile.get("lifestyle_hints"):
        parts.append(f"Lifestyle signals: {', '.join(profile['lifestyle_hints'][:3])}.")

    if isinstance(memory_seed, dict) and memory_seed:
        value_drivers = memory_seed.get("value_drivers") or []
        if value_drivers:
            parts.append(f"Stable value drivers from onboarding: {', '.join(value_drivers[:3])}.")

        shopping_for = memory_seed.get("shopping_for") or []
        if shopping_for:
            parts.append(f"They often shop for: {', '.join(shopping_for[:3])}.")

        high_frequency_categories = memory_seed.get("high_frequency_categories") or []
        if high_frequency_categories:
            parts.append(f"High-frequency spending areas: {', '.join(high_frequency_categories[:3])}.")

        spend_profile = memory_seed.get("high_frequency_spend_profile_label")
        if spend_profile:
            parts.append(f"For their high-frequency categories, their overall spend style is: {spend_profile}.")

    if relevant_history:
        recent = [
            f"{item.get('product_name', 'Unknown')} ({item.get('category', 'other')}, {item.get('verdict', '')})"
            for item in relevant_history
            if item.get("product_name")
        ]
        if recent:
            parts.append("Relevant past decisions: " + "; ".join(recent) + ".")

    if ignored_history:
        parts.append(
            f"Ignored unrelated memory from {len(ignored_history)} other-category decisions so it does not bias this answer."
        )

    return "\n".join(parts), {
        "used_history_count": len(relevant_history),
        "ignored_history_count": len(ignored_history),
        "used_profile_signals": [
            signal
            for signal, enabled in {
                "occupation": bool(profile.get("occupation")),
                "skin_type": category == "beauty" and bool(profile.get("skin_type")),
                "use_cases_soft": bool(profile.get("use_cases") or profile.get("primary_use_cases")),
                "lifestyle_hints": bool(profile.get("lifestyle_hints")),
                "memory_seed": bool(memory_seed),
            }.items()
            if enabled
        ],
    }


def build_scoping_plan(understanding: dict) -> list[tuple[str, str, str]]:
    intent = understanding.get("intent", "evaluate")
    target = understanding.get("search_target") or understanding.get("user_goal") or ""
    if intent == "compare" and understanding.get("comparison_targets"):
        target = " vs ".join(understanding["comparison_targets"])
    plans = SCOPING_PLANS_ZH if prefers_chinese(target) else SCOPING_PLANS
    plan = plans.get(intent, plans["evaluate"])
    return [(key, label, template.format(target=target)) for key, label, template in plan]


def build_search_plan(understanding: dict, clarification: dict | None = None) -> list[tuple[str, str, str]]:
    intent = understanding.get("intent", "evaluate")
    target = understanding.get("search_target") or understanding.get("user_goal") or ""
    if intent == "compare" and understanding.get("comparison_targets"):
        target = " vs ".join(understanding["comparison_targets"])
    if clarification:
        focus = " ".join([clean_text(item, 40) for item in clarification.get("search_focus", [])[:2] if item])
        if focus:
            target = f"{target} {focus}".strip()
    constraints = understanding.get("known_constraints", [])
    if constraints:
        constraint_str = " ".join(constraints[:2])
        target = f"{target} {constraint_str}".strip()

    plans = SEARCH_PLANS_ZH if prefers_chinese(target) else SEARCH_PLANS
    plan = plans.get(intent, plans["evaluate"])
    return [(key, label, template.format(target=target)) for key, label, template in plan]


def run_one_search(query: str, search_depth: str, max_results: int) -> dict:
    response = tavily.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
        include_answer=True,
    )
    formatted_results = [
        {
            "title": clean_text(item.get("title", ""), 90),
            "url": item.get("url", ""),
            "site": extract_site(item.get("url", "")),
            "snippet": clean_text(item.get("content", ""), 180),
        }
        for item in response.get("results", [])
    ]
    return {
        "answer": clean_text(response.get("answer", ""), 240),
        "results": formatted_results,
        "error": "",
    }


def search_multi(
    search_plan: list[tuple[str, str, str]],
    callback: ProgressCallback | None = None,
    phase: str = "full",
) -> dict:
    results = {}
    if not search_plan:
        return results

    search_depth = "basic" if phase == "scoping" else "advanced"
    max_results = 2 if phase == "scoping" else 3
    max_workers = min(4, len(search_plan))

    for _, label, query in search_plan:
        emit_progress(
            callback,
            type="status",
            step="search",
            label=label,
            detail=f"Searching: {query}",
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_one_search, query, search_depth, max_results): (key, label, query)
            for key, label, query in search_plan
        }
        for future in as_completed(future_map):
            key, label, query = future_map[future]
            try:
                payload = future.result()
                results[key] = {
                    "label": label,
                    "query": query,
                    **payload,
                }
                emit_progress(
                    callback,
                    type="status",
                    step="search",
                    label=label,
                    detail=f"Found {len(payload['results'])} sources for {label.lower()}.",
                )
            except Exception as exc:
                results[key] = {
                    "label": label,
                    "query": query,
                    "answer": "",
                    "results": [],
                    "error": clean_text(str(exc), 160),
                }
                emit_progress(
                    callback,
                    type="status",
                    step="search",
                    label=label,
                    detail=f"Search failed: {clean_text(str(exc), 120)}",
                )

    ordered = {}
    for key, _, _ in search_plan:
        if key in results:
            ordered[key] = results[key]
    return ordered


def format_search_block(search_data: dict) -> str:
    blocks = []
    for bucket in search_data.values():
        lines = []
        for item in bucket.get("results", [])[:4]:
            title = item.get("title") or item.get("site") or "Untitled"
            site = item.get("site") or "unknown"
            snippet = item.get("snippet") or "No snippet"
            lines.append(f"- {title} ({site}): {snippet}")
        sources = "\n".join(lines) if lines else "- No usable sources"
        error = f"\nError: {bucket['error']}" if bucket.get("error") else ""
        blocks.append(
            f"=== {bucket['label']} ===\n"
            f"Query: {bucket['query']}\n"
            f"Summary: {bucket.get('answer') or 'No summary'}{error}\n"
            f"Sources:\n{sources}"
        )
    return "\n\n".join(blocks)


def format_scoping_block(search_data: dict) -> str:
    if not search_data:
        return "No scoping evidence."
    blocks = []
    for bucket in search_data.values():
        sources = []
        for item in bucket.get("results", [])[:3]:
            sources.append(
                f"- {item.get('title') or item.get('site') or 'Untitled'} ({item.get('site') or 'unknown'}): {item.get('snippet') or 'No snippet'}"
            )
        blocks.append(
            f"=== {bucket.get('label', 'Scoping')} ===\n"
            f"Query: {bucket.get('query', '')}\n"
            f"Summary: {bucket.get('answer') or 'No summary'}\n"
            f"Sources:\n" + ("\n".join(sources) if sources else "- No usable sources")
        )
    return "\n\n".join(blocks)


def flatten_sources(search_data: dict) -> list:
    seen = set()
    sources = []
    for bucket in search_data.values():
        for item in bucket.get("results", []):
            url = item.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append(
                {
                    "title": item.get("title") or item.get("site") or "Untitled source",
                    "url": url,
                    "site": item.get("site") or "unknown",
                    "bucket": bucket.get("label", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
    return sources[:12]


def build_research_trace(
    product_hint: str,
    understanding: dict,
    search_data: dict,
    memory_meta: dict | None = None,
    followup_qa: list | None = None,
    presearch_followup: dict | None = None,
) -> list:
    trace = [
        {
            "label": "Intent understood",
            "detail": clean_text(understanding.get("user_goal", ""), 160),
        }
    ]
    if product_hint:
        trace.append({"label": "Identified product", "detail": product_hint})

    if memory_meta:
        detail = (
            f"Used {memory_meta.get('used_history_count', 0)} relevant memory signals"
            f" and ignored {memory_meta.get('ignored_history_count', 0)} unrelated history."
        )
        trace.append({"label": "Memory filter", "detail": detail})

    if followup_qa:
        last_answer = clean_text(followup_qa[-1].get("answer", ""), 140)
        if last_answer:
            trace.append({"label": "Your extra context", "detail": last_answer})

    if presearch_followup:
        trace.append(
            {
                "label": "Paused before search",
                "detail": clean_text(presearch_followup.get("reason", ""), 180),
            }
        )

    for bucket in search_data.values():
        sites = [item.get("site") for item in bucket.get("results", []) if item.get("site")]
        detail = bucket.get("answer") or ", ".join(dict.fromkeys(sites).keys()) or "No strong source summary"
        if bucket.get("error"):
            detail = f"{detail} ({bucket['error']})"
        trace.append({"label": bucket.get("label", "Research"), "detail": clean_text(detail, 180)})
    return trace


def build_fact_cards(result: dict, search_data: dict) -> list:
    cards = []
    if result.get("price_range"):
        sites = [
            item.get("site")
            for item in search_data.get("marketplaces", {}).get("results", [])[:3]
            if item.get("site")
        ]
        cards.append(
            {
                "title": "Reference price",
                "value": result["price_range"],
                "detail": f"Cross-checked on {', '.join(dict.fromkeys(sites).keys())}" if sites else "",
            }
        )

    specs = result.get("key_specs") or {}
    if specs:
        preview = " | ".join([f"{k}: {v}" for k, v in list(specs.items())[:3]])
        cards.append(
            {
                "title": "Core specs",
                "value": f"{len(specs)} key signals",
                "detail": preview,
            }
        )

    if result.get("result_type") == "recommendation":
        recommendations = result.get("recommendations") or []
        if recommendations:
            cards.append(
                {
                    "title": "Shortlist prepared",
                    "value": ", ".join([item.get("name", "") for item in recommendations[:2] if item.get("name")]),
                    "detail": "Picked around the scene, tradeoffs, and current market signals.",
                }
            )
    else:
        alternatives = result.get("alternatives") or []
        if alternatives:
            cards.append(
                {
                    "title": "Other options worth a look",
                    "value": ", ".join([item.get("name", "") for item in alternatives[:2] if item.get("name")]),
                    "detail": "Compared with nearby options in the same rough price band.",
                }
            )
    return cards


def build_scoping_fact_cards(strategy: dict, search_data: dict) -> list:
    cards = []
    if strategy.get("preliminary_take"):
        cards.append(
            {
                "title": "Initial read",
                "value": clean_text(strategy["preliminary_take"], 44),
                "detail": "这是系统先基于商品和早期检索抓到的方向判断。",
            }
        )
    dimensions = strategy.get("decision_dimensions") or []
    if dimensions:
        first_two = dimensions[:2]
        cards.append(
            {
                "title": "Key decision angles",
                "value": ", ".join([item.get("title", "") for item in first_two if item.get("title")]),
                "detail": " | ".join([clean_text(item.get("detail", ""), 56) for item in first_two]),
            }
        )
    sources = flatten_sources(search_data)
    if sources:
        cards.append(
            {
                "title": "Early evidence",
                "value": ", ".join(dict.fromkeys([item.get("site", "") for item in sources[:3] if item.get("site")]).keys()),
                "detail": "先做了一轮轻量检索，用来判断接下来该问什么、该查什么。",
            }
        )
    return [card for card in cards if card.get("value")]


def fallback_clarification_strategy(
    understanding: dict,
    user_profile: dict,
    followup_qa: list | None,
    followup_count: int,
    search_data: dict,
) -> dict:
    asked_kinds = {question_kind(item.get("question", "")) for item in (followup_qa or [])}
    combined = " ".join(
        [
            understanding.get("user_goal", ""),
            understanding.get("search_target", ""),
            joined_followup_text(followup_qa),
        ]
    )
    category = understanding.get("category_hint", "other")

    strategy = {
        "preliminary_take": "",
        "decision_dimensions": [],
        "missing_dimension": "none",
        "needs_followup": False,
        "followup_reason": "",
        "question_type": "multiple_choice",
        "followup_question": "",
        "followup_options": [],
        "open_text_placeholder": "",
        "search_focus": [],
    }

    if followup_count >= 3:
        return strategy

    if category == "food" and infer_product_family(combined) == "supplement" and "health_context" not in asked_kinds:
        strategy.update(
            {
                "missing_dimension": "health_context",
                "needs_followup": True,
                "followup_reason": "这类补剂先看你是否适合吃，比先比较品牌参数更关键。",
                "followup_question": "你这次吃它，更接近哪一种情况？",
                "followup_options": ["日常保健 / 想补一补", "有明确不适或化验指标", "医生/营养师建议我补"],
                "search_focus": ["benefit suitability", "contraindications side effects"],
                "decision_dimensions": [
                    {"title": "适用性", "status": "risk", "detail": "先确认你是否真的适合补这个，再谈品牌好坏。"},
                    {"title": "产品质量", "status": "mixed", "detail": "品牌、纯度和剂量也重要，但要排在适用性之后。"},
                ],
            }
        )
        return strategy

    if understanding.get("needs_skin_type") and "skin_type" not in asked_kinds and not user_profile.get("skin_type"):
        strategy.update(
            {
                "missing_dimension": "skin_type",
                "needs_followup": True,
                "followup_reason": "这个品类对肤质很敏感，先确认肤质，后面的判断才不容易偏。",
                "followup_question": "你的肤质更接近哪一种？",
                "followup_options": ["干皮 / 混干", "油皮 / 混油", "敏感肌"],
                "search_focus": ["skin type fit", "sensitive skin issues"],
            }
        )
        return strategy

    if understanding.get("needs_use_case") and "use_case" not in asked_kinds and not has_use_case_signal(combined):
        strategy.update(
            {
                "missing_dimension": "use_case",
                "needs_followup": True,
                "followup_reason": "这件东西在不同场景下标准差很大，先知道你怎么用，判断才会像给你量身做的。",
                "followup_question": "你最主要想把它用在什么场景？",
                "followup_options": ["通勤 / 日常", "居家 / 办公", "运动 / 高频使用"],
                "search_focus": ["fit by scenario", "comfort durability complaints"],
            }
        )
        return strategy

    if understanding.get("needs_budget") and "budget" not in asked_kinds and not has_budget_signal(combined):
        strategy.update(
            {
                "missing_dimension": "budget",
                "needs_followup": True,
                "followup_reason": "预算会改变候选集合和推荐策略，先卡住范围更高效。",
                "followup_question": "你这次大概想把预算放在哪一档？",
                "followup_options": ["先控制预算", "愿意多花一点换明显升级", "预算不是第一优先级"],
                "search_focus": ["price tier", "best value options"],
            }
        )
        return strategy

    strategy["decision_dimensions"] = [
        {"title": "场景适配", "status": "mixed", "detail": "系统会看它到底适不适合你的实际使用方式。"},
        {"title": "产品本身表现", "status": "mixed", "detail": "系统会看规格、口碑、价格和常见槽点。"},
    ]
    return strategy


def generate_clarification_strategy(
    understanding: dict,
    product_hint: str,
    user_profile: dict,
    followup_qa: list | None,
    followup_count: int,
    search_data: dict,
    user_context: str,
    callback: ProgressCallback | None = None,
) -> dict:
    emit_progress(
        callback,
        type="status",
        step="clarify",
        label="Checking what still matters",
        detail="先用一轮轻量检索看清判断维度，再决定要不要问你问题。",
    )

    prompt = f"""INTENT UNDERSTANDING:
{json.dumps(understanding, ensure_ascii=False)}

PRODUCT / TARGET:
{product_hint or understanding.get("search_target", "")}

USER CONTEXT:
{user_context}

FOLLOW-UP CONTEXT:
{joined_followup_text(followup_qa) or "None"}

EARLY WEB EVIDENCE:
{format_scoping_block(search_data)}
"""

    try:
        response = _fast_json_model(SCOPING_PROMPT).generate_content(prompt)
        strategy = parse_json(response.text)
    except Exception:
        strategy = fallback_clarification_strategy(
            understanding=understanding,
            user_profile=user_profile,
            followup_qa=followup_qa,
            followup_count=followup_count,
            search_data=search_data,
        )

    strategy.setdefault("preliminary_take", "")
    strategy.setdefault("decision_dimensions", [])
    strategy.setdefault("missing_dimension", "none")
    strategy.setdefault("needs_followup", False)
    strategy.setdefault("followup_reason", "")
    strategy.setdefault("question_type", "multiple_choice")
    strategy.setdefault("followup_question", "")
    strategy.setdefault("followup_options", [])
    strategy.setdefault("open_text_placeholder", "")
    strategy.setdefault("search_focus", [])
    strategy.setdefault("followup_questions", [])

    if followup_count >= 3:
        strategy["needs_followup"] = False

    combined = " ".join(
        [
            product_hint or "",
            understanding.get("search_target", ""),
            understanding.get("user_goal", ""),
            joined_followup_text(followup_qa),
        ]
    )
    family = infer_product_family(combined)
    asked_kinds = {question_kind(item.get("question", "")) for item in (followup_qa or [])}

    framework_followup = build_framework_followup(
        understanding=understanding,
        product_hint=product_hint,
        user_profile=user_profile or {},
        followup_qa=followup_qa,
        followup_count=followup_count,
        search_data=search_data,
    )

    if framework_followup:
        strategy.update(framework_followup)
        strategy["followup_questions"] = [
            {
                "question": strategy["followup_question"],
                "options": strategy.get("followup_options", []),
                "reason": strategy.get("followup_reason", ""),
                "question_type": strategy.get("question_type", "multiple_choice"),
                "open_text_placeholder": strategy.get("open_text_placeholder", ""),
            }
        ]
    else:
        strategy["needs_followup"] = False
        strategy["missing_dimension"] = "none"
        strategy["followup_reason"] = ""
        strategy["followup_question"] = ""
        strategy["followup_options"] = []
        strategy["followup_questions"] = []
        strategy["open_text_placeholder"] = ""

    # Ensure decision_dimensions has at least a default if empty
    if not strategy.get("decision_dimensions"):
        strategy["decision_dimensions"] = fallback_clarification_strategy(
            understanding=understanding,
            user_profile=user_profile,
            followup_qa=followup_qa,
            followup_count=followup_count,
            search_data=search_data,
        ).get("decision_dimensions", [])

    return strategy


def default_followup(
    understanding: dict,
    user_profile: dict,
    followup_count: int,
    followup_qa: list | None = None,
) -> dict | None:
    if followup_count >= 3:
        return None

    combined = " ".join(
        [
            understanding.get("user_goal", ""),
            understanding.get("search_target", ""),
            joined_followup_text(followup_qa),
        ]
    )
    asked_kinds = {question_kind(item.get("question", "")) for item in (followup_qa or [])}

    if understanding.get("intent") == "recommend" and not has_budget_signal(combined) and "budget" not in asked_kinds:
        return {
            "question": "如果要我把推荐再收得更准一点，你更在意哪边？",
            "options": ["价格尽量稳妥", "舒适/效果更重要", "我想在两者之间平衡"],
        }

    if understanding.get("category_hint") == "beauty" and not user_profile.get("skin_type") and "skin_type" not in asked_kinds:
        return {
            "question": "最后再确认一下，你的肤质更偏哪一种？",
            "options": ["干皮 / 混干", "油皮 / 混油", "敏感肌"],
        }

    return None


def build_early_followup_result(
    understanding: dict,
    product_hint: str,
    clarification: dict,
    search_data: dict,
    memory_meta: dict,
) -> dict:
    result_type = "recommendation" if understanding.get("intent") == "recommend" and not understanding.get("is_specific_product") else "decision"

    # Prefer the full array; fall back to single-question fields for backward compat
    followup_questions = clarification.get("followup_questions") or []
    if not followup_questions and clarification.get("followup_question"):
        followup_questions = [{
            "question": clarification["followup_question"],
            "options": clarification.get("followup_options", []),
            "reason": clarification.get("followup_reason", ""),
            "question_type": clarification.get("question_type", "multiple_choice"),
            "open_text_placeholder": clarification.get("open_text_placeholder", ""),
        }]

    primary_followup = followup_questions[0] if followup_questions else {
        "question": clarification.get("followup_question", ""),
        "options": clarification.get("followup_options", []),
        "reason": clarification.get("followup_reason", ""),
        "question_type": clarification.get("question_type", "multiple_choice"),
        "open_text_placeholder": clarification.get("open_text_placeholder", ""),
    }
    question_count = len(followup_questions) or 1
    return {
        "result_type": result_type,
        "intent": understanding.get("intent", "evaluate"),
        "headline": "先问你一句，再继续帮你查",
        "product_name": product_hint if result_type == "decision" else "",
        "category": understanding.get("category_hint", "other"),
        "price_range": "",
        "verdict": None if result_type == "recommendation" else "cautious",
        "summary": clarification.get("preliminary_take") or f"我先做了个快速判断，这里还差 {question_count} 个会明显影响结论的关键信息。",
        "decision_dimensions": clarification.get("decision_dimensions", []),
        "reasons": [],
        "key_specs": {},
        "alternatives": [],
        "recommendations": [],
        "followup": {
            "question": primary_followup["question"],
            "options": primary_followup.get("options", []),
            "question_type": primary_followup.get("question_type", clarification.get("question_type", "multiple_choice")),
            "open_text_placeholder": primary_followup.get("open_text_placeholder", clarification.get("open_text_placeholder", "")),
        },
        "followup_questions": followup_questions or [primary_followup],
        "fit_summary": "",
        "caution_check": "",
        "buy_if": "",
        "skip_if": "",
        "primary_recommendation": None,
        "budget_alternative": None,
        "better_direction": "",
        "research": {
            "trace": build_research_trace(
                product_hint=product_hint,
                understanding=understanding,
                search_data=search_data,
                memory_meta=memory_meta,
                presearch_followup={
                    "reason": clarification.get("followup_reason", ""),
                },
            ),
            "queries": [
                {"label": bucket["label"], "query": bucket["query"], "error": bucket.get("error", "")}
                for bucket in search_data.values()
            ],
            "sources": flatten_sources(search_data),
            "fact_cards": build_scoping_fact_cards(clarification, search_data),
        },
        "report_sections": [],
        "followup_reason": clarification.get("followup_reason", ""),
    }


def normalize_result(
    result: dict,
    product_hint: str,
    understanding: dict,
    search_data: dict,
    user_profile: dict,
    followup_count: int,
    memory_meta: dict | None = None,
    followup_qa: list | None = None,
) -> dict:
    intent = understanding.get("intent", "evaluate")
    result_type = result.get("result_type")
    if result_type not in {"decision", "recommendation"}:
        result_type = "recommendation" if intent == "recommend" or not understanding.get("is_specific_product") else "decision"

    result["result_type"] = result_type
    result["intent"] = intent
    result.setdefault("headline", "我先帮你挑了几款" if result_type == "recommendation" else "")
    result.setdefault("product_name", product_hint if result_type == "decision" else "")
    result.setdefault("category", understanding.get("category_hint", "other") or "other")
    result.setdefault("price_range", "")
    result.setdefault("verdict", None if result_type == "recommendation" else "cautious")
    result.setdefault("summary", "")
    result.setdefault("decision_dimensions", [])
    result.setdefault("reasons", [])
    result.setdefault("key_specs", {})
    result.setdefault("alternatives", [])
    result.setdefault("recommendations", [])
    result.setdefault("followup", None)
    result.setdefault("fit_summary", "")
    result.setdefault("caution_check", "")
    result.setdefault("buy_if", "")
    result.setdefault("skip_if", "")
    result.setdefault("primary_recommendation", None)
    result.setdefault("budget_alternative", None)
    result.setdefault("better_direction", "")

    if result_type == "recommendation" and not result.get("recommendations"):
        cards = []
        if result.get("primary_recommendation"):
            primary = result["primary_recommendation"]
            cards.append(
                {
                    "name": primary.get("name", ""),
                    "price_hint": "",
                    "best_for": understanding.get("user_goal", ""),
                    "reason": primary.get("reason", ""),
                    "tradeoffs": primary.get("better_points", []),
                }
            )
        if result.get("budget_alternative"):
            budget = result["budget_alternative"]
            cards.append(
                {
                    "name": budget.get("name", ""),
                    "price_hint": "",
                    "best_for": "预算更紧的时候",
                    "reason": budget.get("reason", ""),
                    "tradeoffs": [],
                }
            )
        result["recommendations"] = [card for card in cards if card.get("name")]

    if not result.get("followup"):
        result["followup"] = default_followup(
            understanding,
            user_profile or {},
            followup_count,
            followup_qa=followup_qa,
        )

    if result_type == "recommendation":
        if not result.get("headline"):
            result["headline"] = "我帮你先挑了这几款"
        report_sections = [
            {
                "title": "为什么这更像是给你挑的",
                "body": result.get("fit_summary") or result.get("summary") or understanding.get("user_goal", ""),
            }
        ]
        if result.get("better_direction"):
            report_sections.append(
                {
                    "title": "如果你想继续收窄方向",
                    "body": result["better_direction"],
                }
            )
    else:
        if not result.get("headline"):
            headline_map = {
                "worth_buying": "这件可以认真考虑",
                "cautious": "先别急着下单",
                "not_recommended": "这次我更想劝你先放下",
            }
            result["headline"] = headline_map.get(result.get("verdict"), "给你一个更稳妥的判断")
        report_sections = []
        if result.get("verdict") == "worth_buying":
            report_sections.append(
                {
                    "title": "我为什么会偏向你买",
                    "body": result.get("fit_summary") or result.get("summary") or "",
                }
            )
        elif result.get("verdict") == "cautious":
            report_sections.append(
                {
                    "title": "我会先替你盯住这一点",
                    "body": result.get("caution_check") or result.get("summary") or "",
                }
            )
            if result.get("buy_if"):
                report_sections.append({"title": "如果你是这种情况，还是可以考虑", "body": result["buy_if"]})
            if result.get("skip_if"):
                report_sections.append({"title": "如果不是这种情况，更建议先等等", "body": result["skip_if"]})
        else:
            if result.get("better_direction"):
                report_sections.append({"title": "比继续纠结这款更重要的是", "body": result["better_direction"]})
            if result.get("primary_recommendation"):
                primary = result["primary_recommendation"]
                points = "；".join(primary.get("better_points", [])[:2])
                detail = primary.get("reason", "")
                if points:
                    detail = f"{detail} 核心差距：{points}"
                report_sections.append(
                    {
                        "title": "我更想让你看的替代项",
                        "body": f"{primary.get('name', '替代项')}：{detail}".strip("："),
                    }
                )
            if result.get("budget_alternative"):
                budget = result["budget_alternative"]
                report_sections.append(
                    {
                        "title": "如果你更想把钱花得稳一点",
                        "body": f"{budget.get('name', '备选项')}：{budget.get('reason', '')}".strip("："),
                    }
                )

        if result.get("fit_summary"):
            report_sections.insert(
                0,
                {
                    "title": "它和你的情况到底对不对得上",
                    "body": result["fit_summary"],
                },
            )

    if result.get("decision_dimensions"):
        dimension_lines = []
        for item in result["decision_dimensions"][:4]:
            title = item.get("title", "").strip()
            detail = item.get("detail", "").strip()
            status = item.get("status", "").strip()
            line = f"{title}"
            if status:
                line += f"：{status}"
            if detail:
                line += f" - {detail}"
            dimension_lines.append(line)
        if dimension_lines:
            report_sections.insert(
                0,
                {
                    "title": "这次判断我重点拆了哪几层",
                    "body": "；".join(dimension_lines),
                },
            )

    result["research"] = {
        "trace": build_research_trace(
            product_hint=product_hint,
            understanding=understanding,
            search_data=search_data,
            memory_meta=memory_meta,
            followup_qa=followup_qa,
        ),
        "queries": [
            {"label": bucket["label"], "query": bucket["query"], "error": bucket.get("error", "")}
            for bucket in search_data.values()
        ],
        "sources": flatten_sources(search_data),
        "fact_cards": build_fact_cards(result, search_data),
    }
    result["report_sections"] = [section for section in report_sections if section.get("body")]
    return result


PROFILE_EXTRACT_PROMPT = """You are a person-centric profile extractor for a consumer decision assistant.
Your job is to build a picture of WHO this person is — their life, their relationships, their context —
not just what product they're shopping for right now.

Given a follow-up question and the user's answer, extract structured profile signals.

Return ONLY valid JSON:
{
  "occupation": "<job title or field, or null>",
  "primary_use_cases": ["<use case 1>", "<use case 2>"],
  "age_group": "student" | "young_professional" | "other" | null,
  "lifestyle_hints": ["<hint 1>", "<hint 2>"],
  "skin_type": "<dry|oily|combination|sensitive|null>",
  "budget_sensitivity": "<tight|moderate|flexible|null>",
  "owned_items": ["<item 1>", "<item 2>"],
  "use_cases": ["<use case 1>", "<use case 2>"],
  "pets": [
    {
      "species": "<dog|cat|rabbit|fish|other>",
      "breed": "<breed name or null>",
      "age": "<e.g. 2岁|puppy|senior|null>",
      "health_notes": "<any mentioned health conditions, or null>"
    }
  ],
  "family_context": "<brief note on household if mentioned, e.g. '和伴侣同住', '有小孩', or null>"
}

Rules:
- Only store LONG-LIVED, REUSABLE facts about the user. Do not store temporary details that only apply
  to the current product decision.
- Only populate fields clearly supported by the answer. Set others to null or [].
- NEVER infer what the user DOES based on their job title alone. Only record what they explicitly say.
  BAD: user says "我是产品经理" → do NOT add "uses design tools" to lifestyle_hints.
  GOOD: user says "我是产品经理，平时要做很多原型" → record "prototyping" in primary_use_cases.
- If the question is about the CURRENT ITEM only (e.g. "这次怎么用它", "这次预算多少"),
  do not turn that into permanent memory unless it also clearly reflects a stable user trait.
- Never invent owned_items. Only fill owned_items if the question explicitly asked what they already own
  and the answer explicitly named those items.
- For pets: extract every pet the user mentions. Capture breed, age, and any health issues they name.
  Do not invent health conditions not stated.
- Infer occupation from job titles, school majors, or descriptions of work/study.
- Infer age_group: "student" if they mention 大学/高中/学生/university;
  "young_professional" if they mention 刚工作/上班族/first job.
- Infer lifestyle_hints from activities explicitly mentioned (running, gaming, travel, 摄影, etc.).
- For skin_type: map 干皮/混干→dry, 油皮/混油→oily, 敏感肌→sensitive.
- Keep all values concise. No elaboration.
"""


def should_persist_followup_answer(question: str, answer: str) -> bool:
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return False

    kind = question_kind(q)
    if kind in {"skin_type", "occupation_context", "pet_life_stage", "pet_sensitivity", "family_context"}:
        return True

    if kind == "budget":
        return "这次" not in q and "这一单" not in q

    if kind == "use_case":
        return any(token in q for token in ["平时", "通常", "一般", "长期"]) and not any(token in q for token in ["这次", "这款", "这个", "这件", "这双", "这台", "它"])

    if kind == "owned_items":
        return any(token in q for token in ["已有", "已经有", "手上已经"])

    return False


def extract_profile_signals_from_answer(question: str, answer: str, original_input: str = "") -> dict:
    if not should_persist_followup_answer(question, answer):
        return {}

    prompt = f"QUESTION: {question}\n\nANSWER: {answer}\n\nCURRENT PRODUCT CONTEXT: {original_input}"
    try:
        response = _fast_json_model(PROFILE_EXTRACT_PROMPT).generate_content(prompt)
        signals = parse_json(response.text)
    except Exception:
        # Fallback: simple keyword matching
        question_lower = (question or "").lower()
        answer_stripped = (answer or "").strip()
        kind = question_kind(question)
        signals = {}
        if kind == "skin_type":
            signals["skin_type"] = answer_stripped
        elif kind == "occupation_context":
            signals["occupation"] = answer_stripped
        elif kind == "use_case":
            signals["primary_use_cases"] = [answer_stripped]
        elif kind == "budget":
            signals["budget_sensitivity"] = answer_stripped
        elif kind == "owned_items":
            signals["owned_items"] = [answer_stripped]
        elif kind in {"pet_life_stage", "pet_sensitivity"}:
            species = infer_pet_species(question, answer, original_input)
            pet_signal = {"species": species}
            if kind == "pet_life_stage":
                pet_signal["age"] = answer_stripped
            if kind == "pet_sensitivity":
                pet_signal["health_notes"] = answer_stripped
            signals["pets"] = [pet_signal]

    # Remove null values and empty lists before returning
    return {k: v for k, v in signals.items() if v is not None and v != [] and v != ""}


def generate_final_result(
    input_text: str,
    product_hint: str,
    understanding: dict,
    clarification: dict,
    scoping_data: dict,
    search_data: dict,
    user_context: str,
    followup_qa: list | None = None,
    pil_image: Image.Image | None = None,
    callback: ProgressCallback | None = None,
) -> dict:
    emit_progress(
        callback,
        type="status",
        step="synthesize",
        label="Forming the answer",
        detail="Combining the web evidence, your context, and what matters most to you.",
    )

    followup_text = ""
    if followup_qa:
        qa_lines = [f"Q: {item['question']}\nA: {item['answer']}" for item in followup_qa]
        followup_text = "FOLLOW-UP CONTEXT:\n" + "\n".join(qa_lines) + "\n\n"

    prompt = f"""USER INPUT:
{input_text}

IDENTIFIED PRODUCT:
{product_hint}

INTENT UNDERSTANDING:
{json.dumps(understanding, ensure_ascii=False)}

SCOPING NOTES:
{json.dumps(clarification, ensure_ascii=False)}

MEMORY + USER CONTEXT:
{user_context}

{followup_text}EARLY SCOPING SEARCH:
{format_scoping_block(scoping_data)}

WEB SEARCH RESULTS:
{format_search_block(search_data)}

Return the JSON answer now."""

    model = _json_model(FINAL_PROMPT)
    if pil_image:
        response = model.generate_content([pil_image, prompt])
    else:
        response = model.generate_content(prompt)
    return parse_json(response.text)


def analyze(
    input_text: str,
    image_base64: str = None,
    user_profile: dict = None,
    user_history: list = None,
    analysis_mode: str = "decision",
    callback: ProgressCallback | None = None,
) -> dict:
    emit_progress(
        callback,
        type="status",
        step="identify",
        label="Reading your input",
        detail="Checking whether you gave a specific product or a broader shopping need.",
    )

    pil_image = None
    product_hint = input_text or ""
    if image_base64:
        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        emit_progress(
            callback,
            type="status",
            step="identify",
            label="Inspecting the image",
            detail="Identifying the product or category from the uploaded image.",
        )
        if not input_text:
            product_hint = identify_product_from_image(pil_image)

    profile_glimpse = build_profile_glimpse(user_profile or {})
    understanding = understand_request(
        input_text=input_text or product_hint,
        product_hint=product_hint,
        has_image=bool(pil_image),
        user_context=profile_glimpse,
        callback=callback,
    )
    user_context, memory_meta = build_user_context(user_profile or {}, user_history or [], understanding)

    scoping_plan = build_scoping_plan(understanding)
    scoping_data = search_multi(scoping_plan, callback=callback, phase="scoping")
    clarification = generate_clarification_strategy(
        understanding=understanding,
        product_hint=product_hint,
        user_profile=user_profile or {},
        followup_qa=[],
        followup_count=0,
        search_data=scoping_data,
        user_context=user_context,
        callback=callback,
    )
    if clarification.get("needs_followup"):
        emit_progress(
            callback,
            type="status",
            step="clarify",
            label="One thing to confirm first",
            detail=clarification["followup_reason"],
        )
        return build_early_followup_result(
            understanding=understanding,
            product_hint=product_hint,
            clarification=clarification,
            search_data=scoping_data,
            memory_meta=memory_meta,
        )

    search_plan = build_search_plan(understanding, clarification=clarification)
    search_data = search_multi(search_plan, callback=callback)
    raw_result = generate_final_result(
        input_text=input_text or product_hint,
        product_hint=product_hint,
        understanding=understanding,
        clarification=clarification,
        scoping_data=scoping_data,
        search_data=search_data,
        user_context=user_context,
        pil_image=pil_image,
        callback=callback,
    )
    return normalize_result(
        result=raw_result,
        product_hint=product_hint,
        understanding=understanding,
        search_data=search_data,
        user_profile=user_profile or {},
        followup_count=0,
        memory_meta=memory_meta,
        followup_qa=[],
    )


def analyze_with_followup(
    original_input: str,
    followup_qa: list,
    user_profile: dict = None,
    user_history: list = None,
    analysis_mode: str = "decision",
    image_base64: str = None,
    callback: ProgressCallback | None = None,
) -> dict:
    pil_image = None
    product_hint = original_input
    if image_base64:
        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        if original_input in {"", "[image]"}:
            product_hint = identify_product_from_image(pil_image)

    profile_glimpse = build_profile_glimpse(user_profile or {})
    understanding = understand_request(
        input_text=original_input,
        product_hint=product_hint,
        has_image=bool(pil_image),
        user_context=profile_glimpse,
        followup_qa=followup_qa,
        callback=callback,
    )
    user_context, memory_meta = build_user_context(user_profile or {}, user_history or [], understanding)

    # User already answered our clarifying questions — skip scoping/clarification and
    # go straight to the full search. This eliminates 2 Tavily calls + 1 LLM call.
    emit_progress(
        callback,
        type="status",
        step="search",
        label="Searching with your context",
        detail="Going straight to full research now that we have your input.",
    )
    clarification = {"decision_dimensions": [], "search_focus": [], "preliminary_take": ""}
    scoping_data = {}

    search_plan = build_search_plan(understanding, clarification=clarification)
    search_data = search_multi(search_plan, callback=callback)
    raw_result = generate_final_result(
        input_text=original_input,
        product_hint=product_hint,
        understanding=understanding,
        clarification=clarification,
        scoping_data=scoping_data,
        search_data=search_data,
        user_context=user_context,
        followup_qa=followup_qa,
        pil_image=pil_image,
        callback=callback,
    )
    return normalize_result(
        result=raw_result,
        product_hint=product_hint,
        understanding=understanding,
        search_data=search_data,
        user_profile=user_profile or {},
        followup_count=len(followup_qa),
        memory_meta=memory_meta,
        followup_qa=followup_qa,
    )
