import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import html
import io
import json
import os
import re
from typing import Callable
from urllib.parse import parse_qs, urljoin, urlparse, quote, quote_plus
from urllib.request import Request, urlopen

from dotenv import load_dotenv
import google.generativeai as genai
from google import genai as google_genai
from google.genai import types as google_genai_types
from PIL import Image
load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
google_search_client = google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])

ProgressCallback = Callable[[dict], None]

UNDERSTAND_PROMPT = """You are the routing brain for a consumer decision agent.

Return ONLY valid JSON:
{
  "intent": "evaluate" | "recommend" | "compare",
  "user_goal": "<what the user is actually trying to solve, in their own terms>",
  "search_target": "<best search phrase to use once clarified>",
  "english_search_target": "<compact English search phrase for web search; translate Chinese need into natural English, keep brand/model names intact>",
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
- Default all user-facing free-text fields to natural Chinese.
- Keep brand names, product model names, and website names in their original language when needed.
- english_search_target must always be in English, even if the user writes in Chinese.
- Keep english_search_target short, search-friendly, and concrete.
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
  "dimension_priority": ["<from this exact set: 品质 | 性价比 | 适配度 | 口碑 | 品牌 | 安全>", "..."],
  "decision_dimensions": [
    {
      "title": "<must be one of: 品质 | 性价比 | 适配度 | 口碑 | 品牌 | 安全>",
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

Rules for dimension_priority / decision_dimensions:
- Use ONLY these six dimensions: 品质、性价比、适配度、口碑、品牌、安全.
- Return dimension_priority ordered from highest to lowest importance for THIS product and THIS user context.
- The order must change by category and evidence. Example logic only: ingestibles / topical / baby / pet / food often make 安全 more important;
  scenario-heavy products often make 适配度 more important. Do not hardcode one fixed order.
- decision_dimensions should follow the same order as dimension_priority.
- At minimum return the top 4 dimensions; ideally return all 6.

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
- All user-facing free-text fields MUST be Chinese:
  preliminary_take, decision_dimensions.detail, followup_reason, followup question text, options,
  open_text_placeholder, and search_focus. Do not output English sentences.
- Keep brand names, product model names, and website names in their original language when needed.
"""

FINAL_PROMPT = """You are "Buy or Not" — a sharp, warm consumer decision assistant for people aged 18–30.
You’ve done the research. Now give them a real, direct answer they can actually act on.

Tone:
- Be direct. Give an actual opinion, not a hedge.
- Sound like a knowledgeable friend who genuinely cares — not a product review robot.
- Light personality is welcome: a touch of wit, a dash of warmth. No cringe slang.
- When advising against buying, be candid but not preachy — explain what would serve them better.
- Assume the user is shopping in the United States unless they explicitly say otherwise.
- Prefer North American sources, retailers, prices, and community feedback. Avoid China-market guidance unless the product is clearly specific to that market.
- Community evidence matters: Reddit, creator reviews, and real-user complaint patterns are often more valuable than polished brand copy.
- Chinese contexts: use natural 普通话. Target: how a sharp 25-year-old would explain this
  to a friend over coffee. Avoid 综上所述, 不得不说, and other filler phrases.
- No markdown in string fields. No em-dashes as filler.
- Every user-facing free-text field MUST be Chinese. The only allowed English
  should be product model names, brand names, or site names. Never output English sentences in
  headline, summary, decision_dimensions.detail, reasons, fit_summary, caution_check, buy_if,
  skip_if, alternative.reason, recommendation.reason, recommendation.best_for, tradeoffs,
  followup.question, or followup options.
- If the language is unclear (for example image-only input), default to Chinese.

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
  "summary_source_ids": ["<source id from SOURCE CATALOG>", "<source id from SOURCE CATALOG>"],
  "dimension_priority": ["<from this exact set: 品质 | 性价比 | 适配度 | 口碑 | 品牌 | 安全>", "..."],
  "decision_dimensions": [
    {
      "title": "<must be one of: 品质 | 性价比 | 适配度 | 口碑 | 品牌 | 安全>",
      "status": "strong" | "mixed" | "weak" | "risk",
      "detail": "<how this dimension shaped the verdict — be concrete>"
    }
  ],
  "reasons": [
    "<reason with a specific fact, number, named source, or concrete comparison>",
    "<reason with a specific fact, number, named source, or concrete comparison>",
    "<optional third reason — only include if materially different from above>"
  ],
  "reason_source_ids": [
    ["<source id(s) supporting reasons[0]>"],
    ["<source id(s) supporting reasons[1]>"],
    ["<source id(s) supporting reasons[2]>"]
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
  "report_sections": [
    {
      "title": "<section title>",
      "body": "<section body>",
      "source_ids": ["<source id from SOURCE CATALOG>", "<source id from SOURCE CATALOG>"]
    }
  ],
  "display_modules": [
    {
      "type": "summary_card" | "text_block" | "decision_dimensions" | "recommendation_carousel" | "comparison_cards" | "source_gallery",
      "title": "<module title>",
      "body": "<module body>",
      "source_ids": ["<source id from SOURCE CATALOG>"],
      "items": [
        {
          "title": "<item title>",
          "body": "<item body>",
          "footer": "<short footer or tag>",
          "image_url": "<optional image url>",
          "source_ids": ["<source id from SOURCE CATALOG>"]
        }
      ]
    }
  ],
  "image_search": {
    "needed": true | false,
    "query": "<single compact image search phrase>",
    "reason": "<why extra reference images would help this answer land better>"
  },
  "scores": {
    "quality": <1-5 integer: build quality / reliability / ingredient quality>,
    "cost_value": <1-5 integer: whether the price is fair for what it gives>,
    "fit": <1-5 integer: how well it fits THIS user's actual needs and context>,
    "reviews": <1-5 integer: real-world user reputation and complaint pattern>,
    "brand": <1-5 integer: brand trust, consistency, and after-sales confidence>,
    "safety": <1-5 integer: safety, risk control, ingredient / material / suitability safety>
  }
}

Rules for scores:
- Fill all 6 score dimensions for every result (evaluate, compare, recommend).
- For recommend/compare: base scores on the first/primary recommendation or the winning product.
- Scores must reflect real evidence: a product with thin reviews should score 2-3 on "reviews",
  not a confident 4. Don't default to middle scores — spread them based on actual evidence.
- "fit" score is personal: the same product may score 5 for one user and 2 for another.
- "safety" is not only regulatory danger. It also includes ingredient/material suitability, known risk flags,
  age/stage fit, contraindications, and whether this user should be cautious before buying.

Rules for dimension_priority / decision_dimensions:
- Use ONLY these six dimensions: 品质、性价比、适配度、口碑、品牌、安全.
- Return dimension_priority ordered from highest to lowest importance for THIS product and THIS user.
- decision_dimensions must follow the same order and explain how each dimension affected the call.
- The result page should feel like a consumer assistant explaining how the decision was made:
  what mattered most, what mattered less, and where the user's own context changed the answer.

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
- summary_source_ids, reason_source_ids, report_sections[].source_ids, and display_modules[].source_ids must only use ids from SOURCE CATALOG.
- For display_modules: choose only the modules that actually help this answer land clearly. Do not force every module.
- If you have multiple product options or alternatives, prefer a recommendation_carousel or comparison_cards module.
- If there are useful source images, include a source_gallery module so the result feels richer and easier to trust.
- image_search: set needed=true only when one extra batch of reference images would materially improve trust, visual understanding, or side-by-side comparison.
- image_search.query must be a SINGLE compact search phrase. Never request multiple searches, multiple variants, or operator-heavy queries.
- If you include recommendation_carousel or comparison_cards, image_search.query should try to surface the main product and the strongest alternatives in one search so those cards can also show images.
- If image_search.needed is false, set image_search.query and image_search.reason to empty strings.
- Every key conclusion should cite at least one source id when usable evidence exists.
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
- Every answer should still land on a clear call. Even if tradeoffs exist, tell the user what you would do in their shoes.
- If the user’s situation clearly fits the product, say "买吧" in your own words.
  If it clearly doesn’t, say why plainly without padding.
- When explaining the verdict, always include at least one "because of you" link:
  for example "考虑到你更在意耐用性..." / "你家这只是幼年阶段..." / "你平时买这类东西更看重性价比...".
"""

SEARCH_PLANS = {
    "evaluate": [
        ("fit_signal", "Fit & key constraints", "{target} worth buying who should buy common complaints forum"),
        ("price_specs", "Price & specs", "{target} price specs amazon bestbuy walmart official"),
        ("reviews", "Reviews & complaints", "{target} review pros cons complaints wirecutter cnet the verge youtube"),
    ],
    "recommend": [
        ("needs_landscape", "Best options", "{target} best options buying guide wirecutter cnet the verge"),
        ("fit_signal", "Scene fit", "{target} best for commuting office daily running sensitive skin tradeoffs"),
        ("marketplaces", "Market pricing", "{target} amazon bestbuy walmart target rei sephora price"),
        ("reviews", "User sentiment", "{target} review complaints best value youtube rtings"),
    ],
    "compare": [
        ("comparisons", "Head-to-head comparisons", "{target} comparison review vs rtings cnet"),
        ("fit_signal", "Fit by scenario", "{target} which is better for commuting office travel running"),
        ("reviews", "User sentiment", "{target} review pros cons complaints alternatives cnet the verge"),
    ],
}

SEARCH_PLANS_ZH = {
    "evaluate": [
        ("fit_signal", "适用性分析", "{target} 适合什么人 使用场景 问题反馈 Reddit TikTok Instagram Pinterest 北美用户讨论"),
        ("price_specs", "价格与规格", "{target} 官方规格 价格 参数"),
        ("reviews", "用户评价与问题", "{target} 评测 优缺点 用户反馈 值得买 替代品"),
    ],
    "recommend": [
        ("needs_landscape", "最佳选择", "{target} 最佳选择 购买指南 Reddit TikTok Instagram Pinterest 北美用户讨论"),
        ("fit_signal", "场景匹配", "{target} 哪款更适合 不同场景 对比"),
        ("marketplaces", "市场价格", "{target} 价格 淘宝 京东 天猫"),
        ("reviews", "用户口碑", "{target} 用户评价 投诉 性价比 对比"),
    ],
    "compare": [
        ("comparisons", "对比评测", "{target} 对比 评测 哪个好 Reddit TikTok Instagram Pinterest 北美用户讨论"),
        ("fit_signal", "场景适配", "{target} 哪款更适合不同使用场景"),
        ("reviews", "用户反馈", "{target} 评测 优缺点 投诉 替代品"),
    ],
}

SCOPING_PLANS = {
    "evaluate": [
        ("product_fit", "Product fit snapshot", "{target} best for who common issues who should buy forum"),
        ("suitability", "Suitability check", "{target} suitability contraindications use case complaints"),
    ],
    "recommend": [
        ("decision_lens", "Decision factors", "{target} buying guide what to consider wirecutter cnet"),
        ("fit_signal", "User-fit clues", "{target} best for who common tradeoffs"),
    ],
    "compare": [
        ("decision_lens", "Decision factors", "{target} comparison what matters most cnet rtings"),
        ("fit_signal", "User-fit clues", "{target} which is better for different use cases"),
    ],
}

SCOPING_PLANS_ZH = {
    "evaluate": [
        ("product_fit", "产品适用性", "{target} 适合什么人 常见问题 值得买吗 Reddit TikTok Instagram Pinterest 北美用户讨论"),
        ("suitability", "适用性检查", "{target} 使用禁忌 适用场景 用户投诉"),
    ],
    "recommend": [
        ("decision_lens", "决策维度", "{target} 购买指南 选购要点 Reddit TikTok Instagram Pinterest 北美用户讨论"),
        ("fit_signal", "用户适配", "{target} 最适合什么人 常见权衡"),
    ],
    "compare": [
        ("decision_lens", "对比维度", "{target} 对比 最重要的考量 Reddit TikTok Instagram Pinterest 北美用户讨论"),
        ("fit_signal", "场景适配", "{target} 哪款更适合不同使用场景"),
    ],
}

CANONICAL_DIMENSIONS = [
    ("quality", "品质", ["品质", "做工", "质量", "可靠性"]),
    ("cost_value", "性价比", ["性价比", "价格", "值不值", "价格表现"]),
    ("fit", "适配度", ["适配度", "适合度", "适配", "场景适配"]),
    ("reviews", "口碑", ["口碑", "评价", "反馈", "投诉"]),
    ("brand", "品牌", ["品牌", "品牌力", "品牌信任", "售后"]),
    ("safety", "安全", ["安全", "安全性", "风险", "禁忌", "成分风险"]),
]

DIMENSION_LABEL_TO_KEY = {}
DIMENSION_ALIAS_TO_LABEL = {}
for key, label, aliases in CANONICAL_DIMENSIONS:
    DIMENSION_LABEL_TO_KEY[label] = key
    for alias in aliases + [label, key]:
        DIMENSION_ALIAS_TO_LABEL[alias.lower()] = label


def normalize_dimension_label(value: str) -> str:
    cleaned = clean_text(value or "", 40).replace(" ", "")
    if not cleaned:
        return ""
    return DIMENSION_ALIAS_TO_LABEL.get(cleaned.lower(), cleaned)


def dimension_terms(priority: list[str] | None, zh: bool = True) -> str:
    labels = [normalize_dimension_label(item) for item in (priority or []) if normalize_dimension_label(item)]
    if not labels:
        labels = ["适配度", "品质", "性价比"] if zh else ["fit", "quality", "value"]
    top = labels[:3]
    if zh:
        mapping = {
            "品质": "品质 做工 质量",
            "性价比": "价格 性价比 值不值",
            "适配度": "适合什么人 使用场景 适配",
            "口碑": "用户评价 口碑 投诉",
            "品牌": "品牌 售后 信任",
            "安全": "安全 风险 禁忌 成分",
        }
    else:
        mapping = {
            "品质": "quality reliability build",
            "性价比": "value price worth",
            "适配度": "fit suitability use case",
            "口碑": "reviews complaints sentiment",
            "品牌": "brand trust warranty",
            "安全": "safety risk ingredient contraindication",
        }
    return " ".join([mapping.get(item, item) for item in top])


def fallback_dimension_priority(understanding: dict, product_hint: str, search_data: dict | None = None) -> list[str]:
    combined = " ".join(
        [
            product_hint or "",
            understanding.get("search_target", ""),
            understanding.get("user_goal", ""),
            build_search_blob(search_data or {}),
        ]
    ).lower()
    base = {"品质": 0, "性价比": 0, "适配度": 0, "口碑": 0, "品牌": 0, "安全": 0}

    base["适配度"] += 3
    base["品质"] += 2
    base["性价比"] += 2
    base["口碑"] += 1
    base["品牌"] += 1

    if understanding.get("category_hint") in {"food", "beauty"}:
        base["安全"] += 4
    if any(token in combined for token in ["宠物", "狗", "猫", "supplement", "fish oil", "维生素", "鱼油", "奶粉", "食品", "成分", "敏感肌", "防晒"]):
        base["安全"] += 4
    if any(token in combined for token in ["通勤", "场景", "workflow", "户外", "跑", "使用", "适合", "电脑", "包", "鞋"]):
        base["适配度"] += 2
    if any(token in combined for token in ["奢侈", "礼物", "送礼", "联名", "samsonite", "新秀丽", "apple", "dyson", "品牌"]):
        base["品牌"] += 1
    if any(token in combined for token in ["投诉", "差评", "翻车", "review", "评价"]):
        base["口碑"] += 1

    return [item[0] for item in sorted(base.items(), key=lambda kv: (-kv[1], kv[0]))]


def normalize_dimension_priority(priority: list[str] | None, understanding: dict, product_hint: str, search_data: dict | None = None) -> list[str]:
    normalized = []
    for item in priority or []:
        label = normalize_dimension_label(item)
        if label in DIMENSION_LABEL_TO_KEY and label not in normalized:
            normalized.append(label)
    fallback = fallback_dimension_priority(understanding, product_hint, search_data)
    merged = normalized + [label for label in fallback if label not in normalized]
    return merged[:6]


def normalize_dimension_items(items: list[dict] | None, priority: list[str]) -> list[dict]:
    by_label = {}
    for item in items or []:
        label = normalize_dimension_label(item.get("title", ""))
        if label not in DIMENSION_LABEL_TO_KEY:
            continue
        by_label[label] = {
            "title": label,
            "status": item.get("status", "mixed"),
            "detail": clean_text(item.get("detail", ""), 140),
        }
    for label in priority:
        by_label.setdefault(
            label,
            {
                "title": label,
                "status": "mixed",
                "detail": "",
            },
        )
    ordered = [by_label[label] for label in priority if label in by_label]
    return ordered[:6]


def clarification_dimension_priority(result: dict) -> list[str]:
    items = result.get("decision_dimensions") or []
    return [normalize_dimension_label(item.get("title", "")) for item in items if normalize_dimension_label(item.get("title", ""))]

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
        model_name="gemini-3-flash-preview",
        system_instruction=system_instruction,
        generation_config={"response_mime_type": "application/json"},
    )


def _fast_json_model(system_instruction: str):
    """Fast model for latency-sensitive structured outputs (routing, clarification, extraction).
    Uses gemini-3-flash-preview which has no thinking overhead."""
    return genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        system_instruction=system_instruction,
        generation_config={"response_mime_type": "application/json"},
    )


GOOGLE_SEARCH_SYNTHESIS_PROMPT = """You are the web-search layer for a US consumer decision agent.

Your job:
- First translate the user's Chinese question, deeper need, product context, memory context, and follow-up context into precise English search intent.
- Search broadly across high-quality English sources for a US audience. Reddit is useful, but also actively look for Quora, specialist forums, trusted review blogs, editorial buying guides, and category-specific expert communities.
- Prefer American or North American consumer context, pricing, usage norms, and support expectations.
- Read English evidence first. Then internalize it and answer in high-quality Chinese.
- Avoid low-signal or irrelevant pages. Strongly prefer pages that clearly match the product, category, use case, or buying question.

Return ONLY valid JSON:
{
  "answer": "<2-4 sentence Chinese synthesis of what the web evidence says right now>",
  "query_focus": ["<precise english search query 1>", "<precise english search query 2>", "<precise english search query 3>"],
  "key_points": ["<Chinese point 1>", "<Chinese point 2>", "<Chinese point 3>"]
}
"""


def emit_progress(callback: ProgressCallback | None, **event):
    if callback:
        if event.get("type") == "status":
            event.setdefault("stage", event.get("step", "search"))
            event.setdefault("message", event.get("detail") or event.get("label") or "")
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
    if not text.strip():
        return True
    if re.search(r"[\u4e00-\u9fff]", text):
        return True
    english_words = re.findall(r"[A-Za-z]{2,}", text)
    return len(english_words) < 8


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
                "dimension_priority": ["安全", "适配度", "品质", "口碑", "品牌", "性价比"],
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
                "dimension_priority": ["安全", "适配度", "口碑", "品质", "品牌", "性价比"],
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
                "dimension_priority": ["适配度", "安全", "品质", "口碑", "品牌", "性价比"],
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
            "dimension_priority": ["安全", "适配度", "品质", "口碑", "品牌", "性价比"],
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
            "dimension_priority": ["安全", "适配度", "品质", "口碑", "品牌", "性价比"],
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
            "dimension_priority": ["适配度", "品质", "性价比", "口碑", "品牌", "安全"],
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
            "dimension_priority": ["适配度", "品质", "口碑", "性价比", "品牌", "安全"],
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
            "dimension_priority": ["性价比", "适配度", "品质", "口碑", "品牌", "安全"],
        }

    return None




def identify_product_from_image(pil_image: Image.Image) -> str:
    response = genai.GenerativeModel(model_name="gemini-3-flash-preview").generate_content(
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


DECISION_PHRASE_PATTERNS = [
    r"值不值得买",
    r"值不值得入",
    r"要不要买",
    r"能买吗",
    r"怎么样",
    r"好不好",
    r"适不适合我",
    r"推荐吗",
    r"值得入手吗",
    r"worth buying",
    r"should i buy",
    r"is it worth it",
]


def normalize_preview_query(text: str) -> str:
    value = clean_text(text or "", 140)
    for pattern in DECISION_PHRASE_PATTERNS:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    value = re.sub(r"[？?。!！]+$", "", value).strip(" ,，。！？?~")
    return clean_text(value, 100)


def should_attempt_preview_lookup(text: str) -> bool:
    cleaned = normalize_preview_query(text)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    broad_need_tokens = ["推荐", "帮我找", "怎么选", "哪个好", "对比", "vs", "recommend", "compare", "best "]
    if any(token in lowered for token in broad_need_tokens):
        return False
    return len(cleaned) >= 3


def fetch_url_text(url: str, timeout: int = 12) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return ""
        data = resp.read(500_000)
        encoding = (resp.headers.get("Content-Encoding") or "").lower()
    if encoding == "gzip" or data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except Exception:
            pass
    return data.decode("utf-8", errors="ignore")


def fetch_binary(url: str, timeout: int = 12) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read(3_000_000)


def fetch_json_url(url: str, timeout: int = 8) -> dict:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read(500_000)
    return json.loads(data.decode("utf-8", errors="ignore"))


def decode_bing_result_url(raw_url: str) -> str:
    candidate = html.unescape(raw_url or "").strip()
    parsed = urlparse(candidate)
    if "bing.com" not in parsed.netloc or not parsed.path.startswith("/ck/a"):
        return candidate

    encoded = parse_qs(parsed.query).get("u", [None])[0]
    if not encoded:
        return candidate

    if encoded.startswith("a1"):
        encoded = encoded[2:]
    padding = "=" * (-len(encoded) % 4)
    try:
        decoded = base64.urlsafe_b64decode(encoded + padding).decode("utf-8", errors="ignore")
        if decoded.startswith("http"):
            return decoded
    except Exception:
        return candidate
    return candidate


def search_product_page_candidates_via_tavily(query: str) -> list[dict]:
    try:
        response = tavily.search(query=query, search_depth="basic", max_results=4, include_answer=False)
    except Exception:
        return []

    candidates = []
    seen = set()
    for item in response.get("results", [])[:6]:
        page_url = (item.get("url") or "").strip()
        if not page_url.startswith("http") or page_url in seen:
            continue
        seen.add(page_url)
        candidates.append(
            {
                "url": page_url,
                "title": clean_text(item.get("title", ""), 140),
                "snippet": clean_text(item.get("content", "") or item.get("snippet", ""), 180),
            }
        )
    return candidates


def search_product_page_candidates_via_bing(query: str) -> list[dict]:
    search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
    req = Request(
        search_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
        },
    )
    with urlopen(req, timeout=15) as resp:
        html_text = resp.read(600_000).decode("utf-8", errors="ignore")

    matches = re.findall(
        r"<h2[^>]*>\s*<a href=\"([^\"]+)\"[^>]*>(.*?)</a>.*?</h2>(.*?)(?:</li>|<li class=\"b_algo\")",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates = []
    for raw_url, raw_title, trailing in matches[:8]:
        page_url = decode_bing_result_url(raw_url)
        title = clean_text(re.sub(r"<.*?>", " ", html.unescape(raw_title)), 140)
        snippet_match = re.search(r"<p>(.*?)</p>", trailing, flags=re.IGNORECASE | re.DOTALL)
        snippet = clean_text(re.sub(r"<.*?>", " ", html.unescape(snippet_match.group(1))), 180) if snippet_match else ""
        if page_url.startswith("http"):
            candidates.append({"url": page_url, "title": title, "snippet": snippet})
    return candidates


def search_product_page_candidates(query: str) -> list[dict]:
    return search_product_page_candidates_via_bing(query)


SOCIAL_SITE_DOMAINS = ["reddit.com"]
US_PRIORITY_DOMAINS = [
    "reddit.com",
    "quora.com",
    "wirecutter.com",
    "nytimes.com",
    "theverge.com",
    "wired.com",
    "cnet.com",
    "rtings.com",
    "pcmag.com",
    "techradar.com",
    "tomsguide.com",
    "consumerreports.org",
    "caranddriver.com",
    "edmunds.com",
    "motortrend.com",
    "youtube.com",
    "amazon.com",
    "bestbuy.com",
    "walmart.com",
    "target.com",
    "rei.com",
    "sephora.com",
]
LOW_PRIORITY_DOMAINS = [
    "vertexaisearch.cloud.google.com",
    "colab.research.google.com",
    "google.com",
    "news.ycombinator.com",
    "tmall.com",
    "taobao.com",
    "jd.com",
    "xiaohongshu.com",
    "douyin.com",
    "bilibili.com",
    ".cn/",
]


def wants_social_posts(query: str) -> bool:
    lowered = (query or "").lower()
    return any(
        token in lowered
        for token in [
            "reddit",
            "community",
            "forum",
            "user discussion",
            "北美用户讨论",
            "值不值得",
            "值不值",
            "买吗",
            "推荐吗",
            "怎么样",
            "真实体验",
            "用户评价",
            "wirecutter",
            "cnet",
            "the verge",
            "rtings",
        ]
    )


def build_social_site_query(query: str) -> str:
    cleaned = re.sub(r"\b(reddit|tiktok|instagram|pinterest)\b", " ", query, flags=re.IGNORECASE)
    cleaned = cleaned.replace("北美用户讨论", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if "site:" in cleaned:
        return cleaned
    return (
        f"{cleaned} "
        "(site:reddit.com OR site:quora.com OR site:wirecutter.com OR site:cnet.com OR site:rtings.com "
        "OR site:edmunds.com OR site:caranddriver.com OR site:motortrend.com OR site:consumerreports.org "
        "OR site:theverge.com OR site:wired.com OR site:pcmag.com OR site:techradar.com)"
    ).strip()


def build_reddit_site_query(query: str) -> str:
    cleaned = re.sub(r"\b(reddit|tiktok|instagram|pinterest)\b", " ", query, flags=re.IGNORECASE)
    cleaned = cleaned.replace("北美用户讨论", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return f"{cleaned} site:reddit.com".strip()


def sanitize_tavily_query(query: str) -> str:
    cleaned = re.sub(r"\(?site:[^)]+\)?", " ", query, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(site:reddit\.com|site:tiktok\.com|site:instagram\.com|site:pinterest\.com)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(reddit|tiktok|instagram|pinterest)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or query


def search_social_posts_via_bing(query: str, max_results: int = 4) -> list[dict]:
    candidates = search_product_page_candidates_via_bing(build_reddit_site_query(query))
    seen = set()
    results = []
    for item in candidates:
        page_url = (item.get("url") or "").strip()
        if not page_url or page_url in seen:
            continue
        seen.add(page_url)
        results.append(
            {
                "title": clean_text(item.get("title", ""), 90),
                "url": page_url,
                "site": extract_site(page_url),
                "snippet": clean_text(item.get("snippet", ""), 180),
                "image_url": "",
            }
        )
        if len(results) >= max_results:
            break
    return results


def query_keywords_for_match(query: str) -> list[str]:
    lowered = sanitize_tavily_query(query).lower()
    tokens = re.findall(r"[a-z0-9]{3,}", lowered)
    stop = {
        "best", "with", "from", "that", "this", "what", "when", "where", "which", "review",
        "reviews", "official", "price", "worth", "buying", "guide", "issues", "common",
        "fit", "constraints", "options", "comparison", "better", "who", "for", "use", "case",
        "reddit", "site", "com", "should", "would", "help", "need", "user", "discussion",
    }
    deduped = []
    for token in tokens:
        if token in stop:
            continue
        if token not in deduped:
            deduped.append(token)
    return deduped[:8]


def query_token_overlap(query: str, item: dict) -> int:
    haystack = " ".join([item.get("title", ""), item.get("snippet", ""), item.get("url", "")]).lower()
    return sum(1 for token in query_keywords_for_match(query) if token in haystack)


def is_query_relevant(query: str, item: dict, social_mode: bool = False) -> bool:
    overlap = query_token_overlap(query, item)
    url = (item.get("url") or "").lower()
    site = (item.get("site") or "").lower()
    if social_mode:
        allowed_domains = {
            "reddit.com",
            "quora.com",
            "wirecutter.com",
            "nytimes.com",
            "theverge.com",
            "wired.com",
            "cnet.com",
            "rtings.com",
            "pcmag.com",
            "techradar.com",
            "consumerreports.org",
            "edmunds.com",
            "caranddriver.com",
            "motortrend.com",
            "youtube.com",
        }
        has_allowed_domain = any(domain in url or site == domain for domain in allowed_domains)
        if overlap >= 2 and has_allowed_domain:
            return True
        return overlap >= 1 and has_allowed_domain
    if overlap >= 2:
        return True
    if overlap >= 1 and any(token in url for token in ["review", "product", "official", "spec", "buy", "compare"]):
        return True
    return False


def source_relevance_score(query: str, item: dict) -> float:
    haystack = " ".join([item.get("title", ""), item.get("snippet", ""), item.get("url", "")]).lower()
    score = 0.0
    for token in query_keywords_for_match(query):
        if token in haystack:
            score += 1.5
    if query_token_overlap(query, item) == 0:
        score -= 3.0
    url = (item.get("url") or "").lower()
    site = (item.get("site") or "").lower()
    if "reddit.com" in url or site == "reddit.com":
        score += 4.0
    elif "youtube.com" in url:
        score += 0.5
    for domain in US_PRIORITY_DOMAINS:
        if domain in url or site == domain:
            score += 0.6
            break
    for domain in LOW_PRIORITY_DOMAINS:
        if domain in url or site == domain:
            score -= 2.2
            break
    if wants_social_posts(query) and not any(domain in url or site == domain for domain in [
        "reddit.com",
        "quora.com",
        "wirecutter.com",
        "nytimes.com",
        "theverge.com",
        "wired.com",
        "cnet.com",
        "rtings.com",
        "pcmag.com",
        "techradar.com",
        "consumerreports.org",
        "edmunds.com",
        "caranddriver.com",
        "motortrend.com",
        "youtube.com",
    ]):
        score -= 1.1
    return score


def extract_meta_image_candidates(html: str, page_url: str) -> list[str]:
    patterns = [
        r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
        r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:image["\']',
        r'<meta[^>]*name=["\']twitter:image["\'][^>]*content=["\']([^"\']+)["\']',
        r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']twitter:image["\']',
    ]
    candidates = []
    for pattern in patterns:
        for match in re.findall(pattern, html, flags=re.IGNORECASE):
            absolute = urljoin(page_url, match.strip())
            if absolute and absolute not in candidates:
                candidates.append(absolute)

    if candidates:
        return candidates

    img_patterns = [
        r'<img[^>]+(?:src|data-src|data-original)=["\']([^"\']+)["\']',
        r'"(?:image|img|src|data-src)"\s*:\s*"(https?:\\/\\/[^"\\]+)"',
        r'"thumbnailUrl"\s*:\s*"([^"]+)"',
        r'"thumbnail"\s*:\s*\{\s*"url"\s*:\s*"([^"]+)"',
        r'"url"\s*:\s*"(https?:\\/\\/(?:i\.redd\.it|preview\.redd\.it|external-preview\.redd\.it)[^"\\]+)"',
    ]
    for pattern in img_patterns:
        for match in re.findall(pattern, html, flags=re.IGNORECASE):
            candidate = match.replace('\/', '/')
            absolute = urljoin(page_url, candidate.strip())
            if absolute and absolute not in candidates and looks_like_product_image(absolute):
                candidates.append(absolute)
    return candidates[:8]


def looks_like_product_image(url: str) -> bool:
    lowered = url.lower()
    blocked_domains = [
        "findarticles.com",
    ]
    if any(domain in lowered for domain in blocked_domains):
        return False
    noisy_tokens = [
        "logo",
        "icon",
        "avatar",
        "sprite",
        "banner",
        "favicon",
        "blank",
        "placeholder",
        "spacer",
        "loading",
        "similarweb",
        "screenshot",
        "text-image",
        "permission",
        "access-denied",
        "forbidden",
        "serve-this-content",
        "blocked",
    ]
    if any(token in lowered for token in noisy_tokens):
        return False
    if lowered.endswith('.gif'):
        return False
    return lowered.startswith("http")


def resolve_social_preview_image(url: str) -> str:
    lowered = (url or "").lower()
    try:
        if "reddit.com" in lowered:
            data = fetch_json_url(f"https://www.reddit.com/oembed?url={quote(url, safe='')}", timeout=4)
            return data.get("thumbnail_url") or ""
        if "tiktok.com" in lowered:
            data = fetch_json_url(f"https://www.tiktok.com/oembed?url={quote(url, safe='')}", timeout=4)
            return data.get("thumbnail_url") or ""
        if "pinterest.com" in lowered:
            data = fetch_json_url(f"https://www.pinterest.com/oembed.json?url={quote(url, safe='')}", timeout=4)
            return (
                data.get("thumbnail_url")
                or ((data.get("thumbnail_url_with_size") or {}).get("url") if isinstance(data.get("thumbnail_url_with_size"), dict) else "")
                or ""
            )
    except Exception:
        return ""
    return ""


PREFERRED_PREVIEW_DOMAINS = [
    "samsonite.com.cn",
    "shop.samsonite.com",
    "samsonite.com",
    "suning.com",
    "jd.com",
    "tmall.com",
    "taobao.com",
    "amazon.com",
]


def preview_query_keywords(query: str) -> list[str]:
    raw = re.split(r"[\s/,_+\-]+", query.lower())
    keywords = [token for token in raw if len(token) >= 2]
    if prefers_chinese(query):
        keywords.extend([query[i : i + 2] for i in range(max(0, len(query) - 1))])
    deduped = []
    for token in keywords:
        if token and token not in deduped:
            deduped.append(token)
    return deduped[:8]


def score_preview_candidate(query: str, item: dict) -> float:
    haystack = " ".join(
        [
            item.get("title", ""),
            item.get("snippet", ""),
            item.get("url", ""),
        ]
    ).lower()
    score = 0.0
    for domain in PREFERRED_PREVIEW_DOMAINS:
        if domain in haystack:
            score += 3.0
            break

    for token in preview_query_keywords(query):
        if token in haystack:
            score += 1.0

    lowered_url = (item.get("url", "") or "").lower()
    if any(token in lowered_url for token in ["product", "item", "p/", "dp/", "sku", "detail"]):
        score += 1.0
    return score


def compress_image_for_transport(image_bytes: bytes) -> str | None:
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_image.thumbnail((900, 900))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=84, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return None


PREVIEW_IMAGE_VERIFY_PROMPT = """You are checking whether a candidate product image matches a user's text query.

Return ONLY valid JSON:
{
  "is_match": true | false,
  "confidence": 0.0,
  "identified_product": "<what product this image seems to show>",
  "reason": "<short reason>"
}

Rules:
- Be strict. If the image is scenery, a building, a logo, or a generic lifestyle shot, return is_match=false.
- Only return is_match=true if the image is plausibly the same product or a very close visual match to the named product.
- If the query is broad or the image is ambiguous, prefer false.
- Match can still be true if it is the same model in a different colorway.
"""


REFERENCE_IMAGE_VERIFY_PROMPT = """You are checking whether an image is a usable visual reference for a user's buying question.

Return ONLY valid JSON:
{
  "is_usable": true | false,
  "confidence": 0.0,
  "reason": "<short reason>"
}

Rules:
- Reject screenshots of articles, webpages, social posts, and text-heavy title cards.
- Reject placeholder images, access-denied images, permission-warning images, and broken-image fallbacks.
- Reject logos, icons, charts, pure text graphics, memes, or generic promotional banners.
- Reject images that are clearly unrelated to the product/object the query is about.
- Accept only if the image gives a direct and useful visual reference for the query or named product.
- Be strict. If unsure, return is_usable=false.
"""


def verify_preview_image_match(query: str, page_title: str, page_snippet: str, image_bytes: bytes) -> dict:
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"is_match": False, "confidence": 0.0, "identified_product": "", "reason": "invalid image"}

    prompt = f"""USER QUERY:
{query}

PAGE TITLE:
{page_title}

PAGE SNIPPET:
{page_snippet}
"""
    try:
        response = _fast_json_model(PREVIEW_IMAGE_VERIFY_PROMPT).generate_content([pil_image, prompt])
        payload = parse_json(response.text)
        return {
            "is_match": bool(payload.get("is_match")),
            "confidence": float(payload.get("confidence") or 0.0),
            "identified_product": clean_text(payload.get("identified_product", ""), 120),
            "reason": clean_text(payload.get("reason", ""), 160),
        }
    except Exception as exc:
        return {
            "is_match": False,
            "confidence": 0.0,
            "identified_product": "",
            "reason": clean_text(str(exc), 120),
        }


def verify_reference_image_match(query: str, page_title: str, page_snippet: str, image_bytes: bytes) -> dict:
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"is_usable": False, "confidence": 0.0, "reason": "invalid image"}

    prompt = f"""USER QUERY:
{query}

PAGE TITLE:
{page_title}

PAGE SNIPPET:
{page_snippet}
"""
    try:
        response = _fast_json_model(REFERENCE_IMAGE_VERIFY_PROMPT).generate_content([pil_image, prompt])
        payload = parse_json(response.text)
        return {
            "is_usable": bool(payload.get("is_usable")),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": clean_text(payload.get("reason", ""), 160),
        }
    except Exception as exc:
        return {
            "is_usable": False,
            "confidence": 0.0,
            "reason": clean_text(str(exc), 120),
        }


def image_bytes_look_usable(image_bytes: bytes) -> bool:
    if not image_bytes or len(image_bytes) < 1200:
        return False
    try:
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
    except Exception:
        return False
    if width < 80 or height < 80:
        return False
    aspect_ratio = max(width, height) / max(1, min(width, height))
    if aspect_ratio > 4.0:
        return False
    return True


def choose_verified_reference_image(query: str, item: dict) -> tuple[str, bytes] | tuple[None, None]:
    title = clean_text(item.get("title", "") or item.get("source", "") or query, 100)
    snippet = clean_text(item.get("source", "") or item.get("snippet", "") or "", 160)
    blocked_meta_tokens = [
        "permission",
        "access",
        "serve this content",
        "forbidden",
        "blocked",
        "denied",
        "captcha",
    ]
    lowered_meta = f"{title} {snippet}".lower()
    if any(token in lowered_meta for token in blocked_meta_tokens):
        return None, None
    candidate_urls = []
    for candidate in [
        item.get("original", ""),
        item.get("image", ""),
        item.get("thumbnail", ""),
        item.get("thumbnail_url", ""),
    ]:
        cleaned = clean_text(candidate, 500)
        if cleaned and cleaned not in candidate_urls:
            candidate_urls.append(cleaned)

    for candidate_url in candidate_urls:
        if not looks_like_product_image(candidate_url):
            continue
        try:
            image_bytes = fetch_binary(candidate_url, timeout=3)
        except Exception:
            continue
        if not image_bytes_look_usable(image_bytes):
            continue
        return candidate_url, image_bytes

    return None, None


ENGLISH_SEARCH_PROMPT = """Translate the user's buying need into one compact English web search phrase.

Rules:
- Return only the search phrase. No quotes, no markdown.
- Keep brand names and model names intact.
- Keep it short, concrete, and search-friendly.
- Prefer noun phrases over full sentences.
"""


def ensure_english_search_target(value: str, fallback_text: str) -> str:
    candidate = clean_text(value or "", 140)
    if candidate and not re.search(r"[\u4e00-\u9fff]", candidate):
        return candidate

    source = clean_text(fallback_text or candidate, 140)
    if not source:
        return ""

    try:
        response = genai.GenerativeModel(model_name="gemini-3-flash-preview").generate_content(
            f"{ENGLISH_SEARCH_PROMPT}\n\nINPUT:\n{source}"
        )
        translated = clean_text(response.text, 140)
        if translated and not re.search(r"[\u4e00-\u9fff]", translated):
            return translated
    except Exception:
        pass

    ascii_only = re.sub(r"[\u4e00-\u9fff]+", " ", source)
    ascii_only = re.sub(r"\s+", " ", ascii_only).strip()
    return ascii_only or source


def ensure_chinese_text(value: str, fallback_text: str = "") -> str:
    text = clean_text(value or fallback_text, 220)
    if not text or re.search(r"[\u4e00-\u9fff]", text):
        return text
    try:
        response = genai.GenerativeModel(model_name="gemini-3-flash-preview").generate_content(
            "Translate this UI text into natural Chinese. Return only the translated sentence.\n\nTEXT:\n" + text
        )
        translated = clean_text(response.text, 220)
        if translated:
            return translated
    except Exception:
        pass
    return text


def find_verified_preview_image(input_text: str) -> dict | None:
    if not should_attempt_preview_lookup(input_text):
        return None

    query = normalize_preview_query(input_text)
    search_query = f"{query} product image"
    if prefers_chinese(query):
        search_query = f"{query} 商品 图"

    candidates = search_product_page_candidates(search_query)
    if not candidates:
        return None

    ranked_candidates = sorted(
        candidates,
        key=lambda item: score_preview_candidate(query, item),
        reverse=True,
    )

    for item in ranked_candidates[:3]:
        page_url = item.get("url") or ""
        if not page_url:
            continue
        try:
            html = fetch_url_text(page_url, timeout=5)
        except Exception:
            continue
        if not html:
            continue

        image_candidates = [url for url in extract_meta_image_candidates(html, page_url) if looks_like_product_image(url)]
        if not image_candidates:
            continue

        for image_url in image_candidates[:1]:
            try:
                image_bytes = fetch_binary(image_url, timeout=5)
            except Exception:
                continue
            image_base64 = compress_image_for_transport(image_bytes)
            if not image_base64:
                continue
            return {
                "matched": True,
                "page_url": page_url,
                "image_url": image_url,
                "product_name": clean_text(item.get("title", "") or query, 120),
                "reason": "Single-search preview image from matched product page.",
                "confidence": 0.82,
                "image_base64": image_base64,
            }

    return None


def search_reference_images_once(query: str, limit: int = 6) -> list[dict]:
    cleaned_query = clean_text(query, 140)
    if not cleaned_query or not SERPAPI_API_KEY:
        return []

    endpoint = (
        "https://serpapi.com/search?"
        f"engine=google_images_light&q={quote_plus(cleaned_query)}"
        f"&api_key={quote_plus(SERPAPI_API_KEY)}&safe=active"
    )

    try:
        payload = fetch_json_url(endpoint, timeout=12)
    except Exception:
        return []

    raw_items = payload.get("images_results") or payload.get("image_results") or []
    cards: list[dict] = []
    seen: set[str] = set()
    candidates: list[dict] = []

    for item in raw_items:
        page_url = clean_text(
            item.get("link", "")
            or item.get("source_page", "")
            or item.get("source_url", ""),
            500,
        )
        image_url, image_bytes = choose_verified_reference_image(cleaned_query, item)
        if not image_url:
            continue

        dedupe_key = page_url or image_url
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        if not image_bytes_look_usable(image_bytes or b""):
            continue

        site = extract_site(page_url) or clean_text(item.get("source", ""), 60) or "Google Images"
        title = clean_text(item.get("title", "") or item.get("source", "") or cleaned_query, 100)
        body = clean_text(item.get("source", "") or item.get("snippet", "") or "作为这次判断的外观参考图。", 160)
        source = {
            "title": title or site or "图片参考",
            "url": page_url or image_url,
            "site": site,
            "bucket": "image_reference",
            "snippet": body,
            "image_url": image_url,
        }
        candidates.append(
            {
                "title": title or site or "图片参考",
                "body": body,
                "footer": site,
                "image_url": image_url,
                "sources": [source],
                "_image_bytes": image_bytes,
                "_score": source_relevance_score(cleaned_query, source),
            }
        )

    ranked_candidates = sorted(candidates, key=lambda item: item.get("_score", 0.0), reverse=True)
    ranked_candidates = ranked_candidates[: max(limit * 4, 12)]

    def verify_candidate(candidate: dict) -> tuple[bool, float]:
        image_bytes = candidate.get("_image_bytes") or b""
        verdict = verify_reference_image_match(
            cleaned_query,
            candidate.get("title", ""),
            candidate.get("body", ""),
            image_bytes,
        )
        return bool(verdict.get("is_usable")), float(verdict.get("confidence") or 0.0)

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(verify_candidate, candidate): candidate for candidate in ranked_candidates}
        for future in as_completed(future_map):
            candidate = future_map[future]
            try:
                is_usable, confidence = future.result()
            except Exception:
                continue
            if not is_usable or confidence < 0.55:
                continue
            candidate.pop("_image_bytes", None)
            candidate.pop("_score", None)
            cards.append(candidate)
            if len(cards) >= limit:
                break

    return cards


def merge_gallery_items(primary_items: list[dict], secondary_items: list[dict], limit: int = 6) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()

    for item in primary_items + secondary_items:
        key = item.get("image_url") or ((item.get("sources") or [{}])[0].get("url", "")) or item.get("title", "")
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break

    return merged


def _normalized_match_tokens(*parts: str) -> set[str]:
    text = " ".join(part for part in parts if part).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    stopwords = {
        "the", "and", "for", "with", "this", "that", "from", "real", "life",
        "comparison", "review", "vs", "video", "youtube", "guide", "best",
        "worth", "buying", "product", "photo", "photos", "image", "images",
        "cabin", "carry", "check", "in",
    }
    return {token for token in tokens if len(token) >= 3 and token not in stopwords}


def _best_reference_image_for_item(item: dict, image_cards: list[dict], used_keys: set[str]) -> dict | None:
    item_tokens = _normalized_match_tokens(item.get("title", ""), item.get("body", ""), item.get("footer", ""))
    if not image_cards:
        return None

    best_card = None
    best_score = 0.0
    for index, card in enumerate(image_cards):
        key = card.get("image_url") or card.get("title", "") or str(index)
        if key in used_keys:
            continue
        candidate_tokens = _normalized_match_tokens(
            card.get("title", ""),
            card.get("body", ""),
            ((card.get("sources") or [{}])[0]).get("title", ""),
            ((card.get("sources") or [{}])[0]).get("site", ""),
        )
        overlap = len(item_tokens & candidate_tokens)
        title_hit = 1.5 if item.get("title", "").strip() and item.get("title", "").lower() in card.get("title", "").lower() else 0
        footer_hit = 0.8 if item.get("footer", "").strip() and item.get("footer", "").lower() in card.get("title", "").lower() else 0
        score = overlap + title_hit + footer_hit
        if score > best_score:
            best_score = score
            best_card = card

    if best_card:
        return best_card

    for card in image_cards:
        key = card.get("image_url") or card.get("title", "")
        if key and key not in used_keys:
            return card
    return None


def enrich_visual_module_items_with_reference_images(modules: list[dict], image_cards: list[dict]) -> list[dict]:
    if not image_cards:
        return modules

    used_keys: set[str] = set()
    enriched_modules: list[dict] = []

    for module in modules:
        if module.get("type") not in {"recommendation_carousel", "comparison_cards"}:
            enriched_modules.append(module)
            continue

        updated_module = dict(module)
        updated_items: list[dict] = []
        for item in module.get("items") or []:
            updated_item = dict(item)
            if not updated_item.get("image_url"):
                best_card = _best_reference_image_for_item(updated_item, image_cards, used_keys)
                if best_card:
                    card_key = best_card.get("image_url") or best_card.get("title", "")
                    if card_key:
                        used_keys.add(card_key)
                    updated_item["image_url"] = best_card.get("image_url", "")
                    merged_sources = (updated_item.get("sources") or []) + (best_card.get("sources") or [])
                    deduped_sources = []
                    seen_urls: set[str] = set()
                    for source in merged_sources:
                        url = source.get("url", "")
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        deduped_sources.append(source)
                    updated_item["sources"] = deduped_sources[:2]
            updated_items.append(updated_item)
        updated_module["items"] = updated_items
        enriched_modules.append(updated_module)

    return enriched_modules


def attach_reference_image_gallery(result: dict) -> dict:
    image_search = result.get("image_search")
    if not isinstance(image_search, dict):
        return result

    if not image_search.get("needed"):
        return result

    query = clean_text(image_search.get("query", ""), 140)
    if not query:
        return result

    existing_gallery_items = 0
    for module in result.get("display_modules") or []:
        if module.get("type") == "source_gallery":
            existing_gallery_items += len(module.get("items") or [])
    if existing_gallery_items >= 4:
        return result

    image_cards = search_reference_images_once(query, limit=max(4, 6 - existing_gallery_items))
    if not image_cards:
        return result

    title = "我补了一组图片参考"
    body = clean_text(image_search.get("reason", ""), 180)
    modules = result.get("display_modules") or []
    updated_modules: list[dict] = []
    inserted = False

    for module in modules:
        if module.get("type") == "source_gallery" and not inserted:
            updated = dict(module)
            updated["title"] = module.get("title") or title
            updated["body"] = module.get("body") or body
            updated["items"] = merge_gallery_items(image_cards, module.get("items") or [], limit=6)
            updated_modules.append(updated)
            inserted = True
        else:
            updated_modules.append(module)

    if not inserted:
        updated_modules.append(
            {
                "type": "source_gallery",
                "title": title,
                "body": body,
                "sources": [],
                "items": image_cards[:6],
            }
        )

    updated_modules = enrich_visual_module_items_with_reference_images(updated_modules, image_cards)
    result["display_modules"] = updated_modules

    research = result.get("research")
    if isinstance(research, dict):
        existing_sources = research.get("sources") or []
        extra_sources = [item["sources"][0] for item in image_cards if item.get("sources")]
        merged_sources: list[dict] = []
        seen_source_urls: set[str] = set()
        for source in existing_sources + extra_sources:
            url = source.get("url", "")
            if not url or url in seen_source_urls:
                continue
            seen_source_urls.add(url)
            merged_sources.append(source)
        research["sources"] = merged_sources[:12]

    return result


def attach_images_from_search(result: dict, search_data: dict) -> dict:
    """Build source_gallery from images already present in search results.
    No extra HTTP calls — images come from Google grounded search og:image extraction."""
    seen_keys: set[str] = set()
    cards: list[dict] = []

    for bucket in search_data.values():
        for item in bucket.get("results", []):
            image_url = item.get("image_url", "")
            if not image_url:
                continue
            key = image_url
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cards.append(
                {
                    "title": item.get("title") or item.get("site") or "图片参考",
                    "body": item.get("snippet", ""),
                    "footer": item.get("site", ""),
                    "image_url": image_url,
                    "sources": [
                        {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "site": item.get("site", ""),
                            "snippet": item.get("snippet", ""),
                            "image_url": image_url,
                        }
                    ],
                }
            )

    if not cards:
        return result

    modules = result.get("display_modules") or []
    existing_gallery = next((m for m in modules if m.get("type") == "source_gallery"), None)
    if existing_gallery:
        existing_gallery["items"] = merge_gallery_items(cards, existing_gallery.get("items") or [], limit=6)
    else:
        modules.append(
            {
                "type": "source_gallery",
                "title": "我顺手翻到的图和网页",
                "body": "",
                "sources": [],
                "items": cards[:6],
            }
        )
    result["display_modules"] = modules
    return result


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
        "english_search_target": ensure_english_search_target(product_hint or combined, product_hint or combined),
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
        label="先猜猜你真正想解决什么",
        detail="我先把你的真实诉求拎出来，后面才不会查偏。",
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
    data.setdefault(
        "english_search_target",
        ensure_english_search_target(data.get("search_target", ""), product_hint or combined_text or data.get("user_goal", "")),
    )
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
    data["english_search_target"] = ensure_english_search_target(
        data.get("english_search_target", ""),
        product_hint or combined_text or data.get("search_target") or data.get("user_goal", ""),
    )

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
        label="大概懂你想要什么了",
        detail=clean_text(data["user_goal"], 120),
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
    target = understanding.get("english_search_target") or understanding.get("search_target") or understanding.get("user_goal") or ""
    if intent == "compare" and understanding.get("comparison_targets"):
        target = " vs ".join(understanding["comparison_targets"])
    plan = SCOPING_PLANS.get(intent, SCOPING_PLANS["evaluate"])
    return [(key, label, template.format(target=target)) for key, label, template in plan]


def build_search_plan(understanding: dict, clarification: dict | None = None) -> list[tuple[str, str, str]]:
    intent = understanding.get("intent", "evaluate")
    target = understanding.get("english_search_target") or understanding.get("search_target") or understanding.get("user_goal") or ""
    if intent == "compare" and understanding.get("comparison_targets"):
        target = " vs ".join(understanding["comparison_targets"])
    if clarification:
        focus = " ".join(
            [
                clean_text(item, 40)
                for item in clarification.get("search_focus", [])[:2]
                if item and not re.search(r"[\u4e00-\u9fff]", item)
            ]
        )
        if focus:
            target = f"{target} {focus}".strip()
    if clarification:
        priority = clarification.get("dimension_priority") or []
        target = f"{target} {dimension_terms(priority, zh=False)}".strip()

    plan = SEARCH_PLANS.get(intent, SEARCH_PLANS["evaluate"])
    return [(key, label, template.format(target=target)) for key, label, template in plan]


def run_one_search(query: str, search_depth: str, max_results: int, prefer_fast: bool = False) -> dict:
    """Run a single web search using Gemini's native Google Search grounding."""
    try:
        grounded = run_one_google_grounded_search(query, max_results)
        grounded_results = grounded.get("results", [])
        if grounded_results:
            ranked = sorted(
                grounded_results,
                key=lambda item: source_relevance_score(query, item),
                reverse=True,
            )
            thresholded = [
                item for item in ranked
                if source_relevance_score(query, item) >= 1.8 and is_query_relevant(query, item)
            ]
            grounded["results"] = diversify_sources_by_site(thresholded or ranked, max_results)
        return grounded
    except Exception as exc:
        return {
            "answer": "",
            "results": [],
            "queries": [],
            "key_points": [],
            "error": clean_text(str(exc), 160),
        }


def resolve_page_image(url: str) -> str:
    if not url:
        return ""
    social_image = resolve_social_preview_image(url)
    if social_image and looks_like_product_image(social_image):
        return social_image
    try:
        html = fetch_url_text(url, timeout=2)
    except Exception:
        return ""
    if not html:
        return ""
    candidates = extract_meta_image_candidates(html, url)
    for candidate in candidates:
        if looks_like_product_image(candidate):
            return candidate
    return ""


def hydrate_result_images(results: list[dict], limit: int = 1) -> list[dict]:
    hydrated = []
    for index, item in enumerate(results):
        enriched = dict(item)
        if not enriched.get("image_url") and index < limit:
            enriched["image_url"] = resolve_page_image(enriched.get("url", ""))
        hydrated.append(enriched)
    return hydrated


def hydrate_result_images_parallel(results: list[dict], limit: int = 3) -> list[dict]:
    """Fetch og:images for up to `limit` results in parallel (max 3s wall time)."""
    if not results:
        return results
    to_fetch = [(i, item) for i, item in enumerate(results) if not item.get("image_url")][:limit]
    if not to_fetch:
        return results

    updated = list(results)

    def fetch_one(args: tuple) -> tuple[int, str]:
        idx, item = args
        return idx, resolve_page_image(item.get("url", ""))

    with ThreadPoolExecutor(max_workers=min(3, len(to_fetch))) as executor:
        futures = {executor.submit(fetch_one, args): args[0] for args in to_fetch}
        try:
            for future in as_completed(futures, timeout=4):
                idx, image_url = future.result()
                if image_url:
                    updated[idx] = {**updated[idx], "image_url": image_url}
        except Exception:
            pass
    return updated


def canonicalize_grounded_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        for key in ("url", "q", "dest", "target", "redirect"):
            values = query.get(key) or []
            for value in values:
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    return value
    except Exception:
        return url
    return url


def extract_grounding_sources(response) -> tuple[list[dict], list[str]]:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return [], []
    grounding = getattr(candidates[0], "grounding_metadata", None)
    if not grounding:
        return [], []

    search_queries = list(getattr(grounding, "web_search_queries", None) or [])
    chunks = list(getattr(grounding, "grounding_chunks", None) or [])
    supports = list(getattr(grounding, "grounding_supports", None) or [])

    support_by_chunk: dict[int, list[str]] = {}
    for support in supports:
        segment = getattr(support, "segment", None)
        text = clean_text(getattr(segment, "text", "") if segment else "", 220)
        if not text:
            continue
        for idx in getattr(support, "grounding_chunk_indices", None) or []:
            support_by_chunk.setdefault(idx, []).append(text)

    sources = []
    seen = set()
    for index, chunk in enumerate(chunks):
        web = getattr(chunk, "web", None)
        if not web:
            continue
        raw_url = (getattr(web, "uri", "") or "").strip()
        url = canonicalize_grounded_url(raw_url)
        title = clean_text(getattr(web, "title", "") or "", 90)
        site = clean_text(extract_site(url) or title, 60)
        if not url or url in seen:
            continue
        seen.add(url)
        snippet = clean_text(" ".join(support_by_chunk.get(index, [])[:2]), 180)
        sources.append(
            {
                "title": title or site or "网页线索",
                "url": url,
                "site": site or extract_site(url),
                "snippet": snippet,
                "image_url": "",
            }
        )
    return sources, search_queries


def run_one_google_grounded_search(query: str, max_results: int) -> dict:
    prompt = f"""USER SEARCH REQUEST:
{query}

Please search in English, think from a US consumer perspective, and then return the JSON answer now."""
    response = google_search_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=google_genai_types.GenerateContentConfig(
            system_instruction=GOOGLE_SEARCH_SYNTHESIS_PROMPT,
            tools=[google_genai_types.Tool(googleSearch=google_genai_types.GoogleSearch())],
            response_mime_type="application/json",
        ),
    )
    parsed = parse_json(response.text)
    grounded_sources, grounded_queries = extract_grounding_sources(response)
    # Google grounding returns redirect proxy URLs (vertexaisearch.cloud.google.com)
    # which can't be resolved for og:image within timeout — skip image hydration
    grounded_sources = grounded_sources[:max_results]
    return {
        "answer": clean_text(parsed.get("answer", ""), 240),
        "results": grounded_sources,
        "queries": grounded_queries or parsed.get("query_focus", []) or [],
        "key_points": parsed.get("key_points", []) or [],
        "error": "",
    }


def diversify_sources_by_site(results: list[dict], max_results: int) -> list[dict]:
    if not results:
        return []
    picked: list[dict] = []
    seen_sites: set[str] = set()

    for item in results:
        site = (item.get("site") or "").strip()
        if site and site not in seen_sites:
            picked.append(item)
            seen_sites.add(site)
        if len(picked) >= max_results:
            return picked[:max_results]

    for item in results:
        if item not in picked:
            picked.append(item)
        if len(picked) >= max_results:
            break
    return picked[:max_results]


def search_multi(
    search_plan: list[tuple[str, str, str]],
    callback: ProgressCallback | None = None,
    phase: str = "full",
) -> dict:
    if not search_plan:
        return {}

    key, label, query = search_plan[0]
    query_display = build_social_site_query(query) if wants_social_posts(query) else query
    search_depth = "basic" if phase == "scoping" else "advanced"
    max_results = 2 if phase == "scoping" else 3

    emit_progress(
        callback,
        type="status",
        step="search",
        label="我去网上翻翻",
        detail=f"先抓和「{clean_text(label, 20)}」最相关的网页、讨论和线索。",
    )

    try:
        payload = run_one_search(query, search_depth, max_results, prefer_fast=(phase == "scoping"))
        emit_progress(
            callback,
            type="status",
            step="search",
            label="先摸到几条线索",
            detail="这轮先有个大概轮廓了，我继续往下拼。",
        )
        return {
            key: {
                "label": label,
                "query": query_display,
                **payload,
            }
        }
    except Exception as exc:
        error_text = clean_text(str(exc), 160)
        emit_progress(
            callback,
            type="status",
            step="search",
            label="这一轮线索不太顺",
            detail="这次网上回来的东西不太稳，我会尽量用现有信息继续判断。",
        )
        return {
            key: {
                "label": label,
                "query": query_display,
                "answer": "",
                "results": [],
                "error": error_text,
            }
        }


def should_skip_scoping(understanding: dict, product_hint: str, input_text: str) -> bool:
    combined = f"{input_text or ''} {product_hint or ''}".strip()
    return (
        understanding.get("intent") == "evaluate"
        and understanding.get("is_specific_product")
        and understanding.get("ambiguity_level") == "clear"
        and not understanding.get("needs_budget")
        and not understanding.get("needs_use_case")
        and not understanding.get("needs_skin_type")
        and not understanding.get("needs_occupation_context")
        and not has_budget_signal(combined)
        and not has_use_case_signal(combined)
        and not has_skin_signal(combined)
        and not has_occupation_signal(combined)
    )


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
                    "image_url": item.get("image_url", ""),
                }
            )
    return sources[:12]


def build_source_catalog(search_data: dict) -> list[dict]:
    catalog = []
    for index, source in enumerate(flatten_sources(search_data), start=1):
        item = dict(source)
        item["id"] = f"S{index}"
        catalog.append(item)
    return catalog


def source_catalog_block(source_catalog: list[dict]) -> str:
    if not source_catalog:
        return "No source catalog available."
    lines = []
    for source in source_catalog:
        lines.append(
            f"[{source['id']}] {source.get('title', 'Untitled')} | {source.get('site', 'unknown')} | {source.get('url', '')} | {source.get('snippet', '')}"
        )
    return "\n".join(lines)


def map_source_ids(source_ids: list[str] | None, source_lookup: dict[str, dict], fallback: list[dict] | None = None, limit: int = 3) -> list[dict]:
    resolved = []
    for source_id in source_ids or []:
        source = source_lookup.get(source_id)
        if source and source not in resolved:
            resolved.append(source)
    if not resolved and fallback:
        for source in fallback[:limit]:
            if source and source not in resolved:
                resolved.append(source)
    return resolved[:limit]


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
            "label": "我先弄清你在纠结什么",
            "detail": clean_text(understanding.get("user_goal", ""), 160),
        }
    ]
    if product_hint:
        trace.append({"label": "我认出来的主角", "detail": product_hint})

    if memory_meta:
        detail = (
            f"Used {memory_meta.get('used_history_count', 0)} relevant memory signals"
            f" and ignored {memory_meta.get('ignored_history_count', 0)} unrelated history."
        )
        trace.append({"label": "我顺手翻了下你的记忆", "detail": detail})

    if followup_qa:
        last_answer = clean_text(followup_qa[-1].get("answer", ""), 140)
        if last_answer:
            trace.append({"label": "你补给我的关键信息", "detail": last_answer})

    if presearch_followup:
        trace.append(
            {
                "label": "我先停下来问一句",
                "detail": clean_text(presearch_followup.get("reason", ""), 180),
            }
        )

    for bucket in search_data.values():
        sites = [item.get("site") for item in bucket.get("results", []) if item.get("site")]
        detail = bucket.get("answer") or ", ".join(dict.fromkeys(sites).keys()) or "No strong source summary"
        if bucket.get("error"):
            detail = f"{detail} ({bucket['error']})"
        trace.append({"label": f"我翻到的「{bucket.get('label', '线索')}」", "detail": clean_text(detail, 180)})
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
                "title": "价格快照",
                "value": result["price_range"],
                "detail": f"我顺手对了这些地方：{', '.join(dict.fromkeys(sites).keys())}" if sites else "",
            }
        )

    specs = result.get("key_specs") or {}
    if specs:
        preview = " | ".join([f"{k}: {v}" for k, v in list(specs.items())[:3]])
        cards.append(
            {
                "title": "关键参数",
                "value": f"{len(specs)} 个关键信号",
                "detail": preview,
            }
        )

    if result.get("result_type") == "recommendation":
        recommendations = result.get("recommendations") or []
        if recommendations:
            cards.append(
                {
                    "title": "我先替你圈了几款",
                    "value": ", ".join([item.get("name", "") for item in recommendations[:2] if item.get("name")]),
                    "detail": "是按场景、权衡点和当前市场信息一起收出来的。",
                }
            )
    else:
        alternatives = result.get("alternatives") or []
        if alternatives:
            cards.append(
                {
                    "title": "顺手替你看了别的选项",
                    "value": ", ".join([item.get("name", "") for item in alternatives[:2] if item.get("name")]),
                    "detail": "我拿相近价位和同类选手一起比了一圈。",
                }
            )
    return cards


def build_scoping_fact_cards(strategy: dict, search_data: dict) -> list:
    cards = []
    if strategy.get("preliminary_take"):
        cards.append(
            {
                "title": "初步判断",
                "value": clean_text(strategy["preliminary_take"], 44),
                "detail": "这是系统先基于商品和早期检索抓到的方向判断。",
            }
        )
    dimensions = strategy.get("decision_dimensions") or []
    if dimensions:
        first_two = dimensions[:2]
        cards.append(
            {
                "title": "优先判断维度",
                "value": ", ".join([item.get("title", "") for item in first_two if item.get("title")]),
                "detail": " | ".join([clean_text(item.get("detail", ""), 56) for item in first_two]),
            }
        )
    sources = flatten_sources(search_data)
    if sources:
        cards.append(
            {
                "title": "先看到的线索",
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
        "dimension_priority": [],
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
                "dimension_priority": ["安全", "适配度", "品质", "口碑", "品牌", "性价比"],
                "decision_dimensions": [
                    {"title": "安全", "status": "risk", "detail": "先确认你是否真的适合补这个，再谈品牌和剂量。"},
                    {"title": "适配度", "status": "mixed", "detail": "同样的产品，对不同身体状态和目标人群适配度差很多。"},
                    {"title": "品质", "status": "mixed", "detail": "原料纯度、剂量和稳定性也重要，但优先级要放在安全之后。"},
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
        {"title": "适配度", "status": "mixed", "detail": "先看这件东西和你的真实需求、使用场景到底对不对得上。"},
        {"title": "品质", "status": "mixed", "detail": "再看它本身做工、稳定性或配方质量是不是扎实。"},
        {"title": "性价比", "status": "mixed", "detail": "价格是不是和它能给到的体验、配置或效果相匹配。"},
        {"title": "口碑", "status": "mixed", "detail": "用户真实评价和反复出现的问题，决定值不值得放心下单。"},
    ]
    strategy["dimension_priority"] = ["适配度", "品质", "性价比", "口碑", "品牌", "安全"]
    return strategy


def clarify_without_search(
    understanding: dict,
    product_hint: str,
    user_profile: dict,
    followup_qa: list | None,
    followup_count: int,
    callback: ProgressCallback | None = None,
) -> dict:
    """Decide whether clarification is needed, purely from understanding + user profile.
    No web search or extra LLM call — fast and stable."""
    emit_progress(
        callback,
        type="status",
        step="clarify",
        label="先想想还差哪块关键信息",
        detail="基于你的问题和背景，看看有没有会明显影响判断的信息还没确认。",
    )

    # Safety-critical framework questions always fire first (pets, supplements, skincare, occupation).
    # These are category-specific and valid regardless of how clear the product is.
    framework = build_framework_followup(
        understanding=understanding,
        product_hint=product_hint,
        user_profile=user_profile or {},
        followup_qa=followup_qa,
        followup_count=followup_count,
        search_data={},
    )

    # For specific evaluate/compare products: only ask truly critical questions.
    # Use_case and budget are not critical — we can search and give a complete answer.
    # Recommend intent stays fully checked since the shortlist depends on budget/scenario.
    CRITICAL_KINDS = {"skin_type", "health_context", "pet_life_stage", "pet_sensitivity", "pet_size", "occupation_context"}
    is_specific_evaluate = (
        understanding.get("is_specific_product")
        and understanding.get("intent") in {"evaluate", "compare"}
    )
    if is_specific_evaluate and framework:
        # Suppress framework if it's not safety/context-critical (e.g., generic use_case)
        kind = framework.get("missing_dimension", "")
        if kind not in CRITICAL_KINDS:
            framework = None

    # Build base structure with sensible dimension defaults
    base = fallback_clarification_strategy(
        understanding=understanding,
        user_profile=user_profile or {},
        followup_qa=followup_qa,
        followup_count=followup_count,
        search_data={},
    )
    if is_specific_evaluate:
        base["needs_followup"] = False

    # Framework overrides fallback when it fires (only critical kinds reach here)
    if framework:
        base.update(framework)
        base["needs_followup"] = True

    # Normalize dimensions
    priority = normalize_dimension_priority(
        base.get("dimension_priority"),
        understanding=understanding,
        product_hint=product_hint,
        search_data={},
    )
    base["dimension_priority"] = priority
    base["decision_dimensions"] = normalize_dimension_items(base.get("decision_dimensions"), priority)

    # Build followup_questions array (required by build_early_followup_result)
    if base.get("needs_followup") and base.get("followup_question"):
        base["followup_questions"] = [
            {
                "question": base["followup_question"],
                "options": base.get("followup_options", []),
                "reason": base.get("followup_reason", ""),
                "question_type": base.get("question_type", "multiple_choice"),
                "open_text_placeholder": base.get("open_text_placeholder", ""),
            }
        ]
    else:
        base["needs_followup"] = False
        base["followup_questions"] = []

    base.setdefault("preliminary_take", "")
    base.setdefault("search_focus", [])
    return base


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
        label="我先看看还差哪块拼图",
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
    strategy.setdefault("dimension_priority", [])
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

    has_model_followup = bool(
        strategy.get("needs_followup")
        and (
            (strategy.get("followup_questions") and strategy["followup_questions"][0].get("question"))
            or strategy.get("followup_question")
        )
    )

    if has_model_followup:
        first = (strategy.get("followup_questions") or [{}])[0]
        strategy["followup_question"] = first.get("question") or strategy.get("followup_question", "")
        strategy["followup_options"] = first.get("options") or strategy.get("followup_options", [])
        strategy["question_type"] = first.get("question_type") or strategy.get("question_type", "multiple_choice")
        strategy["open_text_placeholder"] = first.get("open_text_placeholder") or strategy.get("open_text_placeholder", "")
        strategy["followup_reason"] = first.get("reason") or strategy.get("followup_reason", "")
        if strategy.get("question_type") == "multiple_choice":
            cleaned_options = []
            for option in strategy.get("followup_options", []):
                text = clean_text(option, 80)
                if text and text not in cleaned_options:
                    cleaned_options.append(text)
            if len(cleaned_options) >= 3:
                strategy["followup_options"] = cleaned_options[:3]
            elif framework_followup:
                strategy.update(framework_followup)
            else:
                strategy["followup_options"] = cleaned_options[:3]
        strategy["followup_questions"] = [
            {
                "question": strategy.get("followup_question", ""),
                "options": strategy.get("followup_options", []),
                "reason": strategy.get("followup_reason", ""),
                "question_type": strategy.get("question_type", "multiple_choice"),
                "open_text_placeholder": strategy.get("open_text_placeholder", ""),
            }
        ]
    elif framework_followup:
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

    priority = normalize_dimension_priority(
        strategy.get("dimension_priority"),
        understanding=understanding,
        product_hint=product_hint,
        search_data=search_data,
    )
    strategy["dimension_priority"] = priority
    strategy["decision_dimensions"] = normalize_dimension_items(strategy.get("decision_dimensions"), priority)

    # Ensure decision_dimensions has at least a default if empty
    if not strategy.get("decision_dimensions"):
        fallback = fallback_clarification_strategy(
            understanding=understanding,
            user_profile=user_profile,
            followup_qa=followup_qa,
            followup_count=followup_count,
            search_data=search_data,
        )
        strategy["dimension_priority"] = normalize_dimension_priority(
            fallback.get("dimension_priority"),
            understanding=understanding,
            product_hint=product_hint,
            search_data=search_data,
        )
        strategy["decision_dimensions"] = normalize_dimension_items(
            fallback.get("decision_dimensions"),
            strategy["dimension_priority"],
        )

    strategy["preliminary_take"] = ensure_chinese_text(strategy.get("preliminary_take", ""))
    strategy["followup_reason"] = ensure_chinese_text(strategy.get("followup_reason", ""))
    strategy["followup_question"] = ensure_chinese_text(strategy.get("followup_question", ""))
    strategy["open_text_placeholder"] = ensure_chinese_text(strategy.get("open_text_placeholder", ""))
    strategy["followup_options"] = [
        ensure_chinese_text(option) for option in (strategy.get("followup_options") or []) if clean_text(option, 80)
    ]
    normalized_questions = []
    for item in strategy.get("followup_questions", []) or []:
        normalized_questions.append(
            {
                **item,
                "question": ensure_chinese_text(item.get("question", "")),
                "options": [ensure_chinese_text(option) for option in (item.get("options") or []) if clean_text(option, 80)],
                "reason": ensure_chinese_text(item.get("reason", "")),
                "open_text_placeholder": ensure_chinese_text(item.get("open_text_placeholder", "")),
            }
        )
    strategy["followup_questions"] = normalized_questions
    strategy["decision_dimensions"] = [
        {**item, "detail": ensure_chinese_text(item.get("detail", ""))}
        for item in strategy.get("decision_dimensions", [])
    ]

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
    priority = normalize_dimension_priority(
        clarification.get("dimension_priority"),
        understanding=understanding,
        product_hint=product_hint,
        search_data=search_data,
    )
    source_catalog = build_source_catalog(search_data)
    source_lookup = {item["id"]: {k: v for k, v in item.items() if k != "id"} for item in source_catalog}
    fallback_sources = [source_lookup[item["id"]] for item in source_catalog]
    return {
        "result_type": result_type,
        "intent": understanding.get("intent", "evaluate"),
        "headline": "先问你一句，再继续帮你查",
        "product_name": product_hint if result_type == "decision" else "",
        "category": understanding.get("category_hint", "other"),
        "price_range": "",
        "verdict": None if result_type == "recommendation" else "cautious",
        "summary": clarification.get("preliminary_take") or f"我先做了个快速判断，这里还差 {question_count} 个会明显影响结论的关键信息。",
        "dimension_priority": priority,
        "decision_dimensions": normalize_dimension_items(clarification.get("decision_dimensions", []), priority),
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
            "sources": fallback_sources,
            "fact_cards": build_scoping_fact_cards(clarification, search_data),
        },
        "summary_sources": fallback_sources[:2],
        "reason_source_groups": [],
        "report_sections": [],
        "followup_reason": clarification.get("followup_reason", ""),
    }


def normalize_display_modules(modules: list[dict] | None, result: dict, source_lookup: dict[str, dict], fallback_sources: list[dict]) -> list[dict]:
    def fallback_module_candidates() -> list[dict]:
        fallback_modules = []
        summary_sources = result.get("summary_sources") or fallback_sources[:2]
        if result.get("summary"):
            fallback_modules.append(
                {
                    "type": "summary_card",
                    "title": result.get("headline") or "我的判断",
                    "body": result.get("summary", ""),
                    "sources": summary_sources,
                    "items": [],
                }
            )
        if result.get("fit_summary"):
            fallback_modules.append(
                {
                    "type": "text_block",
                    "title": "为什么这个判断跟你有关",
                    "body": result.get("fit_summary", ""),
                    "sources": summary_sources,
                    "items": [],
                }
            )
        if result.get("decision_dimensions"):
            fallback_modules.append(
                {
                    "type": "decision_dimensions",
                    "title": "核心决策维度",
                    "body": "",
                    "sources": summary_sources,
                    "items": [
                        {
                            "title": item.get("title", ""),
                            "body": item.get("detail", ""),
                            "footer": item.get("status", ""),
                            "image_url": "",
                            "sources": summary_sources,
                        }
                        for item in result.get("decision_dimensions", [])[:6]
                        if item.get("status", "").strip().lower() not in {"mixed", "unknown", "info_limited", "information_limited"}
                        and item.get("detail", "").strip()
                    ],
                }
            )
        recommendations = result.get("recommendations") or []
        if recommendations:
            fallback_modules.append(
                {
                    "type": "recommendation_carousel",
                    "title": "如果你想顺手比一比",
                    "body": "",
                    "sources": summary_sources,
                    "items": [
                        {
                            "title": item.get("name", ""),
                            "body": item.get("reason", "") or item.get("best_for", ""),
                            "footer": item.get("price_hint", "") or ("；".join(item.get("tradeoffs", [])[:2]) if item.get("tradeoffs") else ""),
                            "image_url": item.get("image_url", ""),
                            "sources": summary_sources,
                        }
                        for item in recommendations[:5]
                        if item.get("name")
                    ],
                }
            )
        source_cards = []
        for source in (result.get("research", {}) or {}).get("sources", [])[:6]:
            if source.get("image_url"):
                source_cards.append(
                    {
                        "title": source.get("title", source.get("site", "网页线索")),
                        "body": source.get("snippet", ""),
                        "footer": source.get("site", ""),
                        "image_url": source.get("image_url", ""),
                        "sources": [source],
                    }
                )
        if source_cards:
            fallback_modules.append(
                {
                    "type": "source_gallery",
                    "title": "我顺手翻到的图和网页",
                    "body": "",
                    "sources": [],
                    "items": source_cards,
                }
            )
        if result.get("report_sections"):
            for section in result.get("report_sections", [])[:4]:
                fallback_modules.append(
                    {
                        "type": "text_block",
                        "title": section.get("title", ""),
                        "body": section.get("body", ""),
                        "sources": section.get("sources") or summary_sources,
                        "items": [],
                    }
                )
        return [module for module in fallback_modules if module.get("body") or module.get("items")]

    normalized = []
    for module in modules or []:
        module_type = (module.get("type") or "").strip()
        if module_type not in {"summary_card", "text_block", "decision_dimensions", "recommendation_carousel", "comparison_cards", "source_gallery"}:
            continue
        normalized_module = {
            "type": module_type,
            "title": clean_text(module.get("title", ""), 60),
            "body": clean_text(module.get("body", ""), 500),
            "sources": map_source_ids(module.get("source_ids") or [], source_lookup, fallback=fallback_sources[:2], limit=3),
            "items": [],
        }
        for item in module.get("items") or []:
            normalized_module["items"].append(
                {
                    "title": clean_text(item.get("title", ""), 80),
                    "body": clean_text(item.get("body", ""), 320),
                    "footer": clean_text(item.get("footer", ""), 80),
                    "image_url": item.get("image_url", ""),
                    "sources": map_source_ids(item.get("source_ids") or [], source_lookup, fallback=normalized_module["sources"], limit=2),
                }
            )
        if normalized_module.get("body") or normalized_module.get("items"):
            normalized.append(normalized_module)

    if not normalized:
        return fallback_module_candidates()

    singleton_types = {"summary_card", "decision_dimensions", "recommendation_carousel", "comparison_cards", "source_gallery"}

    def module_key(module: dict):
        if module.get("type") in singleton_types:
            return module.get("type")
        return (module.get("type"), module.get("title"), module.get("body"))

    seen = {module_key(module) for module in normalized}
    for module in fallback_module_candidates():
        key = module_key(module)
        if key not in seen:
            normalized.append(module)
            seen.add(key)
        if len(normalized) >= 6:
            break
    return normalized[:6]


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
    result.setdefault("dimension_priority", [])
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
    result.setdefault("summary_source_ids", [])
    result.setdefault("reason_source_ids", [])
    result.setdefault("display_modules", [])

    raw_image_search = result.get("image_search") if isinstance(result.get("image_search"), dict) else {}
    result["image_search"] = {
        "needed": bool(raw_image_search.get("needed")),
        "query": clean_text(raw_image_search.get("query", ""), 140),
        "reason": clean_text(raw_image_search.get("reason", ""), 180),
    }

    priority = normalize_dimension_priority(
        result.get("dimension_priority") or clarification_dimension_priority(result),
        understanding=understanding,
        product_hint=product_hint,
        search_data=search_data,
    )
    result["dimension_priority"] = priority
    result["decision_dimensions"] = normalize_dimension_items(result.get("decision_dimensions"), priority)

    raw_scores = result.get("scores") or {}
    if isinstance(raw_scores, dict):
        result["scores"] = {
            "quality": int(raw_scores.get("quality", 3) or 3),
            "cost_value": int(raw_scores.get("cost_value", 3) or 3),
            "fit": int(raw_scores.get("fit", 3) or 3),
            "reviews": int(raw_scores.get("reviews", 3) or 3),
            "brand": int(raw_scores.get("brand", 3) or 3),
            "safety": int(raw_scores.get("safety", raw_scores.get("longevity", 3)) or 3),
        }

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
                "title": "我是怎么按这 6 个维度帮你筛的",
                "body": result.get("fit_summary") or result.get("summary") or understanding.get("user_goal", ""),
                "source_ids": result.get("summary_source_ids", []),
            }
        ]
        if result.get("better_direction"):
            report_sections.append(
                {
                    "title": "如果你想继续收窄方向",
                    "body": result["better_direction"],
                    "source_ids": result.get("summary_source_ids", []),
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
                    "source_ids": result.get("summary_source_ids", []),
                }
            )
        elif result.get("verdict") == "cautious":
            report_sections.append(
                {
                    "title": "我会先替你盯住这一点",
                    "body": result.get("caution_check") or result.get("summary") or "",
                    "source_ids": result.get("summary_source_ids", []),
                }
            )
            if result.get("buy_if"):
                report_sections.append({"title": "如果你是这种情况，还是可以考虑", "body": result["buy_if"], "source_ids": result.get("summary_source_ids", [])})
            if result.get("skip_if"):
                report_sections.append({"title": "如果不是这种情况，更建议先等等", "body": result["skip_if"], "source_ids": result.get("summary_source_ids", [])})
        else:
            if result.get("better_direction"):
                report_sections.append({"title": "比继续纠结这款更重要的是", "body": result["better_direction"], "source_ids": result.get("summary_source_ids", [])})
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
                        "source_ids": result.get("summary_source_ids", []),
                    }
                )
            if result.get("budget_alternative"):
                budget = result["budget_alternative"]
                report_sections.append(
                    {
                        "title": "如果你更想把钱花得稳一点",
                        "body": f"{budget.get('name', '备选项')}：{budget.get('reason', '')}".strip("："),
                        "source_ids": result.get("summary_source_ids", []),
                    }
                )

        if result.get("fit_summary"):
            report_sections.insert(
                0,
                {
                    "title": "它和你的情况到底对不对得上",
                    "body": result["fit_summary"],
                    "source_ids": result.get("summary_source_ids", []),
                },
            )

    if result.get("decision_dimensions"):
        dimension_lines = []
        for item in result["decision_dimensions"][:6]:
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
                    "title": "这次我是按哪些维度做判断的",
                    "body": "；".join(dimension_lines),
                    "source_ids": result.get("summary_source_ids", []),
                },
            )

    source_catalog = build_source_catalog(search_data)
    source_lookup = {item["id"]: {k: v for k, v in item.items() if k != "id"} for item in source_catalog}
    fallback_sources = [source_lookup[item["id"]] for item in source_catalog]
    result["summary_sources"] = map_source_ids(result.get("summary_source_ids"), source_lookup, fallback=fallback_sources[:2], limit=2)

    reason_groups = result.get("reason_source_ids") or []
    resolved_reason_groups = []
    for index, _ in enumerate(result.get("reasons") or []):
        ids = reason_groups[index] if index < len(reason_groups) and isinstance(reason_groups[index], list) else []
        fallback = fallback_sources[index:index + 2] or fallback_sources[:1]
        resolved_reason_groups.append(map_source_ids(ids, source_lookup, fallback=fallback, limit=2))
    result["reason_source_groups"] = resolved_reason_groups

    normalized_sections = []
    for section in report_sections:
        section_ids = section.get("source_ids") or []
        section["sources"] = map_source_ids(section_ids, source_lookup, fallback=fallback_sources[:2], limit=2)
        normalized_sections.append(section)

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
        "sources": fallback_sources,
        "fact_cards": build_fact_cards(result, search_data),
    }
    result["report_sections"] = [section for section in normalized_sections if section.get("body")]
    result["display_modules"] = normalize_display_modules(
        result.get("display_modules"),
        result,
        source_lookup,
        fallback_sources,
    )
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
        label="我来收个尾",
        detail="把网页线索、你的情况和真正重要的点拼在一起。",
    )

    followup_text = ""
    if followup_qa:
        qa_lines = [f"Q: {item['question']}\nA: {item['answer']}" for item in followup_qa]
        followup_text = "FOLLOW-UP CONTEXT:\n" + "\n".join(qa_lines) + "\n\n"

    source_catalog = build_source_catalog(search_data)

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

SOURCE CATALOG:
{source_catalog_block(source_catalog)}

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
        label="先接住你这条输入",
        detail="我先判断你丢给我的是具体商品，还是一个更泛的购买需求。",
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
            label="我先认认图里的主角",
            detail="先看清图里到底是什么，再继续往下拆。",
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

    # Fast rule-based clarification — no web search needed at this stage
    clarification = clarify_without_search(
        understanding=understanding,
        product_hint=product_hint,
        user_profile=user_profile or {},
        followup_qa=[],
        followup_count=0,
        callback=callback,
    )
    scoping_data = {}
    if clarification.get("needs_followup"):
        emit_progress(
            callback,
            type="status",
            step="clarify",
            label="还差一小块关键信息",
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
    normalized = normalize_result(
        result=raw_result,
        product_hint=product_hint,
        understanding=understanding,
        search_data=search_data,
        user_profile=user_profile or {},
        followup_count=0,
        memory_meta=memory_meta,
        followup_qa=[],
    )
    if len(normalized.get("followup_questions") or []) > 0:
        return normalized
    return attach_images_from_search(normalized, search_data)


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
        label="带着你的回答继续翻",
        detail="这次我就不兜圈子了，直接带着你的补充去查。",
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
    normalized = normalize_result(
        result=raw_result,
        product_hint=product_hint,
        understanding=understanding,
        search_data=search_data,
        user_profile=user_profile or {},
        followup_count=len(followup_qa),
        memory_meta=memory_meta,
        followup_qa=followup_qa,
    )
    if (normalized.get("followup_questions") or []) and len(followup_qa or []) < 3:
        return normalized
    return attach_images_from_search(normalized, search_data)
