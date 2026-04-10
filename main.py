import json
import os
import queue
import threading
import uuid
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent import (
    analyze,
    analyze_with_followup,
    extract_profile_signals_from_answer,
    fetch_binary,
    find_verified_preview_image,
)
from db import get_or_create_profile, get_user_sessions, merge_profile, save_session
from models import AnalyzeRequest, FollowupRequest, PreviewImageRequest, ProfileUpdateRequest

app = FastAPI(title="Buy or Not API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pending_sessions: dict[str, dict] = {}


def ndjson_line(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def build_image_proxy_url(base_url: str, image_url: str) -> str:
    if base_url.startswith("http://") and all(token not in base_url for token in ["localhost", "127.0.0.1"]):
        base_url = "https://" + base_url.removeprefix("http://")
    cleaned = (image_url or "").strip()
    if not cleaned.startswith(("http://", "https://")):
        return cleaned
    prefix = base_url.rstrip("/") + "/image-proxy?url="
    if cleaned.startswith(prefix):
        return cleaned
    return prefix + quote(cleaned, safe="")


def rewrite_image_urls(payload, base_url: str):
    if isinstance(payload, dict):
        updated = {}
        for key, value in payload.items():
            if key == "image_url" and isinstance(value, str):
                updated[key] = build_image_proxy_url(base_url, value)
            else:
                updated[key] = rewrite_image_urls(value, base_url)
        return updated
    if isinstance(payload, list):
        return [rewrite_image_urls(item, base_url) for item in payload]
    return payload


def guess_media_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.lstrip().startswith(b"<svg") or b"<svg" in image_bytes[:200]:
        return "image/svg+xml"
    return "application/octet-stream"



def result_storage_name(result: dict, fallback: str = "") -> str:
    if result.get("product_name"):
        return result["product_name"]
    recommendations = result.get("recommendations") or []
    if recommendations and recommendations[0].get("name"):
        return recommendations[0]["name"]
    primary = result.get("primary_recommendation") or {}
    if primary.get("name"):
        return primary["name"]
    return fallback


def finalize_analysis_response(req: AnalyzeRequest, result: dict) -> tuple[str, dict]:
    session_id = str(uuid.uuid4())
    original_input = req.text or result_storage_name(result, "[image]") or "[image]"

    if result.get("followup"):
        pending_sessions[session_id] = {
            "original_input": original_input,
            "analysis_mode": req.analysis_mode,
            "followup_qa": [],
            "image_base64": req.image_base64,
            "pending_followup_questions": result.get("followup_questions", []),
        }
        response = {
            "session_id": session_id,
            "original_input": original_input,
            "followup_round": 1,
            **result,
        }
        return "followup", response

    save_session(
        user_id=req.user_id,
        product_name=result_storage_name(result),
        category=result.get("category", ""),
        input_text=original_input,
        verdict=result.get("verdict") or "worth_buying",
        verdict_reason=result.get("summary", ""),
        followup_qa=[],
    )
    response = {
        "session_id": session_id,
        "original_input": original_input,
        "followup_round": 0,
        **result,
    }
    return "result", response


def finalize_followup_response(req: FollowupRequest, session_state: dict, result: dict, answer_pairs: list[dict]) -> tuple[str, dict]:
    followup_qa = session_state.get("followup_qa", []) + answer_pairs

    if result.get("followup") and len(followup_qa) < 3:
        session_state["followup_qa"] = followup_qa
        session_state["pending_followup_questions"] = result.get("followup_questions", [])
        pending_sessions[req.session_id] = session_state
        response = {
            "session_id": req.session_id,
            "original_input": session_state["original_input"],
            "followup_round": len(followup_qa) + 1,
            **result,
        }
        return "followup", response

    pending_sessions.pop(req.session_id, None)
    save_session(
        user_id=req.user_id,
        product_name=result_storage_name(result),
        category=result.get("category", ""),
        input_text=session_state["original_input"],
        verdict=result.get("verdict") or "worth_buying",
        verdict_reason=result.get("summary", ""),
        followup_qa=followup_qa,
    )
    response = {
        "session_id": req.session_id,
        "original_input": session_state["original_input"],
        "followup_round": len(followup_qa),
        **result,
    }
    return "result", response


def stream_worker(run_analysis):
    event_queue: queue.Queue = queue.Queue()
    sentinel = object()

    def emit(event: dict):
        event_queue.put(event)

    def runner():
        try:
            final_event = run_analysis(emit)
            event_queue.put(final_event)
        except Exception as exc:
            event_queue.put(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )
        finally:
            event_queue.put(sentinel)

    threading.Thread(target=runner, daemon=True).start()

    while True:
        item = event_queue.get()
        if item is sentinel:
            break
        yield ndjson_line(item)


def run_analyze(req: AnalyzeRequest, emit, base_url: str | None = None) -> dict:
    if not req.text and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide text or image_base64")

    profile = get_or_create_profile(req.user_id)
    history = get_user_sessions(req.user_id, limit=10)

    result = analyze(
        input_text=req.text,
        image_base64=req.image_base64,
        user_profile=profile,
        user_history=history,
        analysis_mode=req.analysis_mode,
        callback=emit,
    )
    event_type, payload = finalize_analysis_response(req, result)
    if base_url:
        payload = rewrite_image_urls(payload, base_url)
    return {"type": event_type, "data": payload}


def run_followup(req: FollowupRequest, emit, base_url: str | None = None) -> dict:
    session_state = pending_sessions.get(req.session_id)
    if not session_state:
        raise HTTPException(status_code=404, detail="Follow-up session not found or expired")

    answer_pairs = []
    if req.answers:
        answer_pairs = [
            {"question": item.question, "answer": item.answer}
            for item in req.answers
            if item.question and item.answer
        ]
    elif req.question and req.answer:
        answer_pairs = [{"question": req.question, "answer": req.answer}]

    if not answer_pairs:
        raise HTTPException(status_code=400, detail="Provide follow-up answers")

    for item in answer_pairs:
        signals = extract_profile_signals_from_answer(
            item["question"],
            item["answer"],
            original_input=session_state.get("original_input", ""),
        )
        if signals:
            merge_profile(req.user_id, signals)

    session_state["followup_qa"] = session_state.get("followup_qa", []) + answer_pairs

    profile = get_or_create_profile(req.user_id)
    history = get_user_sessions(req.user_id, limit=10)
    result = analyze_with_followup(
        original_input=session_state["original_input"],
        followup_qa=session_state["followup_qa"],
        user_profile=profile,
        user_history=history,
        analysis_mode=session_state.get("analysis_mode", "decision"),
        image_base64=session_state.get("image_base64"),
        callback=emit,
    )
    event_type, payload = finalize_followup_response(req, session_state, result, answer_pairs)
    if base_url:
        payload = rewrite_image_urls(payload, base_url)
    return {"type": event_type, "data": payload}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/image-proxy")
def image_proxy(url: str):
    cleaned = (url or "").strip()
    if not cleaned.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid image url")
    try:
        image_bytes = fetch_binary(cleaned, timeout=10)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image fetch failed: {exc}") from exc
    if not image_bytes:
        raise HTTPException(status_code=404, detail="Empty image response")
    return Response(
        content=image_bytes,
        media_type=guess_media_type(image_bytes),
        headers={
            "Cache-Control": "public, max-age=86400",
        },
    )


@app.post("/analyze")
def analyze_product(req: AnalyzeRequest, request: Request):
    capture: list[dict] = []
    payload = run_analyze(req, capture.append, base_url=str(request.base_url).rstrip("/"))
    return payload["data"]


@app.post("/analyze/stream")
def analyze_product_stream(req: AnalyzeRequest, request: Request):
    return StreamingResponse(
        stream_worker(lambda emit: run_analyze(req, emit, base_url=str(request.base_url).rstrip("/"))),
        media_type="application/x-ndjson",
    )


@app.post("/followup")
def answer_followup(req: FollowupRequest, request: Request):
    payload = run_followup(req, lambda _: None, base_url=str(request.base_url).rstrip("/"))
    return payload["data"]


@app.post("/followup/stream")
def answer_followup_stream(req: FollowupRequest, request: Request):
    return StreamingResponse(
        stream_worker(lambda emit: run_followup(req, emit, base_url=str(request.base_url).rstrip("/"))),
        media_type="application/x-ndjson",
    )


@app.get("/history/{user_id}")
def get_history(user_id: str):
    sessions = get_user_sessions(user_id)
    return {"sessions": sessions}


@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    profile = get_or_create_profile(user_id)
    return {"profile": profile}


@app.post("/profile/{user_id}/merge")
def merge_profile_data(user_id: str, req: ProfileUpdateRequest):
    merge_profile(user_id, req.updates or {})
    profile = get_or_create_profile(user_id)
    return {"profile": profile}


@app.post("/preview-image")
def preview_image(req: PreviewImageRequest):
    result = find_verified_preview_image(req.text)
    return result or {
        "matched": False,
        "image_base64": None,
        "page_url": "",
        "image_url": "",
        "reason": "No verified product image found.",
    }


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
