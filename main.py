import json
import os
import queue
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent import analyze, analyze_with_followup, extract_profile_signals_from_answer
from db import get_or_create_profile, get_user_sessions, merge_profile, save_session
from models import AnalyzeRequest, FollowupRequest, ProfileUpdateRequest

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


def run_analyze(req: AnalyzeRequest, emit) -> dict:
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
    return {"type": event_type, "data": payload}


def run_followup(req: FollowupRequest, emit) -> dict:
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
    return {"type": event_type, "data": payload}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze_product(req: AnalyzeRequest):
    capture: list[dict] = []
    payload = run_analyze(req, capture.append)
    return payload["data"]


@app.post("/analyze/stream")
def analyze_product_stream(req: AnalyzeRequest):
    return StreamingResponse(
        stream_worker(lambda emit: run_analyze(req, emit)),
        media_type="application/x-ndjson",
    )


@app.post("/followup")
def answer_followup(req: FollowupRequest):
    payload = run_followup(req, lambda _: None)
    return payload["data"]


@app.post("/followup/stream")
def answer_followup_stream(req: FollowupRequest):
    return StreamingResponse(
        stream_worker(lambda emit: run_followup(req, emit)),
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


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
