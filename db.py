import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ.get("SUPABASE_SECRET_KEY") or os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)


def save_session(user_id: str, product_name: str, category: str,
                 input_text: str, verdict: str, verdict_reason: str,
                 followup_qa: list) -> dict:
    result = supabase.table("sessions").insert({
        "user_id": user_id,
        "product_name": product_name,
        "category": category,
        "input_text": input_text,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "followup_qa": followup_qa,
    }).execute()
    return result.data[0] if result.data else {}


def get_user_sessions(user_id: str, limit: int = 20) -> list:
    result = supabase.table("sessions") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    return result.data or []


def get_or_create_profile(user_id: str) -> dict:
    result = supabase.table("user_profiles") \
        .select("*") \
        .eq("user_id", user_id) \
        .execute()
    if result.data:
        return result.data[0]
    supabase.table("user_profiles").insert({"user_id": user_id}).execute()
    return {"user_id": user_id}


def update_profile(user_id: str, updates: dict):
    supabase.table("user_profiles") \
        .upsert({"user_id": user_id, **updates}) \
        .execute()


def merge_profile(user_id: str, new_signals: dict):
    """Merge new signals into existing profile with list deduplication."""
    existing = get_or_create_profile(user_id)
    merged = {"user_id": user_id}

    plain_list_fields = {"use_cases", "owned_items", "primary_use_cases", "lifestyle_hints"}
    # pets is a list of dicts — deduplicate by species+breed identity
    all_keys = set(existing.keys()) | set(new_signals.keys()) - {"user_id", "created_at", "updated_at"}

    for key in all_keys:
        new_val = new_signals.get(key)
        old_val = existing.get(key)
        if key == "pets":
            old_list = old_val if isinstance(old_val, list) else []
            new_list = new_val if isinstance(new_val, list) else ([new_val] if new_val else [])
            # Merge: update existing pets by species+breed, add truly new ones
            merged_pets = {(p.get("species", ""), p.get("breed", "")): p for p in old_list if isinstance(p, dict)}
            for pet in new_list:
                if isinstance(pet, dict):
                    key_tuple = (pet.get("species", ""), pet.get("breed", ""))
                    if key_tuple in merged_pets:
                        merged_pets[key_tuple] = {**merged_pets[key_tuple], **{k: v for k, v in pet.items() if v}}
                    else:
                        merged_pets[key_tuple] = pet
            merged[key] = list(merged_pets.values())[:5]
        elif key in plain_list_fields:
            old_list = old_val if isinstance(old_val, list) else []
            new_list = new_val if isinstance(new_val, list) else ([new_val] if new_val else [])
            combined = list(dict.fromkeys(old_list + new_list))
            merged[key] = combined[:10]
        elif new_val is not None and new_val != "":
            merged[key] = new_val
        elif old_val is not None:
            merged[key] = old_val

    supabase.table("user_profiles").upsert(merged).execute()
