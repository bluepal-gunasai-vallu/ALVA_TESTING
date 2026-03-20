# Simple in-memory session storage

sessions = {}

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "fsm_state": "INQUIRY",
            "slots": {},
            "_clarif_count": 0,      # TC049: infinite-loop guard
            "readback_done": False,  # TC053: confirmation readback flag
        }
    # Back-fill keys that may be missing in older sessions
    session = sessions[session_id]
    session.setdefault("_clarif_count", 0)
    session.setdefault("readback_done", False)
    return session


def save_session(session_id, data):
    sessions[session_id] = data


def update_fsm_state(session_id: str, state: str):
    """TC006: Persist the current FSM state into the session so it
    survives call-drop / reconnect recovery."""
    if session_id in sessions:
        sessions[session_id]["fsm_state"] = state


def clear_session(session_id):
    """TC058: Remove session data when a call ends."""
    if session_id in sessions:
        del sessions[session_id]


# ──────────────────────────────────────────────────
# TC081: Global call counter
# Stored here (not in main.py) so both main.py and
# doctor_routes.py can read/write it without a
# circular import.
# ──────────────────────────────────────────────────
total_calls: int = 0

def increment_total_calls():
    global total_calls
    total_calls += 1

def get_total_calls() -> int:
    return total_calls