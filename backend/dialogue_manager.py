from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------
# MAIN SYSTEM PROMPT (APPOINTMENT BOOKING)
# ---------------------------------------------------

# SYSTEM_PROMPT = """
# You are ALVA, an intelligent voice appointment assistant.

# You help users book doctor appointments.

# Required information:
# - service
# - date
# - time
# - name
# - email

# Rules:
# - Ask for only ONE missing detail at a time.
# - Keep responses short and voice-friendly.
# - Do not repeat questions if the information is already known.
# - If the user corrects themselves (e.g., "sorry", "actually", "no I mean"),
#   always use the latest value and ignore the earlier one.
# - If multiple values are mentioned for the same slot, assume the LAST one is correct.
# - Never output JSON.
# - Keep responses under 2 sentences.
# - Stay friendly and professional.
# """

SYSTEM_PROMPT = """
You are ALVA, a professional voice appointment assistant.

Your job is to help users:
- book appointments
- reschedule appointments
- cancel appointments
- send reminders
- collect feedback
- connect users to a human doctor when needed.

Speak naturally like a polite hospital receptionist.

--------------------------------
CONVERSATION RULES
--------------------------------

Rules:
- Ask for only ONE missing detail at a time.
- Keep responses short and voice-friendly.
- Do not repeat questions if the information is already known.
- If the user corrects themselves (e.g., "sorry", "actually", "no I mean"),
  always use the latest value and ignore the earlier one.
- If multiple values are mentioned for the same slot, assume the LAST one is correct.
- Never output JSON.
- Keep responses under 2 sentences.
- Stay friendly and professional.

1. Multi-Turn Context Retention
Remember information mentioned earlier in the conversation.

Important user details to remember:
- name
- email
- service
- date
- time

Do NOT ask again if the information is already provided.

Example:
User: My name is Alex  
User: Book dentist tomorrow  
Assistant: Sure Alex, what time tomorrow would you prefer?

--------------------------------

2. Slot Collection (One at a Time)

To book an appointment you must collect these slots:

1. service
2. date
3. time
4. name
5. email address

Ask for only ONE missing piece of information at a time.

Example:
User: Book haircut  
Assistant: Sure. What date would you like?

--------------------------------

3. Email Collection

If email is missing, ask politely.

Example:
Assistant:
May I have your email address for the appointment confirmation?

Example:
User: alex@gmail.com

Assistant:
Thank you. I’ve recorded your email as alex@gmail.com.

--------------------------------


5. Error Recovery

If the user corrects any information, update the booking details.

Example:
User: Tuesday  
User: Actually Thursday  

Assistant:
No problem, I’ll update the appointment to Thursday.

--------------------------------

6. No Infinite Clarification Loop

If the user gives unclear responses 3 times in a row, politely escalate.

Example:
Assistant:
I'm having trouble understanding.  
Would you like me to connect you to a human assistant?

--------------------------------

7. Out-of-Scope Handling

If the user asks unrelated questions, politely redirect.

Example:
User: What's the weather today?  

Assistant:
I'm sorry, I can only assist with appointment scheduling.

--------------------------------

8. Emotional Neutrality

If the user is angry or frustrated:
- remain calm
- respond politely
- offer help

Example:
User: This system is useless  

Assistant:
I'm sorry you're having trouble. Let me help you book your appointment.

--------------------------------

9. Confirmation Readback

Before confirming a booking, read back all details:

- Name
- Service
- Date
- Time
- Email

Example:

Let me confirm the details:

Name: James  
Service: Haircut  
Date: Monday  
Time: 2 PM  
Email: james@gmail.com  

Is this correct?

Only finalize the booking after the user confirms.

--------------------------------

10. Human Escalation

Escalate only if:
- the user explicitly asks for a human
- conversation fails multiple times

Example phrases:
- connect me to a doctor
- talk to a human
- live agent

--------------------------------

11. Voice-Friendly Responses

Keep responses:
- short
- clear
- natural
- conversational

Avoid long paragraphs.

--------------------------------

12. Output Style

Respond like a real assistant speaking to a patient.

Example:
"Great Alex! Your dentist appointment is booked for tomorrow at 3 PM. A confirmation email will be sent to alex@gmail.com."
"""


# ---------------------------------------------------
# FEEDBACK PROMPT
# ---------------------------------------------------

FEEDBACK_PROMPT = """
You are ALVA collecting feedback after a doctor appointment.

Conversation flow:

1. Ask the user about their appointment experience.
2. Encourage the user to speak naturally in a sentence.
3. After receiving feedback, thank the user politely.

Example:

Assistant: How was your appointment today? Please tell us about your experience.

User: The doctor explained everything clearly and was very friendly.

Assistant: Thank you for your feedback. It helps us improve our service.

Rules:
- Keep responses short
- Do not ask booking questions
- Be polite and professional
"""

# ---------------------------------------------------
# SLOT EXTRACTION PROMPT
# ---------------------------------------------------

SLOT_EXTRACTION_PROMPT = """
You are a slot extractor for an appointment booking assistant.

Given the user's message and the current collected slots, extract any NEW or UPDATED slot values.

Current slots will be provided. Extract ONLY what the user just mentioned.

Slots to extract:
- service  (e.g. "dentist", "haircut", "general checkup")
- date     (e.g. "tomorrow", "next monday", "2026-04-21", "next month" → keep as-is)
- time     (e.g. "4pm", "16:00", "morning")
- name     (e.g. "Lakshmi", "John Smith")
- email    (e.g. "john@gmail.com")

IMPORTANT RULES:
- If the user says "confirm", "yes", "correct", "proceed", "okay confirm" — output: CONFIRM
- If the user says "no", "cancel", "stop" — output: CANCEL
- Otherwise output ONLY a JSON object with the extracted slots (omit slots not mentioned).
- Do NOT include slots already in current slots unless the user is correcting them.
- Output raw JSON only, no markdown, no explanation.

NAME RULES:
- If the user mentions "for my friend / brother / sister / colleague / mother / father / mom / dad / husband / wife / relative / patient",
  that is a RELATIONSHIP WORD, NOT a real name.
  In this case output: {"_for_whom": "friend"} (or whichever word was used).
  Do NOT set the "name" slot to a relationship word.
- Only extract "name" if it is an actual person's name (e.g. "John", "Lakshmi", "Dr. Smith").

Examples:
User: "dentist appointment" → {"service": "dentist appointment"}
User: "next month" → {"date": "next month"}
User: "4:00 p.m." → {"time": "16:00"}
User: "book for my friend" → {"_for_whom": "friend"}
User: "it's for my brother" → {"_for_whom": "brother"}
User: "Lakshmi" → {"name": "Lakshmi"}
User: "lakshmi@gmail.com" → {"email": "lakshmi@gmail.com"}
User: "confirm" → CONFIRM
User: "yes that's correct" → CONFIRM
User: "proceed" → CONFIRM
"""

# ---------------------------------------------------
# APPOINTMENT BOOKING DIALOGUE
# ---------------------------------------------------

def _extract_slots(session: dict, user_message: str) -> str:
    """
    Call the LLM to extract slot values from the user message.
    Returns "CONFIRM", "CANCEL", or a JSON string of extracted slots.
    """
    current_slots = session.get("slots", {})
    prompt = f"Current slots: {current_slots}\nUser message: {user_message}"

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SLOT_EXTRACTION_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        result = completion.choices[0].message.content.strip()
        print(f"[SLOT EXTRACT] raw='{result}'")
        return result
    except Exception as e:
        print("Slot extraction ERROR:", e)
        return "{}"


def _parse_extracted_slots(raw: str) -> dict:
    """Parse JSON from slot extraction, stripping markdown fences."""
    import json, re
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}


def generate_reply(session: dict, last_user_message: str) -> str:
    """
    Deterministic slot-filling dialogue manager.
    - Uses LLM only to EXTRACT slot values from user input.
    - Python code decides what to ask next (never re-asks filled slots).
    - Handles confirm/cancel intents explicitly.
    """

    # Initialise session state
    if "history" not in session:
        session["history"] = []
    if "slots" not in session:
        session["slots"] = {}
    if "state" not in session:
        session["state"] = "collecting"  # states: collecting | awaiting_confirm | done

    # Log user message to history
    session["history"].append({"role": "user", "content": last_user_message})

    slots = session["slots"]
    state = session["state"]

    # ------------------------------------------------------------------
    # 0a. Repeat detection — replay last assistant message
    # ------------------------------------------------------------------
    REPEAT_PHRASES = {"repeat", "repeat it", "repeat it again", "say that again",
                      "say again", "pardon", "what", "come again", "once more",
                      "can you repeat", "please repeat", "again",
                      "can you repeat it", "can you repeat it again",
                      "can you repeat that", "can you say that again",
                      "could you repeat", "could you repeat that",
                      "could you repeat it again", "please say that again",
                      "sorry repeat that", "sorry what was that"}
    if last_user_message.strip().lower() in REPEAT_PHRASES:
        history = session.get("history", [])
        # Find last assistant message (skip the user msg we just appended)
        for entry in reversed(history[:-1]):
            if entry["role"] == "assistant":
                reply = entry["content"]
                session["history"].append({"role": "assistant", "content": reply})
                return reply
        # No previous assistant message found — fall through normally

    # ------------------------------------------------------------------
    # 0b. Greeting detection — respond warmly without touching slots
    # ------------------------------------------------------------------
    GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon",
                 "good evening", "howdy", "hiya", "greetings", "sup", "yo"}
    if last_user_message.strip().lower() in GREETINGS:
        reply = "Hello! I'm ALVA, your voice appointment assistant. How can I help you today?"
        session["history"].append({"role": "assistant", "content": reply})
        return reply

    # ------------------------------------------------------------------
    # 0b. Repeat detection — re-send last assistant message
    # ------------------------------------------------------------------
    REPEAT_PHRASES = {
        "repeat", "repeat that", "repeat it", "repeat again", "repeat it again",
        "say that again", "say again", "can you repeat", "can you repeat that",
        "pardon", "what", "what did you say", "come again", "once more",
        "i didn't hear", "i didn't get that", "didn't catch that",
    }
    if last_user_message.strip().lower() in REPEAT_PHRASES:
        # Find last assistant message in history (skip the one we just appended)
        last_reply = None
        for entry in reversed(session["history"][:-1]):
            if entry["role"] == "assistant":
                last_reply = entry["content"]
                break
        if last_reply:
            session["history"].append({"role": "assistant", "content": last_reply})
            return last_reply
        # No previous reply yet — fall through to normal flow

    # ------------------------------------------------------------------
    # 1. Extract intent / slots from user message
    # ------------------------------------------------------------------
    raw = _extract_slots(session, last_user_message)

    is_confirm = raw.strip().upper() == "CONFIRM"
    is_cancel  = raw.strip().upper() == "CANCEL"

    if not is_confirm and not is_cancel:
        new_slots = _parse_extracted_slots(raw)
        # ------------------------------------------------------------------
        # CHANGE 2: Relationship-word guard
        # If the LLM flagged a relationship word (e.g. "friend", "brother"),
        # do NOT store it as name — ask for the real name + email instead.
        # ------------------------------------------------------------------
        if "_for_whom" in new_slots:
            whom = new_slots.pop("_for_whom")
            slots.pop("_for_whom", None)
            session["slots"] = slots
            reply = (
                f"Sure! Could you please tell me your {whom}'s "
                f"full name and email address?"
            )
            session["history"].append({"role": "assistant", "content": reply})
            return reply
        # ------------------------------------------------------------------
        # Merge new/corrected slots
        for k, v in new_slots.items():
            if v:
                slots[k] = v
        session["slots"] = slots
        print(f"[SLOTS NOW] {slots}")

    # ------------------------------------------------------------------
    # 2. State machine
    # ------------------------------------------------------------------

    # --- AWAITING CONFIRMATION ---
    if state == "awaiting_confirm":
        if is_confirm:
            session["state"] = "done"
            reply = (
                f"Your appointment has been booked successfully. "
                f"A confirmation will be sent to {slots.get('email', 'your email')}."
            )
        elif is_cancel:
            session["state"] = "collecting"
            reply = "No problem! Let me know what you'd like to change."
        else:
            # User said something else — maybe correcting a slot
            missing = get_missing_slot_prompt(slots)
            if missing:
                session["state"] = "collecting"
                reply = f"Got it, I've updated that. {missing}"
            else:
                # All slots still filled — re-read back
                reply = build_readback(slots) + "\nPlease say 'confirm' to proceed or let me know what to change."
        session["history"].append({"role": "assistant", "content": reply})
        return reply

    # --- DONE ---
    if state == "done":
        reply = "Your appointment is already booked. Is there anything else I can help you with?"
        session["history"].append({"role": "assistant", "content": reply})
        return reply

    # --- COLLECTING SLOTS ---
    if is_confirm:
        # User said confirm but we haven't read back yet — check if all slots filled
        missing = get_missing_slot_prompt(slots)
        if missing:
            reply = f"I still need a bit more information. {missing}"
        else:
            session["state"] = "awaiting_confirm"
            reply = build_readback(slots)
        session["history"].append({"role": "assistant", "content": reply})
        return reply

    if is_cancel:
        reply = "No problem! Let me know if you'd like to start over or book a different appointment."
        session["history"].append({"role": "assistant", "content": reply})
        return reply

    # Check what's still missing
    missing = get_missing_slot_prompt(slots)

    if missing is None:
        # All slots collected — move to confirmation
        session["state"] = "awaiting_confirm"
        reply = build_readback(slots)
    else:
        # Acknowledge what was just provided (if anything) + ask next missing slot
        just_filled = list(_parse_extracted_slots(raw).keys()) if not is_confirm and not is_cancel else []
        if just_filled:
            ack = f"Got it, I've noted your {just_filled[-1]}. "
        else:
            ack = ""
        reply = ack + missing

    session["history"].append({"role": "assistant", "content": reply})
    return reply


# ---------------------------------------------------
# FEEDBACK DIALOGUE FUNCTION
# ---------------------------------------------------

def feedback(session: dict, user_message: str) -> str:

    # Ensure feedback history exists
    if "feedback_history" not in session:
        session["feedback_history"] = []

    # Save user message
    session["feedback_history"].append({
        "role": "user",
        "content": user_message
    })

    messages = [
        {"role": "system", "content": FEEDBACK_PROMPT}
    ]

    messages.extend(session["feedback_history"])

    try:

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.4
        )

        reply = completion.choices[0].message.content.strip()

    except Exception as e:

        print("Feedback ERROR:", e)

        reply = "Thank you for your feedback."

    session["feedback_history"].append({
        "role": "assistant",
        "content": reply
    })

    return reply

# ---------------------------------------------------
# NO-SHOW PROMPT
# ---------------------------------------------------

NOSHOW_PROMPT = """
You are ALVA, a compassionate voice assistant for a medical clinic.

A patient has missed their appointment. Your job is to:

1. Acknowledge their missed appointment with empathy (no blame).
2. Listen to their reason naturally.
3. Ask if they would like to book a new appointment.
4. If YES - transition warmly into the booking flow.
5. If NO  - wish them well and close the conversation politely.

Rules:
- Keep responses short, warm, and voice-friendly (under 2 sentences).
- Never make the patient feel guilty.
- Be understanding and professional.
- Do not ask booking questions until the patient says yes to rebooking.
"""


# ---------------------------------------------------
# NO-SHOW DIALOGUE FUNCTION
# ---------------------------------------------------

def noshow_dialogue(session: dict, user_message: str) -> str:

    if "noshow_history" not in session:
        session["noshow_history"] = []

    session["noshow_history"].append({
        "role": "user",
        "content": user_message
    })

    messages = [{"role": "system", "content": NOSHOW_PROMPT}]
    messages.extend(session["noshow_history"])

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.4
        )
        reply = completion.choices[0].message.content.strip()

    except Exception as e:
        print("No-show dialogue ERROR:", e)
        reply = "Thank you for letting us know. Would you like to book a new appointment?"

    session["noshow_history"].append({
        "role": "assistant",
        "content": reply
    })

    return reply

# ──────────────────────────────────────────────────────────────────
# TC048: Slot prompts — ask for exactly the missing slot
# ──────────────────────────────────────────────────────────────────

slot_prompts = {
    "service": "What type of service or appointment would you like to book?",
    "date":    "What date would you like for your appointment?",
    "time":    "What time would you prefer?",
    "name":    "May I have your full name please?",
    "email":   "Could you please provide your email address for the confirmation?",
}

REQUIRED_SLOTS = ["service", "date", "time", "name", "email"]


def get_missing_slot_prompt(slots: dict) -> str | None:
    """TC048: Return the prompt for the first missing required slot, or None."""
    for slot in REQUIRED_SLOTS:
        if not slots.get(slot):
            return slot_prompts[slot]
    return None


# ──────────────────────────────────────────────────────────────────
# TC049: Infinite-loop guard
# ──────────────────────────────────────────────────────────────────

MAX_CLARIF_COUNT = 3


def increment_clarif_count(session: dict) -> int:
    """Increment and return the clarification counter."""
    session["_clarif_count"] = session.get("_clarif_count", 0) + 1
    return session["_clarif_count"]


def reset_clarif_count(session: dict):
    session["_clarif_count"] = 0


def clarif_limit_reached(session: dict) -> bool:
    return session.get("_clarif_count", 0) >= MAX_CLARIF_COUNT


# ──────────────────────────────────────────────────────────────────
# TC053: Confirmation readback
# ──────────────────────────────────────────────────────────────────

def build_readback(slots: dict) -> str:
    """TC053: Build a full-detail confirmation readback string."""
    lines = [
        "Let me confirm your appointment details:",
        f"  Name:    {slots.get('name', 'N/A')}",
        f"  Service: {slots.get('service', 'N/A')}",
        f"  Date:    {slots.get('date', 'N/A')}",
        f"  Time:    {slots.get('time', 'N/A')}",
        f"  Email:   {slots.get('email', 'N/A')}",
        "Is everything correct?",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Timeout tiers — 2 reprompts then escalate
# ──────────────────────────────────────────────────────────────────

SILENCE_REPROMPT_1 = (
    "I'm sorry, I didn't hear you. Could you please repeat that?"
)
SILENCE_REPROMPT_2 = (
    "I still can't hear you. Please speak clearly or press any key."
)
SILENCE_ESCALATE = (
    "I'm having trouble reaching you. Let me connect you to a human agent."
)


def get_silence_reply(count: int) -> tuple[str, bool]:
    """
    TC054: Return (reply_text, should_escalate) based on silence count.
    count=1 → first reprompt
    count=2 → second reprompt
    count>=3 → escalate
    """
    if count == 1:
        return SILENCE_REPROMPT_1, False
    elif count == 2:
        return SILENCE_REPROMPT_2, False
    else:
        return SILENCE_ESCALATE, True