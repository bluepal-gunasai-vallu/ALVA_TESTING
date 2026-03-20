from groq import Groq
import json
import os
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are an advanced NLU engine for a voice appointment assistant.

Your task:
1. Detect the user's intent.
2. Extract relevant structured entities from spoken or typed input.
3. Handle partial, ambiguous, conversational, or corrected inputs robustly.
4. Never hallucinate missing data.
5. If uncertain, return null for that field.

INTENTS (choose exactly one):
- schedule
- reschedule
- cancel
- confirm
- check_availability
- greeting
- unknown
- feedback
- human_help

ENTITIES TO EXTRACT:

date:
  Accept ANY natural-language date expression the user might say or type.
  This includes relative references ("tomorrow", "in 3 days", "next week"),
  named weekdays ("next Tuesday", "this Friday"), ordinal phrasing
  ("the fourteenth of March", "June 3rd"), standard formats ("March 5",
  "2026-03-18"), and informal wording ("a week from now", "end of the month").
  Return the date as the user expressed it; downstream code handles normalization.

time:
  Accept ANY natural-language time expression.
  This includes clock format ("3 PM", "15:00"), spoken-word ("two thirty PM",
  "fourteen thirty", "eleven o'clock"), colloquial ("half past two",
  "quarter to five", "noon", "midnight"), and contextual ("3 in the afternoon",
  "8 in the morning").
  Return a normalized time string when possible (e.g. "3:00 PM", "14:30").
  If ambiguous between AM/PM, infer from context or return as-is.

time_period:
  One of: morning, afternoon, evening, night.
  Only extract when the user gives a general period instead of a specific time.

service:
  The type of appointment or service requested.

name:
  The user's name if mentioned.

email:
  The user's email address if mentioned.
  Normalize spacing (e.g. "user @ gmail.com" becomes "user@gmail.com").

SELF-CORRECTION HANDLING:
Users frequently correct themselves mid-sentence.
- ALWAYS extract the FINAL / CORRECTED value for any entity and DISCARD earlier mentions.
- Corrections may be signaled explicitly (words like "actually", "sorry", "I mean", "wait",
  "change that", "make it", "not that") or implicitly by simply stating a new value for the
  same entity after an earlier one.
- General principle: when the same entity appears more than once, the LAST occurrence wins.

RULES:
- Return ONLY valid JSON. No explanations, no markdown, no extra text.
- If a value is missing or unclear, return null for that field.
- If input is unclear but suggests booking context, choose the most logical intent.
- Never fabricate information.
- Always return all keys in the output.

CONFIRM INTENT:
Classify as "confirm" when the user expresses agreement, approval, or readiness to proceed
with a booking, even without the word "confirm". Any affirmative response that signals
approval, acceptance, or permission to continue counts.

This includes acknowledgements (short or long) when the user is responding to a confirmation
question like "Is that correct?" or "Should I book it?".

Examples that MUST be classified as "confirm" in booking context:
- proceed / go ahead / continue / do it / book it
- yep / yeah / yes / ok / okay / sure / alright
- that’s right / correct / sounds good / looks good

HUMAN_HELP INTENT:
Classify as "human_help" when the user wants to be connected to a real person, whether a
doctor, agent, operator, receptionist, or any human. This covers any phrasing that requests
live human assistance, asks to be transferred, or expresses a desire to stop talking to the
automated system.

JSON FORMAT (strict):

{
  "intent": "",
  "date": "",
  "time": "",
  "time_period": "",
  "service": "",
  "name": "",
  "email": ""
}
"""

# ---------- RELATIVE DATE PARSER ----------
def normalize_relative_date(date_text):

    if not date_text:
        return None

    date_text = date_text.lower().strip()
    today = datetime.today()

    weekdays = {
        "monday":0,
        "tuesday":1,
        "wednesday":2,
        "thursday":3,
        "friday":4,
        "saturday":5,
        "sunday":6
    }

    # tomorrow / next day
    if date_text in ["tomorrow", "next day"]:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")

    # day after tomorrow
    if date_text == "day after tomorrow":
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")

    # next 3 days
    if "next 3 days" in date_text:
        return (today + timedelta(days=3)).strftime("%Y-%m-%d")

    # next week
    if date_text == "next week":
        return (today + timedelta(days=7)).strftime("%Y-%m-%d")

    # next month
    if date_text == "next month":
        month = today.month + 1 if today.month < 12 else 1
        year = today.year if today.month < 12 else today.year + 1
        return datetime(year, month, today.day).strftime("%Y-%m-%d")

    # next weekend (Saturday)
    if "next weekend" in date_text:
        saturday = 5
        days_ahead = saturday - today.weekday()

        if days_ahead <= 0:
            days_ahead += 7

        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # next weekday (next friday etc)
    for day in weekdays:
        if f"next {day}" in date_text:

            target = weekdays[day]
            days_ahead = target - today.weekday()

            if days_ahead <= 0:
                days_ahead += 7

            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # this weekday
    for day in weekdays:
        if f"this {day}" in date_text:

            target = weekdays[day]
            days_ahead = target - today.weekday()

            if days_ahead < 0:
                days_ahead += 7

            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    return date_text

_WORD_TO_NUM = {
    "zero": 0, "oh": 0, "o": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50,
}


def _words_to_int(tokens: list) -> int | None:
    """Convert a list of word tokens to an integer (for hour/minute parsing)."""
    total = 0
    for t in tokens:
        v = _WORD_TO_NUM.get(t)
        if v is None:
            return None
        total += v
    return total


def detect_time_regex(text: str):
    """Regex + word-number fallback for time extraction.

    Handles:
      - Colloquial: "half past two", "quarter to five", "quarter past three"
      - Spoken digit: "two thirty PM", "three fifteen", "ten o'clock"
      - Numeric patterns: "3 PM", "14:30", "3:00"
    When multiple matches appear, the last one wins (handles self-corrections).
    """
    t = text.lower()

    # ── 1. Colloquial relative expressions ──────────────────────────
    half_past = re.search(r'half\s+past\s+(\w+)', t)
    if half_past:
        raw = half_past.group(1)
        h = int(raw) if raw.isdigit() else _WORD_TO_NUM.get(raw)
        if h is not None:
            return f"{h:02d}:30"

    quarter_past = re.search(r'quarter\s+past\s+(\w+)', t)
    if quarter_past:
        raw = quarter_past.group(1)
        h = int(raw) if raw.isdigit() else _WORD_TO_NUM.get(raw)
        if h is not None:
            return f"{h:02d}:15"

    quarter_to = re.search(r'quarter\s+to\s+(\w+)', t)
    if quarter_to:
        raw = quarter_to.group(1)
        h = int(raw) if raw.isdigit() else _WORD_TO_NUM.get(raw)
        if h is not None:
            h = (h - 1) % 24
            return f"{h:02d}:45"

    # ── 2. Spoken-word time: "<hour_word> <minute_word(s)> [am|pm]" ─
    # e.g. "two thirty PM", "three fifteen", "ten o'clock"
    am_pm_suffix = None
    am_pm_m = re.search(r'\b(am|pm)\b', t)
    if am_pm_m:
        am_pm_suffix = am_pm_m.group(1)

    word_time = re.search(
        r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)'
        r'(?:\s+(oh|o|zero|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen'
        r'|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty))?'
        r'(?:\s+(oh|zero|one|two|three|four|five|six|seven|eight|nine))?'
        r"(?:\s+o'?clock)?\b",
        t
    )
    if word_time:
        h_word = word_time.group(1)
        m_word1 = word_time.group(2)
        m_word2 = word_time.group(3)
        hour = _WORD_TO_NUM.get(h_word, 0)
        minute = 0
        if m_word1:
            minute = _WORD_TO_NUM.get(m_word1, 0)
        if m_word2:
            minute += _WORD_TO_NUM.get(m_word2, 0)
        if am_pm_suffix == "pm" and hour != 12:
            hour += 12
        elif am_pm_suffix == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    # ── 3. Numeric patterns — take the LAST match ────────────────────
    pattern = r'(\d{1,2})(:\d{2})?\s*(am|pm|o\'?clock)?'
    matches = re.findall(pattern, t)
    if matches:
        hour_str, minute, suffix = matches[-1]
        hour = int(hour_str)
        minute = minute if minute else ":00"
        if suffix and "pm" in suffix and hour != 12:
            hour += 12
        if suffix and "am" in suffix and hour == 12:
            hour = 0
        return f"{hour:02d}{minute}"

    return None


def extract_nlu(text: str) -> dict:
    regex_time = detect_time_regex(text)
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_completion_tokens=300,
            top_p=1,
            stream=False
        )

        response_text = completion.choices[0].message.content.strip()

        # Parse JSON safely
        parsed = json.loads(response_text)

        # Ensure all required keys exist
        required_keys = [
            "intent",
            "date",
            "time",
            "time_period",
            "service",
            "name",
            "email"
        ]

        for key in required_keys:
            if key not in parsed:
                parsed[key] = None

        # ---------- FIX 1: REGEX TIME ----------
        # Always prefer regex_time when LLM returns None, "None" (string),
        # empty string, or any value that doesn't look like a valid time.
        _lm_time = parsed.get("time")
        _lm_time_invalid = (
            _lm_time is None
            or str(_lm_time).strip().lower() in ("none", "", "null")
        )
        if _lm_time_invalid and regex_time:
            parsed["time"] = regex_time
        elif regex_time and not _lm_time_invalid:
            # Also override if LLM returned a vague period but regex found exact time
            pass  # keep LLM value when it looks valid

        # ---------- FIX 2: TIME PERIOD ----------
        if parsed["time"] is None and parsed.get("time_period"):

            period = parsed["time_period"].lower()

            if period == "morning":
                parsed["time"] = "09:00"
            elif period == "afternoon":
                parsed["time"] = "14:00"
            elif period == "evening":
                parsed["time"] = "17:00"
            elif period == "night":
                parsed["time"] = "19:00"

        # ---------- FIX 3: RELATIVE DATE ----------
        if parsed["date"]:
            parsed["date"] = normalize_relative_date(parsed["date"])

        return parsed
    


    except Exception as e:
        print("NLU ERROR:", e)

        # Safe fallback
        return {
            "intent": "unknown",
            "date": None,
            "time": None,
            "time_period": None,
            "service": None,
            "name": None,
            "email": None
        }