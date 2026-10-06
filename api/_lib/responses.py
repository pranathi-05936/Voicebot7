"""Response generation: maps a predicted intent (plus the user's text) to a reply.

Some intents are answered for real (time, date, maths, jokes, coin/dice, ...);
the rest are domain intents (banking, travel, car, work, home) that a production
assistant would hand off to an external service, so the bot confirms what it
understood and explains the action it would take.
"""
import ast
import datetime as dt
import operator
import random
import re

JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "I told my neural network a joke. It didn't get it — not enough training data.",
    "Why was the computer cold? It left its Windows open.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
    "Why did the gradient break up with the loss function? It felt it was always going downhill.",
]
FACTS = [
    "Honey never spoils — edible honey has been found in 3,000-year-old Egyptian tombs.",
    "Octopuses have three hearts and blue blood.",
    "A day on Venus is longer than a year on Venus.",
    "The first computer 'bug' was an actual moth found in a Harvard relay in 1947.",
    "Bananas are berries, but strawberries are not.",
]

# ------------------------------------------------------------- safe maths
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod}
_WORDS = [(r"\bplus\b|\badded to\b", "+"), (r"\bminus\b|\bless\b", "-"),
          (r"\btimes\b|\bmultiplied by\b|\bx\b", "*"), (r"\bdivided by\b|\bover\b", "/"),
          (r"\bto the power of\b", "**"), (r"\bsquared\b", "**2"), (r"\bmod\b", "%")]
_NUMS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_NUMS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
              "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100"})


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError


def calculate(text):
    s = text.lower().replace(",", "")
    s = re.sub(r"square root of (\d+(?:\.\d+)?)", r"(\1**0.5)", s)
    for w, n in _NUMS.items():
        s = re.sub(rf"\b{w}\b", n, s)
    for pat, sym in _WORDS:
        s = re.sub(pat, sym, s)
    m = re.findall(r"[\d\.\s\+\-\*/%\(\)]+", s)
    expr = max((x.strip() for x in m), key=len, default="")
    if not re.search(r"\d", expr) or not re.search(r"[\+\-\*/%]", expr):
        return None
    try:
        val = _eval(ast.parse(expr, mode="eval").body)
        val = round(val, 6)
        val = int(val) if float(val).is_integer() else val
        return str(val) if "**" in expr else f"{expr} = {val}"
    except Exception:
        return None


def _now(offset_h):
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=offset_h)


def _flip(_):
    return f"I flipped a coin — it's {random.choice(['heads', 'tails'])}!"


def _dice(_):
    return f"You rolled a {random.randint(1, 6)}."


def _calc(text):
    r = calculate(text)
    return f"The answer is {r}." if r else "Tell me the calculation, e.g. 'what is 12 times 7'."


def _spell(text):
    m = re.search(r"spell (?:the word )?([a-zA-Z]+)", text, re.I) or \
        re.search(r"([a-zA-Z]+)\W*$", text)
    w = m.group(1) if m else ""
    return f"{w} is spelled {'-'.join(w.upper())}." if w else "Which word should I spell?"


DYNAMIC = {
    "time": lambda t: "It's " + _now(0).strftime('%I:%M %p').lstrip('0') + " UTC ("
                      + _now(5.5).strftime('%I:%M %p').lstrip('0') + " IST).",
    "date": lambda t: f"Today is {_now(5.5).strftime('%A, %d %B %Y')} (IST).",
    "flip_coin": _flip,
    "roll_dice": _dice,
    "tell_joke": lambda t: random.choice(JOKES),
    "fun_fact": lambda t: "Fun fact: " + random.choice(FACTS),
    "calculator": _calc,
    "spelling": _spell,
}

STATIC = {
    # small talk / about the bot
    "greeting": "Hello! I'm VoiceBot. Ask me something — tap the mic and speak.",
    "goodbye": "Goodbye! Have a great day.",
    "thank_you": "You're welcome — happy to help!",
    "yes": "Great, noted.",
    "no": "Okay, no problem.",
    "maybe": "No rush — let me know when you've decided.",
    "repeat": "Sure — scroll up to see my last reply, or ask the question again.",
    "what_is_your_name": "I'm VoiceBot, a voice-enabled deep-learning assistant.",
    "who_made_you": "I was built as a student project: Whisper recognises your speech in the browser, and a BiLSTM-attention network classifies your intent.",
    "who_do_you_work_for": "I work for whoever is talking to me — right now, that's you.",
    "are_you_a_bot": "Yes, I'm a chatbot powered by deep learning.",
    "how_old_are_you": "I was trained quite recently, so I'm very young!",
    "where_are_you_from": "I live in the cloud — my brain runs on a Vercel serverless function.",
    "what_are_your_hobbies": "I enjoy classifying intents and listening to people talk.",
    "do_you_have_pets": "No pets — although I'd love a robot dog.",
    "meaning_of_life": "Some say 42. I say: learning something new every day.",
    "what_can_i_ask_you": "I recognise 150 kinds of requests across banking, travel, cooking, cars, work, home and small talk. Try 'tell me a joke', 'what is 15 times 4', 'what time is it' or 'I lost my credit card'.",
    "user_name": "I don't store names in this demo, so I don't know your name yet.",
    "change_user_name": "Okay — I'd update the name I call you (this demo keeps no profile).",
    "change_ai_name": "You'd like to rename me. I'm fond of VoiceBot, but I'll note it.",
    "whisper_mode": "Switching to whisper mode… (voice output volume would be lowered).",
    "change_volume": "Adjusting the volume setting.",
    "change_speed": "Adjusting my speaking speed.",
    "change_accent": "Changing my voice accent setting.",
    "change_language": "Changing the language setting.",
    "reset_settings": "Resetting all settings to their defaults.",
    "sync_device": "Syncing with your device.",
    "cancel": "Okay, cancelled.",
}

# Domain intents: (what the bot understood, what it would do)
DOMAIN = {
    # banking
    "balance": "check your account balance", "transfer": "transfer money between accounts",
    "transactions": "see recent transactions", "spending_history": "review your spending history",
    "freeze_account": "freeze your account", "account_blocked": "fix a blocked account",
    "report_fraud": "report fraudulent activity", "report_lost_card": "report a lost card",
    "damaged_card": "replace a damaged card", "new_card": "request a new card",
    "replacement_card_duration": "know how long a replacement card takes",
    "card_declined": "find out why your card was declined", "pin_change": "change your PIN",
    "credit_limit": "check your credit limit", "credit_limit_change": "change your credit limit",
    "credit_score": "check your credit score", "improve_credit_score": "improve your credit score",
    "apr": "check your card's APR", "interest_rate": "check the interest rate",
    "min_payment": "know your minimum payment", "pay_bill": "pay a bill",
    "bill_balance": "check a bill balance", "bill_due": "know when a bill is due",
    "expiration_date": "check your card's expiry date", "routing": "get your routing number",
    "direct_deposit": "set up direct deposit", "order_checks": "order new cheques",
    "rewards_balance": "check your rewards balance", "redeem_rewards": "redeem rewards",
    "international_fees": "check international transaction fees", "exchange_rate": "check an exchange rate",
    "income": "check your income", "taxes": "get help with taxes", "w2": "get your W-2 form",
    "rollover_401k": "roll over a 401k", "insurance": "check your insurance",
    "insurance_change": "change your insurance plan", "payday": "know when payday is",
    "application_status": "check an application status",
    # travel
    "book_flight": "book a flight", "book_hotel": "book a hotel", "flight_status": "check a flight status",
    "car_rental": "rent a car", "carry_on": "check carry-on rules", "lost_luggage": "report lost luggage",
    "travel_alert": "check travel alerts", "travel_notification": "set a travel notification on your card",
    "travel_suggestion": "get travel suggestions", "international_visa": "check visa requirements",
    "vaccines": "check vaccine requirements for travel", "plug_type": "know the plug type abroad",
    "timezone": "check a time zone", "translate": "translate something",
    "directions": "get directions", "distance": "know a distance", "traffic": "check traffic",
    "uber": "book a ride", "current_location": "know your current location",
    "share_location": "share your location", "weather": "check the weather",
    # food
    "recipe": "find a recipe", "ingredients_list": "get an ingredients list",
    "ingredient_substitution": "substitute an ingredient", "cook_time": "know a cooking time",
    "calories": "check calories", "nutrition_info": "get nutrition information",
    "food_last": "know how long food lasts", "meal_suggestion": "get a meal suggestion",
    "restaurant_suggestion": "find a restaurant", "restaurant_reviews": "read restaurant reviews",
    "restaurant_reservation": "reserve a table", "accept_reservations": "know if a restaurant takes reservations",
    "confirm_reservation": "confirm a reservation", "cancel_reservation": "cancel a reservation",
    "how_busy": "know how busy a place is", "order": "place an order", "order_status": "check an order status",
    "measurement_conversion": "convert a measurement", "definition": "look up a definition",
    # car
    "gas": "check your fuel level", "gas_type": "know which fuel your car uses", "mpg": "check fuel efficiency",
    "oil_change_how": "learn how to change oil", "oil_change_when": "know when to change oil",
    "tire_change": "change a tyre", "tire_pressure": "check tyre pressure", "jump_start": "jump-start a car",
    "last_maintenance": "see your last maintenance", "schedule_maintenance": "schedule car maintenance",
    # work / productivity
    "pto_balance": "check your leave balance", "pto_request": "request leave",
    "pto_request_status": "check a leave request", "pto_used": "see leave used",
    "meeting_schedule": "check your meetings", "schedule_meeting": "schedule a meeting",
    "calendar": "check your calendar", "calendar_update": "update your calendar",
    "reminder": "check your reminders", "reminder_update": "set a reminder",
    "todo_list": "see your to-do list", "todo_list_update": "update your to-do list",
    "shopping_list": "see your shopping list", "shopping_list_update": "update your shopping list",
    "alarm": "set an alarm", "timer": "set a timer", "next_holiday": "know the next holiday",
    # home / media / phone
    "smart_home": "control a smart-home device", "play_music": "play music", "next_song": "skip to the next song",
    "what_song": "identify the current song", "update_playlist": "update a playlist",
    "make_call": "make a phone call", "text": "send a text message", "find_phone": "find your phone",
}


def respond(intent, text):
    if intent == "oos":
        return ("Sorry, that's outside what I can help with. Try asking about banking, "
                "travel, cooking, your car, your schedule — or ask for a joke.")
    if intent in DYNAMIC:
        return DYNAMIC[intent](text)
    if intent in STATIC:
        return STATIC[intent]
    if DOMAIN.get(intent):
        return (f"Sure — I can help you {DOMAIN[intent]}. (Demo mode: I recognise the "
                f"request, but I'm not connected to a live service to complete it.)")
    return "I understood your request, but I don't have a response for it yet."
