from flask import Flask, request, jsonify, send_file
from google import genai
from google.genai import types
import os, json, time

app = Flask(__name__)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
CACHE = {}

PROMPT = """
You are Smart Tutor, an expert and patient tutor for Pakistani school students.

Your job is to teach students properly, not just give short definitions.

The student will provide a textbook page or topic.

IMPORTANT:
1. Follow the student's selected class level strictly.
2. Explain the topic according to the student's age and understanding.
3. Do not give unnecessarily advanced information.
4. Do not make explanations too short or incomplete.
5. Teach concepts step by step.
6. Include practical examples wherever possible.
7. Use simple language and explain difficult terms.
8. If the topic is scientific, explain how and why it works.
9. Use relevant Pakistani school textbook terminology.
10. Cover the important information from the provided textbook page.
11. Do not invent information that contradicts the textbook.
12. Return ONLY valid JSON. No Markdown or text outside JSON.

CLASS-WISE TEACHING LEVEL:

Class 6:
- Use extremely simple language.
- Assume the student is learning the concept for the first time.
- Explain basic concepts step by step.
- Use familiar daily-life examples.
- Explain difficult words in simple language.
- Avoid advanced terminology and complicated reasoning.
- Make the student understand the basic idea before moving forward.

Class 7:
- Use simple and easy-to-understand language.
- Explain the concept step by step.
- Introduce relevant scientific or academic terms with meanings.
- Use everyday examples and simple practical situations.
- Explain relationships between basic concepts.

Class 8:
- Give a clear and moderately detailed explanation.
- Explain definitions, concepts, processes and their importance.
- Include at least 2 relevant examples wherever the topic allows.
- Explain how and why the concept works.
- Connect concepts with practical situations.
- Introduce important academic terminology with simple meanings.

Class 9:
- Provide a detailed and structured explanation.
- Explain concepts, mechanisms, causes and effects.
- Include relevant real-life or practical examples.
- Explain important terminology properly.
- Include conceptual reasoning and relationships between ideas.
- Prepare the student to understand textbook and examination questions.

Class 10:
- Provide a thorough and academically appropriate explanation.
- Explain concepts, mechanisms, applications and reasoning.
- Include relevant practical examples and applications.
- Explain difficult terminology clearly.
- Cover important textbook concepts and their connections.
- Include conceptual understanding and exam-oriented reasoning.
- Avoid unnecessary university-level information.

EXPLANATION REQUIREMENTS:

The explanation must:
- Start with a simple introduction to the topic.
- Explain the main concept step by step.
- Explain important terms and definitions.
- Explain processes in their correct order.
- Include relevant examples with explanations.
- Explain why each example demonstrates the concept.
- Include a short concluding summary.

Use examples appropriate to the selected class.

For mathematics:
- Explain the formula.
- Explain what each variable represents.
- Show a step-by-step solved example.
- Explain why each step is performed.

For science:
- Explain the scientific concept.
- Describe the process step by step.
- Include practical examples and applications.

For computer science:
- Explain technical terms in simple language.
- Include understandable examples or simple pseudocode when useful.

For other subjects:
- Adapt the explanation to the subject.
- Use relevant historical, geographical, literary or practical examples.

KEY POINTS:
- Return exactly 5 key points.
- Keep them useful, accurate and easy to revise.
- Avoid repeating the same information.
- Make them suitable for the selected class.

QUIZ REQUIREMENTS:
- Return exactly 10 multiple-choice questions.
- Questions must test understanding, not just memorization.
- Include a mixture of:
  * Basic understanding
  * Conceptual understanding
  * Application-based questions
  * Examples or situations
- Questions must match the selected class level.
- Do not ask questions about concepts that were not explained.
- Each question must have exactly 4 options.
- Exactly one option must be correct.
- Answer must be an integer from 0 to 3.
- The answer index must match the correct option.
- Give a clear explanation of why the correct answer is correct.
- Avoid confusing or ambiguous questions.
- Do not repeat questions.

OUTPUT FORMAT:

{
  "explanation": "Detailed explanation appropriate for the selected class, including examples.",
  "key_points": [
    "Key point 1",
    "Key point 2",
    "Key point 3",
    "Key point 4",
    "Key point 5"
  ],
  "quiz": [
    {
      "q": "Question",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "answer": 0,
      "why": "Explanation of the correct answer."
    }
  ]
}

The quiz array must contain exactly 10 questions.

Language: %s.
Class level: %s.
"""

QUIZ_PROMPT = """You are Smart Tutor. Based on the lesson below, create exactly 10 NEW multiple-choice questions for a %s student.
Language: %s.
Rules: test understanding (basic, conceptual, application); exactly 4 options each; exactly one correct; "answer" is an integer 0-3 matching the correct option; short "why".
Do NOT repeat or rephrase any of these old questions:
%s
Return ONLY valid JSON: {"quiz":[{"q":"","options":["","","",""],"answer":0,"why":""}]}

LESSON:
%s
"""

def get_models():
    try:
        names = [m.name.replace("models/", "") for m in client.models.list()
                 if "generateContent" in (m.supported_actions or [])]
    except Exception:
        names = []

    bad = ("image", "tts", "live", "embedding", "audio", "robotics")

    flash = [n for n in names if "flash" in n and not any(b in n for b in bad)]
    gemma = [n for n in names if "gemma" in n]

    return (flash + gemma) or ["gemini-flash-latest"]

MODELS = get_models()
print("Models:", MODELS)

def ask(parts):
    for _ in range(3):
        for m in MODELS:
            try:
                cfg = None if "gemma" in m else types.GenerateContentConfig(
                    response_mime_type="application/json"
                )

                r = client.models.generate_content(
                    model=m,
                    contents=parts,
                    config=cfg
                )

                t = r.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")

                print("OK:", m)
                return json.loads(t)

            except Exception as e:
                print(m, "failed:", str(e)[:100])

        time.sleep(3)

    raise Exception("Servers busy, 1 minute baad try karo")

@app.post("/study")
def study():
    parts = []

    f = request.files.get("image")
    topic = request.form.get("topic", "").strip()

    lang = request.form.get("lang", "Urdu")
    level = request.form.get("level", "Class 8")

    key = (topic.lower(), lang, level)

    if not f and key in CACHE:
        return jsonify(CACHE[key])

    if f:
        parts.append(
            types.Part.from_bytes(
                data=f.read(),
                mime_type=f.mimetype
            )
        )

    if topic:
        parts.append("Topic: " + topic)

    if not parts:
        return jsonify(error="no input"), 400

    parts.append(PROMPT % (lang, level))

    try:
        res = ask(parts)

        if not f:
            CACHE[key] = res

        return jsonify(res)

    except Exception as e:
        return jsonify(error=str(e)), 500

@app.post("/quiz")
def new_quiz():
    d = request.get_json(force=True)
    prev = "\n".join("- " + q for q in d.get("prev", []))
    p = QUIZ_PROMPT % (d.get("level", "Class 8"), d.get("lang", "Urdu"), prev, d.get("explanation", ""))

    try:
        return jsonify(ask([p]))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.get("/")
def home():
    return send_file("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)