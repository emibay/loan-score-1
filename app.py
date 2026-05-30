import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load all artifacts from the pickle bundle
_bundle = joblib.load("loan_scoring_model.pkl")
model        = _bundle["model"]
scaler       = _bundle["scaler"]
le_employ    = _bundle["label_encoder_employment"]
le_decision  = _bundle["label_encoder_decision"]

# Employment types available in the encoder (sorted by encoder order)
EMPLOYMENT_OPTIONS = le_employ.classes_.tolist()

# Derived-feature computation --------------------------------------------------

def build_features(monthly_income: float, employment_type: str,
                   employment_years: float, requested_amount: float) -> np.ndarray:
    emp_encoded = int(le_employ.transform([employment_type])[0])
    ratio       = requested_amount / monthly_income
    annual_dti  = requested_amount / (monthly_income * 12)
    log_income  = np.log(monthly_income)
    log_amount  = np.log(requested_amount)
    return np.array([[
        monthly_income,
        employment_years,
        requested_amount,
        ratio,
        annual_dti,
        log_income,
        log_amount,
        emp_encoded,
    ]])


def compute_score(features: np.ndarray) -> tuple[int, str]:
    scaled  = scaler.transform(features)
    proba   = model.predict_proba(scaled)[0]   # [P(approved), P(manual), P(rejected)]
    # Score = probability of approval × 1000, clamped to [0, 1000]
    score   = int(round(np.clip(proba[0] * 1000, 0, 1000)))
    return score


def get_decision(score: int) -> dict:
    if score >= 700:
        return {
            "label":   "Зөвшөөрөх",
            "css":     "approved",
            "icon":    "✓",
            "message": "Таны зээлийн хүсэлт зөвшөөрөгдлөө. Менежер тантай удахгүй холбогдох болно.",
        }
    elif score >= 450:
        return {
            "label":   "Гар шалгалт",
            "css":     "review",
            "icon":    "⟳",
            "message": "Таны хүсэлтийг мэргэжилтэн нэмэлт шалгалт хийнэ. 2–3 ажлын өдөрт хариу өгнө.",
        }
    else:
        return {
            "label":   "Татгалзах",
            "css":     "declined",
            "icon":    "✕",
            "message": "Таны зээлийн хүсэлт одоогоор татгалзагдлаа. Цалин болон ажлын туршлагаа нэмэгдүүлсний дараа дахин хандана уу.",
        }


# Routes -----------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", employment_options=EMPLOYMENT_OPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    # Collect raw form values first so they can be echoed back on any error
    raw = {
        "monthly_income":   request.form.get("monthly_income", ""),
        "employment_type":  request.form.get("employment_type", ""),
        "employment_years": request.form.get("employment_years", ""),
        "requested_amount": request.form.get("requested_amount", ""),
    }

    def render_error(msg):
        return render_template("index.html",
                               employment_options=EMPLOYMENT_OPTIONS,
                               error=msg, **raw)

    try:
        monthly_income   = float(raw["monthly_income"])
        employment_type  = raw["employment_type"]
        employment_years = float(raw["employment_years"])
        requested_amount = float(raw["requested_amount"])

        if monthly_income <= 0 or requested_amount <= 0:
            return render_error("Цалин болон зээлийн дүн 0-ээс их байх ёстой.")
        if employment_years < 0:
            return render_error("Ажилласан жил 0-ээс бага байж болохгүй.")
        if employment_type not in EMPLOYMENT_OPTIONS:
            return render_error("Ажил эрхлэлтийн төрлийг сонгоно уу.")

        features = build_features(monthly_income, employment_type,
                                  employment_years, requested_amount)
        score    = compute_score(features)
        decision = get_decision(score)

        return render_template(
            "index.html",
            employment_options=EMPLOYMENT_OPTIONS,
            score=score,
            decision=decision,
            monthly_income=monthly_income,
            employment_type=employment_type,
            employment_years=employment_years,
            requested_amount=requested_amount,
        )

    except ValueError:
        return render_error("Тоон утгуудыг зөв оруулна уу.")
    except Exception as e:
        return render_error(f"Системийн алдаа: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
