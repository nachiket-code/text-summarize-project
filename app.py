from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path
import torch
import re

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


# =========================================================
# 1. CREATE FASTAPI APP
# =========================================================

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)


# =========================================================
# 2. LOAD MODEL AND TOKENIZER
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

model = T5ForConditionalGeneration.from_pretrained(
    BASE_DIR / "saved_summary_model"
)

tokenizer = T5Tokenizer.from_pretrained(
    BASE_DIR / "saved_summary_model"
)


# =========================================================
# 3. SELECT DEVICE
# =========================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("Using device:", device)


# Move model to selected device
model.to(device)

# We are doing prediction, not training
model.eval()


# =========================================================
# 4. HTML TEMPLATE LOCATION
# =========================================================

templates = Jinja2Templates(directory=str(BASE_DIR))


# =========================================================
# 5. INPUT DATA MODEL
# =========================================================

class DialogueInput(BaseModel):

    dialogue: str


# =========================================================
# 6. CLEAN DATA
# =========================================================

def clean_data(text):

    text = re.sub(r"\r\n", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = text.strip().lower()

    return text


# =========================================================
# 7. SUMMARIZATION FUNCTION
# =========================================================

def summarize_dialogue(dialogue: str) -> str:

    # -----------------------------
    # Clean dialogue
    # -----------------------------

    dialogue = clean_data(dialogue)


    # -----------------------------
    # Tokenization
    # -----------------------------

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )


    # Move input tensors to GPU/CPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    # -----------------------------
    # Generate summary
    # -----------------------------

    with torch.no_grad():

        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )


    # -----------------------------
    # Decode output
    # -----------------------------

    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )


    return summary


# =========================================================
# 8. API ENDPOINT
# =========================================================

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(
        dialogue_input.dialogue
    )

    return {
        "summary": summary
    }


# =========================================================
# 9. HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )