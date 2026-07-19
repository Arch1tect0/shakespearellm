"""Generate a side-by-side qualitative comparison with Gemini Flash."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import google.generativeai as genai
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "my-transformer"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_PATH = Path(__file__).resolve().parent / "comparison_results.md"
sys.path.insert(0, str(MODEL_DIR))

from model import GPT  # noqa: E402
from tokenizer import ByteTokenizer  # noqa: E402

PROMPTS = [
    "To be, or not to ",
    "O Romeo, Romeo, ",
    "Now is the winter of ",
    "Friends, Romans, countrymen, ",
]
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_K = 40


def load_model(checkpoint_name: str) -> GPT:
    """Load a trained GPT checkpoint onto the CPU."""
    checkpoint = torch.load(
        CHECKPOINT_DIR / checkpoint_name,
        map_location="cpu",
        weights_only=False,
    )
    model = GPT(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate_local(model: GPT, tokenizer: ByteTokenizer, prompt: str) -> str:
    """Generate exactly MAX_NEW_TOKENS byte tokens and return the continuation."""
    prompt_tokens = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long)
    generated = model.generate(
        idx,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )
    continuation = generated[0, len(prompt_tokens) :].tolist()
    return tokenizer.decode(continuation)


def select_flash_model() -> str:
    """Choose the newest preferred Flash model exposed to this API key."""
    available = {
        model.name
        for model in genai.list_models()
        if "generateContent" in model.supported_generation_methods
    }
    preferred = [
        "models/gemini-flash-latest",
        "models/gemini-3.5-flash",
        "models/gemini-3-flash-preview",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
    ]
    for model_name in preferred:
        if model_name in available:
            return model_name

    flash_models = sorted(name for name in available if "flash" in name.lower())
    if not flash_models:
        raise RuntimeError("No Gemini Flash model is available for this API key.")
    return flash_models[-1]


def generate_gemini(model: genai.GenerativeModel, prompt: str) -> str:
    """Ask Gemini for a continuation capped to about 150 characters."""
    request = (
        "Continue the Shakespearean text below. Return only the continuation, "
        "with no explanation, title, or quotation marks. Keep the continuation "
        f"to about {MAX_NEW_TOKENS} characters.\n\n{prompt}"
    )
    response = model.generate_content(
        request,
        generation_config=genai.GenerationConfig(
            temperature=TEMPERATURE,
            # Newer Flash models may spend part of this budget on internal
            # reasoning; the returned text is still truncated to 150 chars.
            max_output_tokens=512,
        ),
    )
    # Character truncation keeps the visual comparison close to the 150
    # byte-token budget used by the local models.
    return response.text.strip()[:MAX_NEW_TOKENS]


def markdown_cell(text: str) -> str:
    """Escape generated text for a readable one-row-per-prompt Markdown table."""
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def write_results(rows: list[dict[str, str]], gemini_model: str) -> None:
    lines = [
        "# Three-model Shakespeare comparison",
        "",
        (
            f"Local models generated {MAX_NEW_TOKENS} byte tokens with "
            f"temperature {TEMPERATURE} and top-k {TOP_K}. "
            f"Gemini used `{gemini_model}` and was capped to roughly "
            f"{MAX_NEW_TOKENS} characters."
        ),
        "",
        "| Prompt | Model A output | Model B output | Gemini output |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(row[column])
                for column in ("prompt", "model_a", "model_b", "gemini")
            )
            + " |"
        )
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from the project .env file.")

    genai.configure(api_key=api_key)
    gemini_model_name = select_flash_model()
    gemini = genai.GenerativeModel(gemini_model_name)
    print(f"Using {gemini_model_name}", flush=True)

    tokenizer = ByteTokenizer()
    model_a = load_model("model_a.pt")
    model_b = load_model("model_b.pt")

    rows = []
    for index, prompt in enumerate(PROMPTS, start=1):
        # Fixed per-prompt seed makes local sampling repeatable.
        torch.manual_seed(1337 + index)
        model_a_output = generate_local(model_a, tokenizer, prompt)
        torch.manual_seed(1337 + index)
        model_b_output = generate_local(model_b, tokenizer, prompt)
        gemini_output = generate_gemini(gemini, prompt)
        rows.append(
            {
                "prompt": prompt,
                "model_a": model_a_output,
                "model_b": model_b_output,
                "gemini": gemini_output,
            }
        )
        print(f"Completed prompt {index}/{len(PROMPTS)}", flush=True)

    write_results(rows, gemini_model_name)
    print(f"Saved comparison to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
