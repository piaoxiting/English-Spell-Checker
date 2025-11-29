import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, TreebankWordDetokenizer
from spellchecker import SpellChecker
from pathlib import Path
import csv
from datetime import datetime

# ----------------------------
# NLTK setup (Streamlit Cloud compatible)
# ----------------------------
def ensure_nltk():
    # punkt
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    # punkt_tab (NLTK 3.8+ requires this)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

ensure_nltk()


# ----------------------------
# Tokenization & Spell Checking
# ----------------------------
def is_candidate_word(tok: str, ignore_short=True, ignore_upper=True):
    if not tok.isalpha():
        return False
    if ignore_short and len(tok) <= 2:
        return False
    if ignore_upper and tok.isupper():
        return False
    return True


def check_spelling(text, ignore_short=True, ignore_upper=True):
    spell = SpellChecker(language="en")
    tokens = word_tokenize(text)
    errors = {}

    for tok in tokens:
        if is_candidate_word(tok, ignore_short, ignore_upper):
            if tok.lower() in spell.unknown([tok.lower()]):
                errors[tok] = spell.correction(tok.lower())

    return errors


def correct_spelling(text, spell_checker, ignore_short=True, ignore_upper=True):
    detok = TreebankWordDetokenizer()
    tokens = word_tokenize(text)

    candidate_indices = [i for i, t in enumerate(tokens)
                         if is_candidate_word(t, ignore_short, ignore_upper)]
    candidate_words = [tokens[i].lower() for i in candidate_indices]
    misspelled = spell_checker.unknown(candidate_words)

    for i, lw in zip(candidate_indices, candidate_words):
        if lw in misspelled:
            orig = tokens[i]
            suggestion = spell_checker.correction(lw)

            if orig.istitle():
                suggestion = suggestion.capitalize()
            elif orig.isupper():
                suggestion = suggestion.upper()

            tokens[i] = suggestion

    return detok.detokenize(tokens)


# ----------------------------
# CSV export & text report
# ----------------------------
def export_to_csv(error_summary: dict, output_folder: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_csv_path = output_folder / f"spelling_error_summary_{timestamp}.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Total Words", "Error Count", "Error Rate (%)"])

        for filename, data in error_summary.items():
            text = data["original_text"]
            total_words = sum(1 for t in word_tokenize(text) if t.isalpha())
            error_rate = (
                data["error_count"] / total_words * 100 if total_words > 0 else 0
            )
            writer.writerow(
                [filename, total_words, data["error_count"], f"{error_rate:.2f}"]
            )

    detailed_csv_path = output_folder / f"spelling_error_details_{timestamp}.csv"
    with detailed_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Misspelled Word", "Correction"])

        for filename, data in error_summary.items():
            for word in sorted(data["errors"].keys(), key=str.lower):
                writer.writerow([filename, word, data["errors"][word]])

    return summary_csv_path, detailed_csv_path


def write_text_report(error_summary: dict, output_folder: Path):
    report_path = output_folder / "_spelling_error_report.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("🧾 Spelling Error Summary Report\n")
        f.write("=" * 40 + "\n\n")

        for filename, data in error_summary.items():
            f.write(f"📄 File: {filename}\n")
            f.write(f"❌ Total errors: {data['error_count']}\n")
            f.write("🔧 Corrections:\n")
            for word, corr in data["errors"].items():
                f.write(f"  {word} → {corr}\n")
            f.write("\n" + "-" * 30 + "\n\n")

    return report_path


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="English Spell Checker", layout="wide")
st.title("✨ English Spell Checker")

ignore_short = st.sidebar.checkbox("🔠 Ignore short words (≤2 letters)", True)
ignore_upper = st.sidebar.checkbox("🧢 Ignore ALL CAPS words", True)

tab1, tab2 = st.tabs(["📝 Text Mode", "📂 Folder Mode"])


# ----------------------------
# TAB 1 — Text Mode
# ----------------------------
with tab1:
    st.subheader("💬 Enter your text")
    text_input = st.text_area("Input text", height=200)

    if st.button("Check Spelling"):
        if not text_input.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            errors = check_spelling(text_input, ignore_short, ignore_upper)

            if errors:
                st.subheader("✏️ Corrections")
                for w, c in errors.items():
                    st.write(f"- **{w} → {c}**")

                st.success(f"Total errors found: **{len(errors)}**")
            else:
                st.success("🎉 No spelling errors found!")


# ----------------------------
# TAB 2 — Folder Mode
# ----------------------------
with tab2:
    st.subheader("📂 Upload .txt files to process")

    files = st.file_uploader(
        "Upload multiple .txt files",
        type=["txt"],
        accept_multiple_files=True
    )

    output_folder = st.text_input("Output folder name", "spellcheck_output")

    if st.button("🚀 Start Spell Check on Folder"):
        if not files:
            st.warning("⚠️ Please upload at least one .txt file.")
        else:
            out_path = Path(output_folder)
            out_path.mkdir(exist_ok=True)

            spell = SpellChecker()
            error_summary = {}

            for f in files:
                raw = f.read().decode("utf-8", errors="replace")
                errors = check_spelling(raw, ignore_short, ignore_upper)
                corrected = correct_spelling(raw, spell, ignore_short, ignore_upper)

                (out_path / f.name).write_text(corrected, encoding="utf-8")

                error_summary[f.name] = {
                    "error_count": len(errors),
                    "errors": errors,
                    "original_text": raw
                }

                st.write(f"📄 Processed **{f.name}** | Errors: {len(errors)}")

            sum_csv, det_csv = export_to_csv(error_summary, out_path)
            report_file = write_text_report(error_summary, out_path)

            st.success("🎉 Spell checking completed!")

            st.write(f"📊 Summary CSV: `{sum_csv.name}`")
            st.write(f"📋 Detailed CSV: `{det_csv.name}`")
            st.write(f"🧾 Report: `{report_file.name}`")

            st.download_button("⬇️ Download Summary CSV", sum_csv.read_bytes(), file_name=sum_csv.name)
            st.download_button("⬇️ Download Detailed CSV", det_csv.read_bytes(), file_name=det_csv.name)
            st.download_button("⬇️ Download Report", report_file.read_bytes(), file_name=report_file.name)
