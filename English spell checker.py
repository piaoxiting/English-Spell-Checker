import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, TreebankWordDetokenizer
from spellchecker import SpellChecker
from pathlib import Path
import csv
from datetime import datetime
import os

# ----------------------------
# NLTK setup
# ----------------------------
def ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

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
    spell = SpellChecker(language='en')
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
# CSV export & report
# ----------------------------
def export_to_csv(error_summary: dict, output_folder: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_csv_path = output_folder / f"spelling_error_summary_{timestamp}.csv"
    with summary_csv_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Total Word Count', 'Error Count', 'Error Rate (%)'])
        for filename, data in error_summary.items():
            total_words = sum(1 for t in word_tokenize(data['original_text']) if t.isalpha())
            error_rate = (data['error_count'] / total_words * 100) if total_words > 0 else 0
            writer.writerow([filename, total_words, data['error_count'], f"{error_rate:.2f}"])

    detailed_csv_path = output_folder / f"spelling_error_details_{timestamp}.csv"
    with detailed_csv_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Misspelled Word', 'Correction'])
        for filename, data in error_summary.items():
            for word in sorted(data['errors'].keys(), key=str.lower):
                writer.writerow([filename, word, data['errors'][word]])

    return summary_csv_path, detailed_csv_path


def write_text_report(error_summary: dict, output_folder: Path):
    report_path = output_folder / "_spelling_error_report.txt"
    with report_path.open('w', encoding='utf-8') as report:
        report.write("🧾 Spelling Error Summary Report\n")
        report.write("=" * 35 + "\n\n")
        for filename, data in error_summary.items():
            report.write(f"📄 File: {filename}\n")
            report.write(f"❌ Total errors: {data['error_count']}\n")
            report.write("🔧 Corrections:\n")
            for word in sorted(data['errors'].keys(), key=str.lower):
                report.write(f"  {word} → {data['errors'][word]}\n")
            report.write("\n" + "-" * 30 + "\n\n")
    return report_path


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="English Spell Checker", layout="wide")
st.title("✨ English Spell Checker (Streamlit Ver.)")

ignore_short = st.sidebar.checkbox("🔠 Ignore short words (<=2 letters)", value=True)
ignore_upper = st.sidebar.checkbox("🧢 Ignore ALL CAPS words", value=True)


# ----------------------------
# Tab structure
# ----------------------------
tab1, tab2 = st.tabs(["📝 Text Mode", "📂 Folder Mode"])


# ============================
# TAB 1 — TEXT MODE
# ============================
with tab1:
    st.subheader("💬 Enter your text:")
    text_input = st.text_area("Text input", height=200)

    if st.button("Check Spelling"):
        if text_input.strip():
            errors = check_spelling(text_input, ignore_short, ignore_upper)
            if errors:
                st.subheader("✏️ Corrections")
                for word, corr in errors.items():
                    st.write(f"**{word} → {corr}**")
                st.success(f"Total errors found: {len(errors)}")
            else:
                st.success("🎉 No spelling errors found!")


# ============================
# TAB 2 — FOLDER MODE
# ============================
with tab2:
    st.subheader("📂 Upload a folder of .txt files")

    uploaded_files = st.file_uploader(
        "Upload multiple .txt files",
        type=["txt"],
        accept_multiple_files=True
    )

    output_folder = st.text_input(
        "Output folder name (will be created if not exists)",
        value="spellcheck_output"
    )

    if st.button("🚀 Start Spell Check on Folder"):
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one .txt file.")
        else:
            out_path = Path(output_folder)
            out_path.mkdir(exist_ok=True)

            error_summary = {}
            spell = SpellChecker(language='en')

            for file in uploaded_files:
                content = file.read().decode("utf-8", errors="replace")

                errors = check_spelling(content, ignore_short, ignore_upper)
                corrected = correct_spelling(content, spell, ignore_short, ignore_upper)

                # Save corrected file
                corrected_path = out_path / file.name
                corrected_path.write_text(corrected, encoding='utf-8')

                error_summary[file.name] = {
                    "error_count": len(errors),
                    "errors": errors,
                    "original_text": content
                }

                st.write(f"📄 Processed: **{file.name}** | Errors: {len(errors)}")

            # Export CSV + Report
            summary_csv, detailed_csv = export_to_csv(error_summary, out_path)
            report_path = write_text_report(error_summary, out_path)

            st.success("🎉 Spell checking completed!")
            st.write(f"📊 Summary CSV: `{summary_csv.name}`")
            st.write(f"📋 Detailed CSV: `{detailed_csv.name}`")
            st.write(f"🧾 Text report: `{report_path.name}`")

            with open(summary_csv, "rb") as f:
                st.download_button("⬇️ Download Summary CSV", f, file_name=summary_csv.name)

            with open(detailed_csv, "rb") as f:
                st.download_button("⬇️ Download Detailed CSV", f, file_name=detailed_csv.name)

            with open(report_path, "rb") as f:
                st.download_button("⬇️ Download Text Report", f, file_name=report_path.name)
