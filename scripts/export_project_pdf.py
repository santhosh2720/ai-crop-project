from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "Project_Full_Documentation.md"
OUTPUT = ROOT / "docs" / "Project_Full_Documentation.pdf"


def _load_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def export_with_reportlab(text: str) -> bool:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except Exception:
        return False

    styles = getSampleStyleSheet()
    story = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 8))
            continue

        if line.startswith("### "):
            story.append(Paragraph(f"<b>{line[4:]}</b>", styles["Heading3"]))
        elif line.startswith("## "):
            story.append(Paragraph(f"<b>{line[3:]}</b>", styles["Heading2"]))
        elif line.startswith("# "):
            story.append(Paragraph(f"<b>{line[2:]}</b>", styles["Title"]))
        elif line.startswith("- "):
            story.append(Paragraph(f"&bull; {line[2:]}", styles["BodyText"]))
        else:
            escaped = (
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            story.append(Paragraph(escaped, styles["BodyText"]))
        story.append(Spacer(1, 4))

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    doc.build(story)
    return True


def export_with_fpdf(text: str) -> bool:
    try:
        from fpdf import FPDF
    except Exception:
        return False

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            pdf.ln(4)
            continue

        if line.startswith("# "):
            pdf.set_font("Arial", "B", 16)
            pdf.multi_cell(0, 8, line[2:])
            pdf.set_font("Arial", size=11)
        elif line.startswith("## "):
            pdf.set_font("Arial", "B", 14)
            pdf.multi_cell(0, 7, line[3:])
            pdf.set_font("Arial", size=11)
        elif line.startswith("### "):
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(0, 6, line[4:])
            pdf.set_font("Arial", size=11)
        elif line.startswith("- "):
            pdf.multi_cell(0, 6, f"* {line[2:]}")
        else:
            pdf.multi_cell(0, 6, line)

    pdf.output(str(OUTPUT))
    return True


def main() -> None:
    text = _load_text()
    if export_with_reportlab(text) or export_with_fpdf(text):
        print(OUTPUT)
        return
    raise RuntimeError("No supported PDF library found. Install reportlab or fpdf.")


if __name__ == "__main__":
    main()
