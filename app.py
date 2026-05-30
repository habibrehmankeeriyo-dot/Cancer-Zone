import gradio as gr
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "InstaDeepAI/nucleotide-transformer-500m-human-ref")

print("Loading tokenizer and model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.eval()
print("Model ready.")

# ── Known cancer driver genes (COSMIC Cancer Gene Census Tier 1) ───────────────
DRIVER_GENES = {
    "TP53","KRAS","EGFR","BRAF","PIK3CA","PTEN","RB1","CDKN2A","APC","VHL",
    "BRCA1","BRCA2","MLH1","MSH2","STK11","SMAD4","FBXW7","NOTCH1","IDH1",
    "IDH2","NPM1","FLT3","DNMT3A","TET2","ASXL1","SF3B1","U2AF1","SRSF2",
    "KEAP1","NFE2L2","MET","ALK","RET","ROS1","NTRK1","NTRK2","NTRK3",
    "ERBB2","ERBB3","MYC","MYCN","CCND1","CDK4","CDK6","MDM2","MDM4",
    "NF1","NF2","TSC1","TSC2","PTCH1","SMO","CTNNB1","AXIN1","AXIN2",
    "KIT","PDGFRA","ABL1","BCR","JAK2","STAT3","STAT5A","STAT5B",
    "POLE","POLD1","MSH6","PMS2","EPCAM","ATM","CHEK2","PALB2",
}

# Cancer type hints per gene
CANCER_HINTS = {
    "TP53":   "Pan-cancer (breast, lung, colon, ovarian…)",
    "KRAS":   "Lung, pancreatic, colorectal",
    "EGFR":   "Lung adenocarcinoma",
    "BRAF":   "Melanoma, colorectal, thyroid",
    "PIK3CA": "Breast, endometrial, cervical",
    "BRCA1":  "Breast, ovarian",
    "BRCA2":  "Breast, ovarian, pancreatic",
    "IDH1":   "Glioma, AML",
    "IDH2":   "Glioma, AML",
    "FLT3":   "AML",
    "ABL1":   "CML (BCR-ABL fusion)",
    "VHL":    "Renal cell carcinoma",
    "APC":    "Colorectal",
    "PTEN":   "Endometrial, glioma, breast",
    "ALK":    "Lung, ALCL",
    "MET":    "Lung, gastric",
    "ERBB2":  "Breast, gastric",
    "KIT":    "GIST, AML",
    "RB1":    "Retinoblastoma, osteosarcoma",
    "NF1":    "NF1, MPNST",
    "CDKN2A": "Melanoma, pancreatic",
    "STK11":  "Lung, Peutz-Jeghers",
}

VARIANT_SEVERITY = {
    "Nonsense_Mutation": "High",
    "Frame_Shift_Del":   "High",
    "Frame_Shift_Ins":   "High",
    "Splice_Site":       "High",
    "Missense_Mutation": "Medium",
    "In_Frame_Del":      "Medium",
    "In_Frame_Ins":      "Medium",
    "Silent":            "Low",
    "3'UTR":             "Low",
    "5'UTR":             "Low",
    "Intron":            "Low",
}


def classify_sequence(seq: str) -> tuple[str, float]:
    """Run model on a short DNA/variant text. Returns (label, confidence)."""
    inputs = tokenizer(seq, return_tensors="pt", truncation=True,
                       padding="max_length", max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    label = "Driver" if np.argmax(probs) == 1 else "Passenger"
    confidence = float(np.max(probs))
    return label, confidence


def tier(gene, prediction, severity):
    """Assign clinical tier 1-3 or Passenger."""
    if prediction == "Passenger":
        return "Passenger"
    if gene in DRIVER_GENES and severity == "High":
        return "Tier 1 — Strong"
    if gene in DRIVER_GENES:
        return "Tier 2 — Likely"
    if severity == "High":
        return "Tier 3 — Possible"
    return "Tier 3 — Possible"


def color_tier(t):
    colors = {
        "Tier 1 — Strong":  "background-color:#fde8e8; color:#7f1d1d",
        "Tier 2 — Likely":  "background-color:#fef3c7; color:#78350f",
        "Tier 3 — Possible":"background-color:#e0f2fe; color:#0c4a6e",
        "Passenger":        "background-color:#f0fdf4; color:#14532d",
    }
    return colors.get(t, "")


def process_maf(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Hugo_Symbol","Chromosome","Start_Position",
                "Reference_Allele","Tumor_Seq_Allele2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in file: {missing}")

    results = []
    for _, row in df.iterrows():
        gene   = str(row.get("Hugo_Symbol","?"))
        chrom  = str(row.get("Chromosome","?"))
        pos    = str(row.get("Start_Position","?"))
        ref    = str(row.get("Reference_Allele","?"))
        alt    = str(row.get("Tumor_Seq_Allele2","?"))
        vclass = str(row.get("Variant_Classification","?"))
        sample = str(row.get("Tumor_Sample_Barcode", row.get("Sample_ID","?")))

        seq_text = f"{chrom}:{pos} {ref}>{alt} {gene}"
        try:
            pred, conf = classify_sequence(seq_text)
        except Exception:
            pred, conf = "Error", 0.0

        sev  = VARIANT_SEVERITY.get(vclass, "Unknown")
        t    = tier(gene, pred, sev)
        hint = CANCER_HINTS.get(gene, "—")
        in_cosmic = "Yes" if gene in DRIVER_GENES else "No"

        results.append({
            "Sample":           sample,
            "Gene":             gene,
            "Variant":          f"{ref}>{alt}",
            "Position":         f"chr{chrom}:{pos}",
            "Classification":   vclass,
            "Severity":         sev,
            "Prediction":       pred,
            "Confidence":       f"{conf:.1%}",
            "Clinical tier":    t,
            "In COSMIC CGC":    in_cosmic,
            "Cancer type hint": hint,
        })

    return pd.DataFrame(results)


def analyze_file(file):
    if file is None:
        return None, "Please upload a MAF or TSV file."
    try:
        df = pd.read_csv(file.name, sep="\t", comment="#", low_memory=False)
        result_df = process_maf(df)

        drivers = result_df[result_df["Prediction"] == "Driver"]
        n_total   = len(result_df)
        n_drivers = len(drivers)
        n_tier1   = len(result_df[result_df["Clinical tier"].str.startswith("Tier 1")])
        cosmic_hits = result_df[result_df["In COSMIC CGC"] == "Yes"]["Gene"].unique()

        summary = (
            f"**Total variants analysed:** {n_total}  \n"
            f"**Predicted driver mutations:** {n_drivers} ({n_drivers/max(n_total,1):.1%})  \n"
            f"**Tier 1 (strong evidence):** {n_tier1}  \n"
            f"**COSMIC CGC gene hits:** {', '.join(sorted(cosmic_hits)) if len(cosmic_hits) else 'None'}"
        )
        return result_df, summary

    except Exception as e:
        return None, f"Error processing file: {e}"


def analyze_variant(gene, chrom, pos, ref, alt, vclass):
    if not all([gene, chrom, pos, ref, alt]):
        return "Please fill in all fields."
    seq_text = f"{chrom}:{pos} {ref}>{alt} {gene}"
    try:
        pred, conf = classify_sequence(seq_text)
    except Exception as e:
        return f"Model error: {e}"

    sev  = VARIANT_SEVERITY.get(vclass, "Unknown")
    t    = tier(gene, pred, sev)
    hint = CANCER_HINTS.get(gene.upper(), "No specific hint available")
    cosmic = "Yes" if gene.upper() in DRIVER_GENES else "No"

    return (
        f"### Result for {gene} {ref}>{alt}\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Prediction | **{pred}** |\n"
        f"| Confidence | {conf:.1%} |\n"
        f"| Severity | {sev} |\n"
        f"| Clinical tier | {t} |\n"
        f"| In COSMIC CGC | {cosmic} |\n"
        f"| Cancer type hint | {hint} |"
    )


# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Cancer Mutation Detector", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # Cancer Mutation Detector
        Upload a somatic mutation file (MAF/TSV) or enter a single variant manually.
        The model predicts whether each mutation is a **driver** or **passenger**,
        assigns a clinical evidence tier, and cross-references COSMIC Cancer Gene Census.

        > **Data sources accepted:** cBioPortal MAF · TCGA GDC MAF · any TSV with standard MAF columns
        """
    )

    with gr.Tab("Upload MAF file"):
        with gr.Row():
            file_input = gr.File(label="Upload MAF / TSV file", file_types=[".txt",".tsv",".maf",".csv"])
        analyze_btn = gr.Button("Analyse mutations", variant="primary")
        summary_out = gr.Markdown(label="Summary")
        table_out   = gr.Dataframe(
            label="Mutation predictions",
            wrap=True,
            interactive=False,
        )
        analyze_btn.click(fn=analyze_file,
                          inputs=file_input,
                          outputs=[table_out, summary_out])

    with gr.Tab("Single variant"):
        with gr.Row():
            gene_in  = gr.Textbox(label="Gene symbol", placeholder="TP53")
            chrom_in = gr.Textbox(label="Chromosome",  placeholder="17")
            pos_in   = gr.Textbox(label="Position",    placeholder="7674220")
        with gr.Row():
            ref_in   = gr.Textbox(label="Reference allele", placeholder="C")
            alt_in   = gr.Textbox(label="Alternate allele", placeholder="T")
            vclass_in = gr.Dropdown(
                label="Variant classification",
                choices=list(VARIANT_SEVERITY.keys()),
                value="Missense_Mutation"
            )
        single_btn = gr.Button("Predict", variant="primary")
        single_out = gr.Markdown()
        single_btn.click(fn=analyze_variant,
                         inputs=[gene_in, chrom_in, pos_in, ref_in, alt_in, vclass_in],
                         outputs=single_out)

    with gr.Tab("How to use"):
        gr.Markdown(
            """
            ## Getting your data

            ### Option A — cBioPortal (easiest, no login)
            1. Go to [cbioportal.org](https://www.cbioportal.org)
            2. Search for a cancer study e.g. **TCGA Lung Adenocarcinoma**
            3. Click **Download** → **All data** → unzip
            4. Upload the `data_mutations.txt` file here

            ### Option B — TCGA via GDC portal
            1. Go to [portal.gdc.cancer.gov](https://portal.gdc.cancer.gov)
            2. Filter by **Data Type: Masked Somatic Mutation**
            3. Add to cart → Download manifest → use GDC Data Transfer Tool
            4. Upload the `.maf.gz` file (unzip first)

            ## Understanding the output

            | Tier | Meaning |
            |---|---|
            | Tier 1 — Strong | Known COSMIC driver + high-impact variant |
            | Tier 2 — Likely | Known COSMIC driver gene |
            | Tier 3 — Possible | Model predicts driver, not in COSMIC |
            | Passenger | Likely non-functional mutation |

            ## Required MAF columns
            `Hugo_Symbol` · `Chromosome` · `Start_Position` · `Reference_Allele` · `Tumor_Seq_Allele2`
            """
        )

demo.launch()
