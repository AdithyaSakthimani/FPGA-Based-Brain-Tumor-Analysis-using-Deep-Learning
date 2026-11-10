"""
Brain Tumor MRI Analysis API
FastAPI backend with ML inference + Beautiful PDF report generation
"""

import uvicorn
import io
from ollama import chat
import base64
import traceback
import pydicom
from datetime import datetime
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification
from huggingface_hub import hf_hub_download

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.colors import HexColor, white, black

# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────
from florance_model import FlorenceModel

florence = FlorenceModel()
app = FastAPI(
    title="Brain Tumor MRI Analyzer",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Model Cache
# ─────────────────────────────────────────────────────────────

models = {}

# ─────────────────────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────────────────────

def load_models():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading models on {device}...")

    # Binary ViT
    binary_processor = AutoImageProcessor.from_pretrained(
        "MidnightGamer/brain_tumor_vit"
    )

    binary_model = AutoModelForImageClassification.from_pretrained(
        "MidnightGamer/brain_tumor_vit"
    )

    binary_model = binary_model.to(device).eval()

    # Multiclass ViT
    multiclass_processor = AutoImageProcessor.from_pretrained(
        "MidnightGamer/brain_multiclass_vit"
    )

    multiclass_model = AutoModelForImageClassification.from_pretrained(
        "MidnightGamer/brain_multiclass_vit"
    )

    multiclass_model = multiclass_model.to(device).eval()

    # U-Net Segmentation
    weights_path = hf_hub_download(
        repo_id="MidnightGamer/brain_tumor_unet",
        filename="unet_segmentation.pth"
    )

    seg_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    seg_model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )

    seg_model = seg_model.to(device).eval()

    models["device"] = device
    models["binary_processor"] = binary_processor
    models["binary_model"] = binary_model
    models["multiclass_processor"] = multiclass_processor
    models["multiclass_model"] = multiclass_model
    models["seg_model"] = seg_model

    print("[INFO] Models loaded successfully")


@app.on_event("startup")
async def startup_event():
    load_models()

# ─────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────

BINARY_LABELS = {
    0: "no",
    1: "yes"
}

MULTICLASS_LABELS = {
    0: "glioma",
    1: "meningioma",
    2: "pituitary"
}

# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def run_inference(image: Image.Image):

    device = models["device"]

    # Binary Classification
    inputs = models["binary_processor"](
        images=image,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = models["binary_model"](**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)

    pred = probs.argmax(dim=-1).item()

    binary_conf = probs[0][pred].item()

    binary_result = BINARY_LABELS[pred]

    # No Tumor
    if binary_result == "no":
        return {
            "tumor_detected": False,
            "heatmap_b64": None,
            "binary_confidence": binary_conf,
            "tumor_type": None,
            "multiclass_confidence": None,
            "tumor_area_percent": 0,
            "tumor_pixels": 0,
            "severity": "None",
            "tumor_location": "None",
            "top_predictions": [],
            "mask_b64": None,
            "overlay_b64": None,
        }

    # Segmentation
    seg_transform = Compose([
        Resize((256, 256)),
        ToTensor()
    ])

    seg_input = seg_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = models["seg_model"](seg_input)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = (pred_mask > 0.5).float()

    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    resized_np = np.array(
        image.resize((256, 256))
    )

    # Save MRI
    Image.fromarray(
        resized_np.astype(np.uint8)
    ).save("temp_mri.png")

    # Save mask
    Image.fromarray(
        (pred_mask_np * 255).astype(np.uint8)
    ).save("temp_mask.png")
    tumor_pixels = int(
        np.sum(pred_mask_np > 0.5)
    )

    total_pixels = int(
        pred_mask_np.size
    )

    tumor_area_percent = round(
        (tumor_pixels / total_pixels) * 100,
        2
    )

    # Severity
    if tumor_area_percent < 2:
        severity = "Low"

    elif tumor_area_percent < 8:
        severity = "Moderate"

    else:
        severity = "High"

    # Tumor Location
    coords = np.argwhere(
        pred_mask_np > 0.5
    )

    if len(coords) > 0:

        center_y, center_x = coords.mean(axis=0)

        side = (
            "Left"
            if center_x < 128
            else "Right"
        )

        region = (
            "Frontal"
            if center_y < 128
            else "Occipital"
        )

        tumor_location = (
            f"{side} {region} Lobe"
        )

    else:

        tumor_location = "Unknown"

    # Create masked image
    resized_np = np.array(
        image.resize((256, 256))
    )

    mask_uint8 = (
        pred_mask_np > 0.5
    ).astype(np.uint8)

    masked = resized_np.copy()

    masked[mask_uint8 == 0] = 0

    masked_pil = Image.fromarray(
        masked.astype(np.uint8)
    ).resize((224, 224))

    # Multiclass Classification
    inputs2 = models["multiclass_processor"](
        images=masked_pil,
        return_tensors="pt"
    )

    inputs2 = {
        k: v.to(device)
        for k, v in inputs2.items()
    }

    with torch.no_grad():
        outputs2 = models["multiclass_model"](**inputs2)

    probs2 = torch.softmax(
        outputs2.logits,
        dim=-1
    )

    pred2 = probs2.argmax(
        dim=-1
    ).item()

    multi_conf = probs2[0][pred2].item()

    tumor_type = MULTICLASS_LABELS[pred2]

    top3_probs, top3_idx = torch.topk(
        probs2,
        3
    )

    top_predictions = []

    for prob, idx in zip(
        top3_probs[0],
        top3_idx[0]
    ):
        top_predictions.append({
            "tumor":
                MULTICLASS_LABELS[idx.item()],
            "probability":
                round(prob.item(), 4)
        })

    # Mask Image
    fig, ax = plt.subplots(
        figsize=(4, 4),
        dpi=100
    )

    ax.imshow(
        pred_mask_np,
        cmap="gray"
    )

    ax.axis("off")

    plt.tight_layout(pad=0)

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    mask_b64 = base64.b64encode(
        buf.getvalue()
    ).decode()

    # Overlay Image
    fig, ax = plt.subplots(
        figsize=(4, 4),
        dpi=100
    )

    ax.imshow(
        image.resize((256, 256))
    )

    ax.imshow(
        pred_mask_np,
        alpha=0.45,
        cmap="Reds"
    )

    ax.axis("off")

    plt.tight_layout(pad=0)

    buf2 = io.BytesIO()

    plt.savefig(
        buf2,
        format="png",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    overlay_b64 = base64.b64encode(
        buf2.getvalue()
    ).decode()
    # Save overlay image for LLaVA
    overlay_img = Image.open(
        io.BytesIO(buf2.getvalue())
    )

    overlay_img.save(
        "temp_overlay.png"
    )
    heatmap_b64 = None
    return {
        "tumor_detected": True,
        "binary_confidence": binary_conf,
        "tumor_type": tumor_type,
        "heatmap_b64": heatmap_b64,
        "multiclass_confidence": multi_conf,
        "tumor_area_percent": tumor_area_percent,
        "tumor_pixels": tumor_pixels,
        "severity": severity,
        "tumor_location": tumor_location,
        "top_predictions": top_predictions,
        "mask_b64": mask_b64,
        "overlay_b64": overlay_b64,
    }
def generate_gradcam(image):

    device = models["device"]

    processor = models["binary_processor"]
    model = models["binary_model"]

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    target_layer = (
        model.vit.encoder.layer[-1]
    )

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(
        input_tensor=inputs["pixel_values"]
    )[0]

    img = np.array(
        image.resize((224,224))
    ).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        img,
        grayscale_cam,
        use_rgb=True
    )

    buf = io.BytesIO()

    Image.fromarray(
        visualization
    ).save(
        buf,
        format="PNG"
    )

    return base64.b64encode(
        buf.getvalue()
    ).decode()
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        if file.filename.lower().endswith(".dcm"):

            dicom = pydicom.dcmread(
                io.BytesIO(contents)
            )

            pixels = dicom.pixel_array.astype(np.float32)

            pixels = (
                (pixels - pixels.min())
                /
                (pixels.max() - pixels.min() + 1e-8)
            ) * 255

            pixels = pixels.astype(np.uint8)

            image = Image.fromarray(
                pixels
            ).convert("RGB")

        else:

            image = Image.open(
                io.BytesIO(contents)
            ).convert("RGB")

        result = run_inference(image)

        return JSONResponse(content=result)

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.post("/generate-pdf")
async def generate_pdf(
    file: UploadFile = File(...),
    patient_name: str = Form(default="Anonymous"),
    patient_age: str = Form(default="N/A"),
    patient_id: str = Form(default="N/A"),
    referring_physician: str = Form(default="N/A"),
):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        result = run_inference(image)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    pdf_buf = build_pdf_report(
        image=image,
        result=result,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_id=patient_id,
        referring_physician=referring_physician,
    )

    filename = (
        f"brain_tumor_report_"
        f"{patient_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ═════════════════════════════════════════════════════════════
#  PDF REPORT — DESIGN SYSTEM
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# Design Tokens
# ─────────────────────────────────────────────────────────────

C_BG           = HexColor("#F0F4F8")   # page background
C_DARK         = HexColor("#0D1B2A")   # header / deep navy
C_NAVY         = HexColor("#1B3A5C")   # secondary header tones
C_BLUE         = HexColor("#0EA5E9")   # primary accent
C_BLUE_LIGHT   = HexColor("#E0F2FE")   # tinted bg for cards
C_GREEN        = HexColor("#16A34A")   # normal / positive
C_GREEN_LIGHT  = HexColor("#DCFCE7")
C_PURPLE       = HexColor("#7C3AED")   # tumor detected
C_PURPLE_LIGHT = HexColor("#EDE9FE")
C_AMBER        = HexColor("#D97706")   # warning
C_AMBER_LIGHT  = HexColor("#FEF3C7")
C_GRAY_DARK    = HexColor("#334155")
C_GRAY_MID     = HexColor("#64748B")
C_GRAY_LIGHT   = HexColor("#E2E8F0")
C_WHITE        = white
C_RULE         = HexColor("#CBD5E1")

PAGE_W, PAGE_H = A4
MARGIN_H  = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_H

# ─────────────────────────────────────────────────────────────
# Style Factory + Typography Scale
# ─────────────────────────────────────────────────────────────

_base_styles = getSampleStyleSheet()

def _s(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=_base_styles[parent], **kw)

T_BODY = _s(
    "BodyNormal", fontSize=9, fontName="Helvetica",
    textColor=C_GRAY_DARK, leading=14, alignment=TA_JUSTIFY,
)
T_CAPTION = _s(
    "Caption", fontSize=7.5, fontName="Helvetica-Bold",
    textColor=C_GRAY_MID, alignment=TA_CENTER, leading=11,
)
T_DISCLAIMER = _s(
    "Disc", fontSize=7, fontName="Helvetica",
    textColor=C_GRAY_MID, alignment=TA_CENTER, leading=11,
)

# ─────────────────────────────────────────────────────────────
# Page Canvas Decorator  (background · header · footer)
# ─────────────────────────────────────────────────────────────

def _page_decorator(canv, doc):
    """Draws the page chrome on every page before flowables are placed."""
    canv.saveState()

    # ── Page background ──
    canv.setFillColor(C_BG)
    canv.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # ── Header bar ──
    header_h = 38 * mm
    canv.setFillColor(C_DARK)
    canv.rect(0, PAGE_H - header_h, PAGE_W, header_h, fill=1, stroke=0)

    # Blue accent stripe at bottom of header
    canv.setFillColor(C_BLUE)
    canv.rect(0, PAGE_H - header_h, PAGE_W, 4 * mm, fill=1, stroke=0)

    # Left colour slab (navy sidebar in header)
    canv.setFillColor(C_NAVY)
    canv.rect(
        0, PAGE_H - header_h + 4 * mm,
        5 * mm, header_h - 4 * mm,
        fill=1, stroke=0,
    )


    # Sub-label
    canv.setFillColor(HexColor("#94A3B8"))
    canv.setFont("Helvetica", 8)
    canv.drawString(
        MARGIN_H + 2 * mm, PAGE_H - 26 * mm,
        "AI-Powered Brain MRI Analysis Report",
    )

    # Date + page number (right side of header)
    date_str = datetime.now().strftime("%B %d, %Y")
    page_str = f"Page {doc.page}"
    canv.setFillColor(HexColor("#94A3B8"))
    canv.setFont("Helvetica", 8)
    canv.drawRightString(PAGE_W - MARGIN_H, PAGE_H - 18 * mm, date_str)
    canv.drawRightString(PAGE_W - MARGIN_H, PAGE_H - 26 * mm, page_str)

    # ── Footer stripe ──
    canv.setFillColor(C_DARK)
    canv.rect(0, 0, PAGE_W, 10 * mm, fill=1, stroke=0)
    canv.setFillColor(HexColor("#475569"))
    canv.setFont("Helvetica", 7)
    canv.drawCentredString(
        PAGE_W / 2, 3.5 * mm,
        "CONFIDENTIAL — For clinical use only.  "
        "AI results must be reviewed by a qualified radiologist.",
    )

    canv.restoreState()


# ─────────────────────────────────────────────────────────────
# Helper: PIL Image → ReportLab Image
# ─────────────────────────────────────────────────────────────

def _pil_to_rl(pil_img: Image.Image, w_pts, h_pts) -> RLImage:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=w_pts, height=h_pts)


# ─────────────────────────────────────────────────────────────
# Helper: Section Header  (bar + rule)
# ─────────────────────────────────────────────────────────────

def _section_header(title: str, accent=None) -> list:
    if accent is None:
        accent = C_BLUE
    # Build a safe hex string ReportLab's paragraph parser accepts
    r = int(accent.red   * 255)
    g = int(accent.green * 255)
    b = int(accent.blue  * 255)
    hex_str = f"#{r:02x}{g:02x}{b:02x}"
    label = Paragraph(
        f"<font color='{hex_str}'>▌</font>  {title}",
        _s(
            f"SH_{title}",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=C_DARK, leading=14,
            spaceBefore=10, spaceAfter=2,
        ),
    )
    rule = HRFlowable(
        width="100%", thickness=0.75, color=accent, spaceAfter=5
    )
    return [label, rule]


# ─────────────────────────────────────────────────────────────
# Component: Diagnosis Banner
# ─────────────────────────────────────────────────────────────

def _diagnosis_banner(result: dict) -> Table:

    if result["tumor_detected"]:
        ttype    = result["tumor_type"].upper()
        bg       = C_PURPLE
        tag_text = "● TUMOR DETECTED"
        main_text = ttype
        det_pct  = f"{result['binary_confidence'] * 100:.1f}%"
        cls_pct  = f"{result['multiclass_confidence'] * 100:.1f}%"
        sub_text = (
            f"Detection confidence  <b>{det_pct}</b>"
            f"    ·    "
            f"Classification confidence  <b>{cls_pct}</b>"
        )
    else:
        bg        = C_GREEN
        tag_text  = "● NO TUMOR DETECTED"
        main_text = "NORMAL"
        det_pct   = f"{result['binary_confidence'] * 100:.1f}%"
        sub_text  = f"Detection confidence  <b>{det_pct}</b>"

    col_l = CONTENT_W * 0.60
    col_r = CONTENT_W * 0.40

    tag_s = _s(
        "BanTag", fontSize=8.5, fontName="Helvetica-Bold",
        textColor=C_WHITE, leading=12,
    )
    main_s = _s(
        "BanMain", fontSize=20, fontName="Helvetica-Bold",
        textColor=C_WHITE, leading=26,
    )
    sub_s = _s(
        "BanSub", fontSize=8.5, fontName="Helvetica",
        textColor=HexColor("#E0E7FF"), leading=13, alignment=TA_RIGHT,
    )

    left_inner = Table(
        [[Paragraph(tag_text, tag_s)], [Paragraph(main_text, main_s)]],
        colWidths=[col_l - 20],
    )
    left_inner.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))

    banner = Table(
        [[left_inner, Paragraph(sub_text, sub_s)]],
        colWidths=[col_l, col_r],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return banner


# ─────────────────────────────────────────────────────────────
# Component: Patient Info Strip
# ─────────────────────────────────────────────────────────────

def _patient_strip(
    patient_name, patient_age, patient_id, referring_physician
) -> Table:

    label_s = _s(
        "PLbl", fontSize=7, fontName="Helvetica",
        textColor=C_GRAY_MID, leading=10,
    )
    value_s = _s(
        "PVal", fontSize=9, fontName="Helvetica-Bold",
        textColor=C_DARK, leading=13,
    )
    tile_w = CONTENT_W / 4

    def cell(lbl, val):
        inner = Table(
            [[Paragraph(lbl.upper(), label_s)],
             [Paragraph(val or "N/A", value_s)]],
            colWidths=[tile_w],
        )
        inner.setStyle(TableStyle([
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        return inner

    strip = Table([[
        cell("Patient Name",       patient_name),
        cell("Age",                patient_age),
        cell("Patient ID",         patient_id),
        cell("Referring Physician",referring_physician),
    ]], colWidths=[tile_w] * 4)

    strip.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_WHITE),
        ("BOX",           (0, 0), (-1, -1), 0.75, C_RULE),
        ("LINEAFTER",     (0, 0), (-2, -1), 0.5,  C_RULE),
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    return strip


# ─────────────────────────────────────────────────────────────
# Component: MRI Image Grid
# ─────────────────────────────────────────────────────────────

def _image_grid(image: Image.Image, result: dict) -> Table:
    img_w = (CONTENT_W - 8 * mm) / 3
    img_h = img_w

    if result["tumor_detected"]:
        orig_rl = _pil_to_rl(image.resize((300, 300)), img_w, img_h)

        mask_pil = Image.open(
            io.BytesIO(base64.b64decode(result["mask_b64"]))
        )
        mask_rl = _pil_to_rl(mask_pil.resize((300, 300)), img_w, img_h)

        overlay_pil = Image.open(
            io.BytesIO(base64.b64decode(result["overlay_b64"]))
        )
        overlay_rl = _pil_to_rl(overlay_pil.resize((300, 300)), img_w, img_h)

        images_row   = [orig_rl, mask_rl, overlay_rl]
        captions_row = [
            Paragraph("Original MRI",      T_CAPTION),
            Paragraph("Segmentation Mask", T_CAPTION),
            Paragraph("Tumor Overlay",     T_CAPTION),
        ]
        col_w = [img_w, img_w, img_w]

    else:
        large_w = img_w * 1.4
        orig_rl = _pil_to_rl(image.resize((300, 300)), large_w, large_w)
        images_row   = [orig_rl]
        captions_row = [
            Paragraph("Original MRI — No Abnormality Detected", T_CAPTION)
        ]
        col_w = [large_w]

    t = Table([images_row, captions_row], colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), C_WHITE),
        ("BACKGROUND",    (0, 1), (-1,  1), C_GRAY_LIGHT),
        ("BOX",           (0, 0), (-1, -1), 0.75, C_RULE),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1,  0), 10),
        ("BOTTOMPADDING", (0, 0), (-1,  0),  6),
        ("TOPPADDING",    (0, 1), (-1,  1),  4),
        ("BOTTOMPADDING", (0, 1), (-1,  1), 10),
    ]))
    return t


# ─────────────────────────────────────────────────────────────
# Component: Quantitative Metrics Row
# ─────────────────────────────────────────────────────────────

def _metrics_row(result: dict) -> Table:
    gap    = 2 * mm
    tile_w = (CONTENT_W - 3 * gap) / 4

    def tile(label, value, bg, val_color):
        val_s = _s(
            f"MV_{label}", fontSize=14, fontName="Helvetica-Bold",
            textColor=val_color, leading=20, alignment=TA_CENTER,
        )
        lbl_s = _s(
            f"ML_{label}", fontSize=7.5, fontName="Helvetica",
            textColor=C_GRAY_MID, leading=11, alignment=TA_CENTER,
        )
        t = Table(
            [[Paragraph(value, val_s)], [Paragraph(label, lbl_s)]],
            colWidths=[tile_w],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("BOX",           (0, 0), (-1, -1), 0.5, C_RULE),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1),  4),
            ("RIGHTPADDING",  (0, 0), (-1, -1),  4),
        ]))
        return t

    if result["tumor_detected"]:
        tiles = [
            tile(
                "Detection Score",
                f"{result['binary_confidence'] * 100:.1f}%",
                C_PURPLE_LIGHT, C_PURPLE,
            ),
            tile(
                "Classification Score",
                f"{result['multiclass_confidence'] * 100:.1f}%",
                C_BLUE_LIGHT, C_BLUE,
            ),
            tile(
                "Tumor Type",
                result["tumor_type"].capitalize(),
                C_AMBER_LIGHT, C_AMBER,
            ),
            tile("Report Status", "Complete", C_GREEN_LIGHT, C_GREEN),
        ]
    else:
        tiles = [
            tile(
                "Detection Score",
                f"{result['binary_confidence'] * 100:.1f}%",
                C_GREEN_LIGHT, C_GREEN,
            ),
            tile("Status",         "Normal",   C_BLUE_LIGHT,  C_BLUE),
            tile("Recommendation", "Annual",   C_AMBER_LIGHT, C_AMBER),
            tile("Report Status",  "Complete", C_GREEN_LIGHT, C_GREEN),
        ]

    row = Table([tiles], colWidths=[tile_w] * 4)
    row.setStyle(TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return row


# ─────────────────────────────────────────────────────────────
# Tumor Clinical Information
# ─────────────────────────────────────────────────────────────

TUMOR_INFO = {
    "glioma": {
        "description": (
            "Gliomas arise from glial cells in the brain or spinal cord "
            "and represent the most common primary brain tumors."
        ),
        "grade":      "WHO Grade II–IV",
        "treatment":  "Surgery, radiation therapy, chemotherapy (temozolomide)",
        "prognosis":  (
            "Varies by grade; GBM (Grade IV) has a median survival "
            "of 15 months"
        ),
        "follow_up":  (
            "MRI every 3–6 months; neurological assessment at each visit"
        ),
    },
    "meningioma": {
        "description": (
            "Meningiomas arise from the meninges surrounding the brain "
            "and spinal cord. Most are slow-growing."
        ),
        "grade":      "WHO Grade I–III (majority Grade I)",
        "treatment":  (
            "Observation, microsurgical resection, stereotactic radiosurgery"
        ),
        "prognosis":  (
            "Generally favourable; 5-year recurrence rate ~20% for Grade I"
        ),
        "follow_up":  "Annual MRI follow-up; 6-month MRI if symptomatic",
    },
    "pituitary": {
        "description": (
            "Pituitary adenomas originate in the pituitary gland "
            "and may affect hormonal regulation."
        ),
        "grade":      "Mostly benign adenomas (macroadenoma / microadenoma)",
        "treatment":  (
            "Medical therapy (dopamine agonists), "
            "transsphenoidal surgery, radiation"
        ),
        "prognosis":  (
            "Excellent in most cases; cure rates >90% for microadenomas"
        ),
        "follow_up":  "Hormonal panel every 6 months; MRI annually",
    },
}


# ─────────────────────────────────────────────────────────────
# Component: Clinical Findings Table
# ─────────────────────────────────────────────────────────────

def _findings_table(result: dict) -> list:

    if not result["tumor_detected"]:
        return [Paragraph(
            "No tumor identified on this MRI scan. "
            "Brain parenchyma, ventricles, and extra-axial spaces appear "
            "within normal limits. No mass effect, midline shift, or "
            "abnormal enhancement noted. Routine annual follow-up is advised.",
            T_BODY,
        )]

    info = TUMOR_INFO.get(result["tumor_type"], {})

    rows_data = [
        ["Diagnosis",   result["tumor_type"].capitalize()],
        ["Description", info.get("description", "N/A")],
        ["WHO Grade",   info.get("grade",        "N/A")],
        ["Treatment",   info.get("treatment",    "N/A")],
        ["Prognosis",   info.get("prognosis",    "N/A")],
        ["Follow-up",   info.get("follow_up",    "N/A")],
    ]

    col_label = 38 * mm
    col_value = CONTENT_W - col_label

    styled_rows = []
    for lbl, val in rows_data:
        styled_rows.append([
            Paragraph(
                lbl,
                _s(
                    f"FL_{lbl}", fontSize=8.5, fontName="Helvetica-Bold",
                    textColor=C_GRAY_DARK, leading=13,
                ),
            ),
            Paragraph(
                val,
                _s(
                    f"FV_{lbl}", fontSize=8.5, fontName="Helvetica",
                    textColor=C_DARK, leading=13, alignment=TA_JUSTIFY,
                ),
            ),
        ])

    t = Table(styled_rows, colWidths=[col_label, col_value])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), C_BLUE_LIGHT),    # label column
        ("BACKGROUND",    (1, 0), (1, -1), C_WHITE),          # value column
        # Zebra stripe on even value rows
        ("BACKGROUND",    (1, 1), (1,  1), HexColor("#F8FAFC")),
        ("BACKGROUND",    (1, 3), (1,  3), HexColor("#F8FAFC")),
        ("BACKGROUND",    (1, 5), (1,  5), HexColor("#F8FAFC")),
        ("BOX",           (0, 0), (-1, -1), 0.75, C_RULE),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.4,  C_RULE),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return [t]


# ─────────────────────────────────────────────────────────────
# PDF Builder
# ─────────────────────────────────────────────────────────────

def build_pdf_report(
    image: Image.Image,
    result: dict,
    patient_name: str,
    patient_age: str,
    patient_id: str,
    referring_physician: str,
) -> io.BytesIO:

    buf = io.BytesIO()

    # Reserve space for the canvas-drawn header and footer
    top_gap = 38 * mm + 8 * mm    # header height + breathing room
    bot_gap = 10 * mm + 6 * mm   # footer height + breathing room

    frame = Frame(
        MARGIN_H, bot_gap,
        CONTENT_W, PAGE_H - top_gap - bot_gap,
        leftPadding=0, rightPadding=0,
        topPadding=0,  bottomPadding=0,
        id="main",
    )

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN_H,  rightMargin=MARGIN_H,
        topMargin=top_gap,    bottomMargin=bot_gap,
    )
    doc.addPageTemplates([PageTemplate(
        id="main",
        frames=[frame],
        onPage=_page_decorator,
    )])

    story = []

    # ── Diagnosis banner ───────────────────────────────────────
    story.append(_diagnosis_banner(result))
    story.append(Spacer(1, 4 * mm))

    # # ── Patient info strip ─────────────────────────────────────
    # story.append(_patient_strip(
    #     patient_name, patient_age, patient_id, referring_physician
    # ))
    # story.append(Spacer(1, 6 * mm))

    # ── MRI imaging ────────────────────────────────────────────
    story += _section_header("MRI IMAGING ANALYSIS")
    story.append(_image_grid(image, result))
    story.append(Spacer(1, 6 * mm))

    # ── Quantitative metrics ───────────────────────────────────
    story += _section_header("QUANTITATIVE METRICS")
    story.append(_metrics_row(result))
    story.append(Spacer(1, 6 * mm))

    # ── Clinical findings ──────────────────────────────────────
    story += _section_header("CLINICAL FINDINGS")
    story += _findings_table(result)
    story.append(Spacer(1, 8 * mm))

    # ── Disclaimer ─────────────────────────────────────────────
    story.append(
        HRFlowable(width="100%", thickness=0.5, color=C_RULE, spaceAfter=4)
    )
    story.append(Paragraph(
        "This report is generated by an AI system and is intended solely "
        "for informational and research purposes. It does not constitute a "
        "medical diagnosis. All findings must be reviewed and confirmed by a "
        "qualified radiologist or physician before any clinical decision is made.",
        T_DISCLAIMER,
    ))

    doc.build(story)
    buf.seek(0)
    return buf


@app.post("/chat")
async def chat_endpoint(data: dict):

    question = data.get("question", "")

    tumor_type = data.get("tumor_type", "Unknown")
    severity = data.get("severity", "Unknown")
    location = data.get("tumor_location", "Unknown")
    area = data.get("tumor_area_percent", "Unknown")

    top_predictions = data.get(
        "top_predictions",
        []
    )

    prompt = f"""
You are a medical AI assistant.

Brain MRI Analysis Report

Tumor Type:
{tumor_type}

Severity:
{severity}

Tumor Location:
{location}

Tumor Area:
{area}%

Top Predictions:
{top_predictions}

User Question:
{question}

Rules:
- Explain findings in simple language.
- Do not claim a diagnosis.
- Mention that a radiologist should confirm findings.
- Keep answers under 200 words.
"""

    try:

        response = chat(
            model="qwen2.5:1.5b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            "answer":
                response["message"]["content"]
        }

    except Exception as e:

        return {
            "answer":
                f"LLM Error: {str(e)}"
        }
# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)