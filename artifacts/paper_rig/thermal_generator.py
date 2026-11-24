#!/usr/bin/env python3
"""Thermal receipt family (real RAGA PVT LTD pharmacy bill as template) + the
smudge/legibility layer.

Design contract (main-pipeline shape):
  gen_thermal(rng, seed) -> (page_np, gt)
    gt["fields"][fid] = {"text", "bbox"}        <- drawn truth
    gt["degradation_plan"] = {fid: op}          <- chosen BEFORE rendering ops
  apply_degradation(page, gt, rng) -> (degraded_np, applied)
    applied[fid] = op actually applied          <- legibility is part of gold
Deterministic: same seed -> same pixels + same plan, so the factory regenerates
identical rows. Teacher labels from gt text + applied plan; never from pixels.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 384, 1150                      # 80mm thermal roll @ ~150dpi
FD = "/usr/share/fonts/truetype/liberation"
def _f(name, size):
    try: return ImageFont_load(f"{FD}/{name}", size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
def ImageFont_load(path, size):
    from PIL import ImageFont
    return ImageFont.truetype(path, size)

MONO, MONO_B = "LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf"
PAPER = (252, 252, 250)               # thermal paper, near-white
INK = (28, 28, 30)

SHOPS = [("RAGA PVT LTD", "S USMAN ROAD, T. NAGAR,", "CHENNAI, TAMIL NADU.", "044 25863222"),
         ("MEDPLUS PHARMACY", "12, GN CHETTY ROAD,", "CHENNAI, TAMIL NADU.", "044 28141555"),
         ("APOLLO PHARMACY", "45, KILPAUK GARDEN ROAD,", "CHENNAI, TAMIL NADU.", "044 26431122"),
         ("LIFELINE MEDICALS", "3, BLUMENTHALIT STREET,", "CHENNAI, TAMIL NADU.", "044 25224418"),
         ("SUBIKSHA PHARMA", "77, BAZULLAH ROAD,", "CHENNAI, TAMIL NADU.", "044 28152233")]
ITEMS = [("Alternagel", 200.0), ("Bepanthen", 560.0), ("Crocin 500", 32.0), ("Dolo 650", 34.0),
         ("Benadryl DR", 118.0), ("Digene Gel", 92.0), ("Eno Fruit Salt", 58.0),
         ("Volini Spray", 210.0), ("Betadine 100ml", 148.0), ("ORS Sachet", 22.0),
         ("Calpol Syrup", 68.0), ("Zyrtec 10mg", 96.0), ("Soframycin", 76.0),
         ("Electral Powder", 24.0), ("Moov Spray", 185.0), ("Hansaplast", 45.0)]
CUSTOMERS = ["David Stores", "Walk-in", "Mr Ramesh", "Mrs Sujatha", "Sri Medical Stores", ""]

def _gst(rng):
    return "33" + "".join(chr(65 + rng.randint(0, 26)) for _ in range(5)) + \
           "P" + "".join(chr(65 + rng.randint(0, 26)) for _ in range(2)) + \
           f"{rng.randint(1, 9999):04d}" + "F" + "1Z" + chr(65 + rng.randint(0, 26))

def gen_thermal(rng, seed):
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    f_head = _f(MONO_B, 22); f_body = _f(MONO, 16); f_small = _f(MONO, 14)
    fields = {}

    def dotted(yy):
        d.line([14, yy, 370, yy], fill=INK, width=2)
        for x in range(14, 370, 6):
            d.line([x, yy, x + 2, yy], fill=PAPER, width=2)

    def field(fid, text, xy, font, align=None, fill=INK):
        """draw + register bbox. align: None=left, 'center', 'right' (xy = anchor x)"""
        wpx = d.textlength(text, font=font)
        x = xy[0]
        if align == "center": x = xy[0] - wpx / 2
        elif align == "right": x = xy[0] - wpx
        y = xy[1]
        d.text((x, y), text, font=font, fill=fill)
        bb = d.textbbox((x, y), text, font=font)
        fields[fid] = {"text": text, "bbox": [int(v) for v in bb]}
        return bb[3]

    name, addr, city, phone = SHOPS[rng.randint(0, len(SHOPS))]
    gstin = _gst(rng)
    y = 14
    field("shop_name", name, (W // 2, y), f_head, "center"); y += 32
    field("address", addr, (W // 2, y), f_body, "center"); y += 22
    field("city", city, (W // 2, y), f_body, "center"); y += 22
    field("phone", f"PHONE : {phone}", (W // 2, y), f_body, "center"); y += 24
    field("gstin", f"GSTIN : {gstin}", (W // 2, y), f_body, "center"); y += 30
    field("doc_type", "Retail Invoice", (W // 2, y), f_head, "center"); y += 34
    dd, mm, yy = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2019, 2024)
    hh, mi = rng.randint(9, 21), rng.randint(0, 59)
    date_s = f"Date : {dd:02d}/{mm:02d}/{yy}, {hh:02d}:{mi:02d} PM"
    field("date", date_s, (14, y), f_body); y += 26
    cust = CUSTOMERS[rng.randint(0, len(CUSTOMERS))]
    y += 8
    if cust:
        field("customer", cust, (14, y), f_body); y += 26
    bill_no = f"SR{rng.randint(1, 999)}"
    field("bill_no", f"Bill No: {bill_no}", (14, y), f_body); y += 24
    pay = rng.choice(["Cash", "Card", "UPI"])
    field("payment_mode", f"Payment Mode: {pay}", (14, y), f_body); y += 24
    field("dr_ref", f"DR Ref : {rng.randint(1, 9)}", (14, y), f_body); y += 14
    dotted(y); y += 18

    adv = d.textlength("0", font=f_body)          # monospace advance
    col_qty, col_amt = 258, 370
    field("th_item", "Item", (14, y), f_body)
    field("th_qty", "Qty", (col_qty, y), f_body, "right")
    field("th_amt", "Amt", (col_amt, y), f_body, "right")
    y += 26
    dotted(y); y += 10
    n = rng.randint(3, 7)
    picks = [ITEMS[i] for i in rng.choice(len(ITEMS), size=n, replace=False)]
    subtotal, items = 0.0, []
    for i, (iname, base_p) in enumerate(picks):
        qty = int(rng.randint(1, 4))
        amt = round(base_p * qty, 2)
        subtotal += amt
        items.append({"item": iname, "qty": qty, "amount": amt})
        field(f"item_{i}_name", iname, (14, y), f_body)
        field(f"item_{i}_qty", str(qty), (col_qty, y), f_body, "right")
        field(f"item_{i}_amt", f"{amt:.2f}", (col_amt, y), f_body, "right")
        y += 24
    field("subtotal_label", "Sub Total", (14, y), f_body)
    field("subtotal_qty", str(sum(i["qty"] for i in items)), (col_qty, y), f_body, "right")
    field("subtotal_amt", f"{subtotal:.2f}", (col_amt, y), f_body, "right")
    y += 28
    discount = round(rng.choice([0, 26.0, 38.1, 51.5]), 2)
    if discount > 0:
        field("discount_label", "(-) Discount", (col_qty - 60, y), f_body)
        field("discount_amt", f"{discount:.2f}", (col_amt, y), f_body, "right")
        y += 24
    taxable = subtotal - discount
    slab = rng.choice([14.0, 2.5, 5.0])
    cgst = round(taxable * slab / 200, 2); sgst = cgst
    field(f"cgst_label", f"CGST @ {slab:.2f}%", (col_qty - 76, y), f_small)
    field(f"cgst_amt", f"{cgst:.2f}", (col_amt, y), f_small, "right"); y += 22
    field(f"sgst_label", f"SGST @ {slab:.2f}%", (col_qty - 76, y), f_small)
    field(f"sgst_amt", f"{sgst:.2f}", (col_amt, y), f_small, "right"); y += 26
    total = round(taxable + cgst + sgst, 2)
    dotted(y); y += 12
    field("total_label", "TOTAL", (14, y), f_body)
    field("total_amt", f"Rs {total:.2f}", (col_amt, y), f_body, "right"); y += 28
    field("cash_line", f"Cash :", (14, y), f_body)
    field("cash_amt", f"Rs {total:.2f}", (col_amt, y), f_body, "right"); y += 24
    field("tendered_label", "Cash tendered:", (14, y), f_body)
    field("tendered_amt", f"Rs {total:.2f}", (col_amt, y), f_body, "right"); y += 30
    field("eoe", "E & O.E", (370, y), f_small, "right")

    gt = {"family": "thermal_pharmacy_retail", "page_size": [W, H], "seed": int(seed),
          "shop": name, "gstin": gstin, "date": date_s, "customer": cust,
          "bill_no": bill_no, "payment_mode": pay,
          "items": items, "subtotal": round(subtotal, 2), "discount": discount,
          "gst_slab_pct": slab, "cgst": cgst, "sgst": sgst, "total": total,
          "fields": fields, "degradation_plan": {}}
    return np.asarray(img, np.float32), gt

# ---------------- degradation ops (deterministic, recipe-recorded) ----------------
def _crop_box(arr, bbox, pad=3):
    x0, y0, x1, y1 = bbox
    return max(0, x0 - pad), max(0, y0 - pad), min(arr.shape[1], x1 + pad), min(arr.shape[0], y1 + pad)

def _fade(arr, bbox, keep=0.35):
    x0, y0, x1, y1 = _crop_box(arr, bbox)
    reg = arr[y0:y1, x0:x1]
    arr[y0:y1, x0:x1] = 255 - (255 - reg) * keep
    return arr

def _smudge(arr, bbox, rng, heavy=True):
    """ink smear: horizontal text smear + gray-brown blob. heavy -> unreadable."""
    x0, y0, x1, y1 = _crop_box(arr, bbox, pad=6)
    reg = Image.fromarray(np.clip(arr[y0:y1, x0:x1], 0, 255).astype(np.uint8))
    reg = reg.filter(ImageFilter.GaussianBlur(2.2 if heavy else 1.2)).filter(ImageFilter.BoxBlur(4 if heavy else 2))
    sm = np.asarray(reg, np.float32)
    h, w = sm.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * (0.3 + 0.4 * rng.rand()), h * (0.35 + 0.3 * rng.rand())
    blob = np.exp(-((xx - cx) ** 2 / (2 * (w * 0.32) ** 2) + (yy - cy) ** 2 / (2 * (h * 0.42) ** 2)))
    blob = blob * (0.62 if heavy else 0.34) * (0.75 + 0.25 * rng.rand(h, w))
    smear_col = np.array([88, 84, 82], np.float32)          # gray grease smear
    out = sm * (1 - blob[..., None]) + smear_col * blob[..., None]
    arr[y0:y1, x0:x1] = out
    return arr

def _thermal_band(arr, y0, y1, keep=0.30):
    """dying printhead: whole horizontal band prints light gray."""
    y0 = max(0, y0); y1 = min(arr.shape[0], y1)
    arr[y0:y1] = 255 - (255 - arr[y0:y1]) * keep
    return arr

def _crease(arr, y, strength=1.0):
    y0, y1 = max(0, y - 10), min(arr.shape[0], y + 10)
    yy = np.arange(y0, y1, dtype=np.float32)
    prof = 1 - strength * 0.55 * np.exp(-((yy - y) ** 2) / 18)
    arr[y0:y1] *= prof[:, None, None]
    line = int(np.clip(y, 1, arr.shape[0] - 2))
    arr[line - 1:line + 2] *= 0.45
    return arr

def _tear(arr, bbox, rng):
    x0, y0, x1, y1 = _crop_box(arr, bbox, pad=8)
    h, w = y1 - y0, x1 - x0
    noise = rng.rand(h, w)
    rag = np.asarray(Image.fromarray((noise * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(3)), np.float32) / 255.0
    mask = rag > 0.5
    arr[y0:y1, x0:x1][mask] = np.array(PAPER, np.float32)
    return arr

def apply_degradation(page, gt, rng):
    """apply gt['degradation_plan'] {fid: op}; op in
    {fade, smudge_light, smudge_heavy, thermal_band, crease, tear}.
    Returns (degraded_page, applied {fid: op}). thermal_band targets the field's row band."""
    arr = page.copy()
    applied = {}
    for fid, op in gt["degradation_plan"].items():
        if fid not in gt["fields"]:
            continue
        bbox = gt["fields"][fid]["bbox"]
        if op == "fade":
            arr = _fade(arr, bbox, keep=rng.uniform(0.25, 0.45))
        elif op == "smudge_light":
            arr = _smudge(arr, bbox, rng, heavy=False)
        elif op == "smudge_heavy":
            arr = _smudge(arr, bbox, rng, heavy=True)
        elif op == "thermal_band":
            arr = _thermal_band(arr, bbox[1] - 6, bbox[3] + 6, keep=rng.uniform(0.22, 0.38))
        elif op == "crease":
            arr = _crease(arr, (bbox[1] + bbox[3]) // 2)
        elif op == "tear":
            arr = _tear(arr, bbox, rng)
        else:
            continue
        applied[fid] = op
    gt = dict(gt); gt["legibility_applied"] = applied
    return arr, applied

def choose_plan(rng, n_ops=3, candidate_fids=None):
    """degradation plan picker for the factory: which fields suffer, how."""
    ops = ["fade", "smudge_light", "smudge_heavy", "thermal_band", "crease", "tear"]
    return {fid: ops[rng.randint(0, len(ops))] for fid in (candidate_fids or [])[:n_ops]}

if __name__ == "__main__":
    import os, json, sys
    OUT = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.RandomState(5)
    seed = 12345
    page, gt = gen_thermal(rng, seed)
    Image.fromarray(np.clip(page, 0, 255).astype(np.uint8)).save(f"{OUT}/thermal_clean.png")
    # demo plan: hit decision fields — this is what the factory will randomize
    gt["degradation_plan"] = {"total_amt": "smudge_heavy", "date": "fade",
                              "item_0_name": "smudge_light", "gstin": "thermal_band",
                              "bill_no": "crease"}
    deg, applied = apply_degradation(page, gt, np.random.RandomState(seed + 1))
    Image.fromarray(np.clip(deg, 0, 255).astype(np.uint8)).save(f"{OUT}/thermal_degraded.png")
    json.dump(gt, open(f"{OUT}/thermal_gt.json", "w"), indent=1)
    print("PLAN   ", json.dumps(gt["degradation_plan"]))
    print("APPLIED", json.dumps(applied))
    # matched-scale comparison with the REAL bill
    real_path = "/home/riju279/.hermes/cache/documents/native-inputs/521629eff30096f7f5a539da28f714c7d556527025028dfa7889d5149451444a/0129889b-76a0-4e78-a1ce-0a5266c15f01.png"
    real = Image.open(real_path).convert("RGB")
    Wc = 420
    rh = int(real.height * Wc / real.width); mh = int(H * Wc / W)
    strip = Image.new("RGB", (Wc * 3 + 32, max(rh, mh)), (25, 25, 28))
    strip.paste(real.resize((Wc, rh)), (0, 0))
    strip.paste(Image.fromarray(np.clip(page, 0, 255).astype(np.uint8)).resize((Wc, mh)), (Wc + 16, 0))
    strip.paste(Image.fromarray(np.clip(deg, 0, 255).astype(np.uint8)).resize((Wc, mh)), (2 * Wc + 32, 0))
    strip.save(f"{OUT}/thermal_compare.png")
    print("SAVED thermal_compare.png [REAL | mine clean | mine degraded]")
