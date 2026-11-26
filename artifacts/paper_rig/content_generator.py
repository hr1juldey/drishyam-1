#!/usr/bin/env python3
"""Real-bill content generator (ICICI families) — READABLE training pages.
Family A: savings-account statement (clean 4-col transaction table).
Family B: credit-card e-statement (banner, masked cards, summary boxes, rewards, tx table).
Every page = f(recipe); ground truth JSON returned alongside. Drawn with PIL, never
generative. Page size 1241x1754 = A4 @150dpi (the rig's Document_color texture size).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json, random, string
from institutions import BANKS, ACC_LABELS, draw_date, inr_group, fmt_amount
from fin_docs import OMITTABLE_FIELDS, omitted_fields  # K5 omission registry (no cycle: fin_docs never imports this module)

W, H = 1241, 1754
FD = "/usr/share/fonts/truetype/liberation"
def _f(name, size):
    try: return ImageFont.truetype(f"{FD}/{name}", size)
    except Exception:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
F_REG, F_BOLD, F_ITALIC = _f("LiberationSans-Regular.ttf", 20), _f("LiberationSans-Bold.ttf", 20), None
_RUP = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def _rup(sz): return ImageFont.truetype(_RUP, sz)
def fonts():
    return {
        "logo":   _f("LiberationSans-Bold.ttf", 46),
        "label":  _f("LiberationSans-Regular.ttf", 20),
        "value":  _f("LiberationSans-Regular.ttf", 20),
        "th":     _f("LiberationSans-Regular.ttf", 22),
        "cell":   _f("LiberationSans-Regular.ttf", 20),
        "small":  _f("LiberationSans-Regular.ttf", 18),
        "title":  _f("LiberationSerif-BoldItalic.ttf", 40),
        "banner": _f("LiberationSans-Bold.ttf", 26),
        "big":    _f("LiberationSans-Bold.ttf", 44),
        "chip":   _f("LiberationSans-Regular.ttf", 18),
    }

NAVY   = (25, 60, 110)
ICICI_RED = (230, 90, 20)
MAROON = (150, 30, 40)
ORANGE = (243, 119, 53)
GRAY_BOX = (217, 217, 217)
BORDER = (90, 90, 90)

NAMES = ["MR SURYANARAYANA G", "MRS LAKSHMI DEVI R", "MR RAJESH KUMAR M", "MRS ANITHA S",
         "MR VIKRAM PRASAD N", "MRS DEEPA SHETTY", "MR ARUN KUMAR B", "MRS MEENA RAO",
         "MR SANJAY GUPTA", "MRS PRIYA NAIR"]
COMPANIES = ["THOMSON REUTERS INTERNATIONAL SERVICES PVT LTD", "INFOSYS LIMITED",
             "WIPRO TECHNOLOGIES LTD", "TATA CONSULTANCY SERVICES", "FLIPKART INTERNET PVT LTD",
             "AXIS BANK LTD", "HDFC BANK LTD"]
MONTHS3 = [m.upper() for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]]
CITY_LINES = ["124 AND 125 36 /2 DIVYA SREE TEHN POLIS YEMLUR OFF HALL OLD AIRPORT ROAD ,",
              "PLOT 44, KIADB INDUSTRIAL AREA, HOSUR ROAD,",
              "3RD FLOOR, TECHNOPOLIS, BANNERGHATTA ROAD,", "SECTOR 22, OUTER RING ROAD,"]
CITY_TAILS = ["KAR,BANGALORE-560037", "KAR,BANGALORE-560100", "KAR,BANGALORE-560076",
              "KAR,BANGALORE-560068"]
CARD_DESCRIPTORS = ["DISCOUNT ON FUEL PURCHASE CHENNAI IN", "AMAR SERVICE STATION BANGALORE IN",
                    "BANGALORE ONE -1 IN", "SRI HARI CHANAKESAVA BANGALORE IN",
                    "RK YU PAYMENTS PVT LTD 10 GURGAON IN", "HOUSE OF SPIRIT BANGALORE IN",
                    "AMRUTHA FOODS BANGALORE IN", "SWIGGY BANGALORE IN",
                    "UBER INDIA SYSTEMS ESHELL IN", "BIGCINEMAS BANGALORE IN",
                    "RELANCE FRESH BANGALORE IN", "NETFLIX.COM MUMBAI IN"]

def _digits(rng, n):
    return "".join(str(rng.randint(0, 10)) for _ in range(n))

def inr(x):
    s = f"{x:.2f}"
    whole, frac = s.split(".")
    whole = whole[::-1]
    parts = [whole[i:i+3] for i in range(0, len(whole), 3)]
    return ".".join([",".join(parts)[::-1], frac])

def mask_card(rng):
    return f"{rng.randint(3000, 5999)} XXXX XXXX {rng.randint(0, 9999):04d}"

def ddmmyyyy(rng, y0=2019, y1=2024):
    y = rng.randint(y0, y1); m = rng.randint(1, 12); d = rng.randint(1, 28)
    return f"{d:02d}/{m:02d}/{y}"

def draw_bank_logo(d, x, y, f, bank, credit=False):
    """bank = (name, brand_color, logo_style, tagline). Style varies by institution."""
    name, color, style, tag = bank
    if style == "oval_mark":
        d.ellipse([x, y, x + 52, y + 52], fill=color)
        d.ellipse([x + 14, y + 8, x + 38, y + 30], fill=(255, 255, 255))
    elif style == "square_mark":
        d.rectangle([x, y, x + 52, y + 52], fill=color)
        d.rectangle([x + 12, y + 12, x + 40, y + 40], fill=(255, 255, 255))
    elif style == "circle_mark":
        d.ellipse([x, y, x + 52, y + 52], outline=color, width=6)
        d.line([x + 8, y + 44, x + 44, y + 8], fill=color, width=5)
    else:                                            # wordmark: no glyph, colored bar
        d.rectangle([x, y, x + 14, y + 52], fill=color)
    d.text((x + 60, y + 2), name, font=f["logo"], fill=(25, 45, 80))
    if credit:
        d.text((x + 60, y + 56), tag, font=f["chip"], fill=(25, 45, 80))

# ---------------- Family A: savings account statement ----------------
def gen_statement(rng, seed=0):
    # K5 counterfactual omissions: registry maps (family, seed) -> fields to drop
    # (same contract as fin_docs.gen_salary). Old seeds not in the registry render
    # identical. from_date omission drops the whole period line (and to_date with it).
    omit = omitted_fields("savings", seed)
    f = fonts()
    bank = BANKS[seed % len(BANKS)]                  # LOIO-derivable: seed % len(POOL)
    acc_label = ACC_LABELS[min(rng.randint(0, len(ACC_LABELS)), len(ACC_LABELS) - 1)]
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    draw_bank_logo(d, 80, 100, f, bank)
    acct = _digits(rng, 12)
    d1, y1, m1, dd1 = draw_date(rng)
    d2 = d1
    if "account_number" not in omit:
        d.text((150, 330), acc_label, font=f["label"], fill=(0, 0, 0))
        d.text((375, 330), acct, font=f["value"], fill=(0, 0, 0))
    txn_label = ["Transaction date :", "Statement Period:", "Period :"][min(rng.randint(0, 3), 2)]
    if "from_date" not in omit:
        d.text((150, 385), txn_label, font=f["label"], fill=(0, 0, 0))
        d.text((375, 385), f"From {d1} To {d2}", font=f["value"], fill=(0, 0, 0))
    # table
    x0, x1 = 110, 1130
    cols = [x0, 260, 705, 985, 1105, x1]
    ty = 510
    d.rectangle([x0, ty, x1, ty + 34], outline=BORDER, width=1)
    head_opts = [["Date", "Description", "Amount", "Type"],
                 ["Txn Date", "Narration", "Amount(INR)", "Dr/Cr"],
                 ["Date", "Particulars", "Amount", "Indicator"]]
    heads = head_opts[min(rng.randint(0, len(head_opts)), len(head_opts) - 1)]
    for i, h in enumerate(heads):
        d.text((cols[i] + 8, ty + 7), h, font=f["th"], fill=(0, 0, 0))
        if i > 0:
            d.line([cols[i], ty, cols[i], ty + 34], fill=BORDER, width=1)
    rows = []
    y = ty + 34
    n = rng.randint(9, 15)
    bank_slug = bank[0][:3].upper()                  # brand-flavored ref prefix
    for i in range(n):
        desc_pool = [
            f"ECSRTNCHGS{rng.randint(1, 12):02d}{rng.randint(22, 24)}{rng.randint(100000, 999999)}_SR{rng.randint(9000000, 9999999)}{rng.randint(100, 999)}GST",
            f"NACH_AD_RTN_CHRG-INGENOT-{rng.randint(1, 12):02d}DEC23-00{rng.randint(600000, 699999)}-PAYU FINA",
            f"UPI/{rng.choice(['PAY','RCV'])}/{rng.randint(1000000000, 9999999999)}/{rng.choice(['SWIGGY','AMAZON','BLINKIT','ZOMATO','JIO','AMUL'])}",
            f"IMPS/P2A/{rng.randint(200000000000, 599999999999)}-{bank_slug}-{rng.randint(10, 99)}",
            f"POS {rng.randint(1000, 9999)}{rng.choice(['MYJIO','RELIANCE DIGITAL','DMART','MORE'])}/{rng.randint(1, 28):02d}{rng.choice(MONTHS3)}",
            f"ACH/D-{rng.choice(['BBDL','TNEB','BSNL','AIRTEL','LICI','HDFCL'])}/{rng.randint(1000000, 9999999)}",
            f"SALARY-{rng.randint(100000, 999999)}-{rng.choice(['CREDIT','NEFT'])}",
            f"ATM-WDL/{rng.randint(100000, 999999)}/-",
        ]
        desc = rng.choice(desc_pool)
        # realistic mixed magnitudes + occasional credits
        mag = rng.rand()
        amt = (round(rng.uniform(50, 500), 2) if mag < .45 else
               round(rng.uniform(500, 5000), 2) if mag < .85 else
               round(rng.uniform(5000, 40000), 2))
        typ = rng.choice(["DR", "DR", "DR", "DR", "CR", "DR"])
        rows.append({"date": d1, "description": desc, "amount": amt, "type": typ})
        hcell = 62 if len(desc) > 30 else 34
        d.rectangle([x0, y, x1, y + hcell], outline=BORDER, width=1)
        for ci in range(1, 4):
            d.line([cols[ci], y, cols[ci], y + hcell], fill=BORDER, width=1)
        d.text((cols[0] + 8, y + 7), d1, font=f["cell"], fill=(0, 0, 0))
        if len(desc) > 30:
            cut = desc[:30].rfind("_") if "_" in desc[:30] else 30
            d.text((cols[1] + 8, y + 4), desc[:cut + 1], font=f["cell"], fill=(0, 0, 0))
            d.text((cols[1] + 8, y + 26), desc[cut + 1:], font=f["cell"], fill=(0, 0, 0))
        else:
            d.text((cols[1] + 8, y + 7), desc, font=f["cell"], fill=(0, 0, 0))
        d.text((cols[2] + 8, y + 7), f"{amt:.2f}", font=f["cell"], fill=(0, 0, 0))
        d.text((cols[3] + 8, y + 7), typ, font=f["cell"], fill=(0, 0, 0))
        y += hcell
    d.text((80, y + 130), "This is a system-generated statement. Hence, it does not require any signature.",
           font=f["small"], fill=(0, 0, 0))
    d.text((1040, y + 130), "Page 1", font=f["small"], fill=(0, 0, 0))
    period_dropped = "from_date" in omit
    gt = {k: v for k, v in
          {"family": "bank_savings_statement", "bank": bank[0], "bank_index": seed % len(BANKS),
           "account_number": acct,
           "from_date": None if period_dropped else d1,
           "to_date": None if period_dropped else d2, "n_transactions": n,
           "transactions": rows, "page": 1}.items()
          if v is not None and k not in omit}
    return np.asarray(img, float), gt

# ---------------- Family B: credit card e-statement ----------------
def gen_card(rng, seed=0):
    # K5 counterfactual omissions: registry maps (family, seed) -> fields to drop
    # (same contract as fin_docs.gen_salary). Old seeds not in the registry render
    # identical. card_masked omission also drops card_number_row (no card digits
    # anywhere on the page); customer_name omission scrubs the name from the box
    # AND from the card-number row suffix.
    omit = omitted_fields("card", seed)
    f = fonts()
    bank = BANKS[seed % len(BANKS)]                  # LOIO-derivable
    _, color, _, tag = bank
    title = rng.choice(["CREDIT CARD E-STATEMENT", "Credit Card Statement",
                        "CARD E-STATEMENT"])
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((110, 55), title, font=f["title"], fill=MAROON)
    d.text((760, 60), bank[0], font=f["logo"], fill=(25, 45, 80))
    d.text((975, 112), tag, font=f["chip"], fill=(25, 45, 80))
    # banner
    bx0, by, bx1, bh = 30, 140, 1210, 130
    d.rectangle([bx0, by, bx1, by + bh], fill=color)
    banner_l1 = rng.choice(["Instant & easy Credit Card payments,",
                            "Pay your card bills instantly,",
                            "Manage your card on the go,"])
    banner_l2 = rng.choice(["now through secured UPI system",
                            "from anywhere, anytime",
                            "with one-tap payment options"])
    d.text((bx0 + 240, by + 28), banner_l1, font=f["banner"], fill=(255, 255, 255))
    d.text((bx0 + 240, by + 66), banner_l2, font=f["banner"], fill=(255, 255, 255))
    name = rng.choice(NAMES); company = rng.choice(COMPANIES)
    card = mask_card(rng)
    # customer box
    cx0, cy, cx1, ch = 30, 290, 1210, 175
    d.rectangle([cx0, cy, cx1, cy + ch], outline=BORDER, width=1)
    d.line([770, cy, 770, cy + ch], fill=BORDER, width=1)
    d.text((95, cy + 4), "Customer Name", font=f["chip"], fill=(60, 60, 60))
    d.text((40, cy + 30), name, font=f["value"], fill=(0, 0, 0))
    d.text((40, cy + 60), company, font=f["cell"], fill=(0, 0, 0))
    d.text((40, cy + 86), rng.choice(CITY_LINES), font=f["cell"], fill=(0, 0, 0))
    d.text((40, cy + 112), "BESIDE CGI", font=f["cell"], fill=(0, 0, 0))
    d.text((40, cy + 138), rng.choice(CITY_TAILS), font=f["cell"], fill=(0, 0, 0))
    d.text((855, cy + 4), "Card Account No", font=f["chip"], fill=(60, 60, 60))
    if "card_masked" not in omit:
        d.text((855, cy + 30), card, font=F_BOLD, fill=(0, 0, 0))
    if "customer_name" not in omit:
        d.text((40, cy + 30), name, font=f["value"], fill=(0, 0, 0))
    stmt_date = ddmmyyyy(rng); due_date = ddmmyyyy(rng)
    sy, sm = int(stmt_date[6:]), int(stmt_date[3:5])
    def stmt_month_date():
        m = sm - 1 if sm > 1 else 12
        y = sy if sm > 1 else sy - 1
        return f"{rng.randint(1, 28):02d}/{m:02d}/{y}"

    min_due = float(rng.choice([300.00, 450.00, 600.00, 800.00]))
    total_due = round(rng.uniform(800, 9000), 2)
    credit_limit = float(rng.choice([53000, 83000, 118000, 200000]))
    avail = round(credit_limit - rng.uniform(500, 6000), 2)
    # gray chips row
    chips = [(40, "Statement Date", stmt_date)]
    if "min_amount_due" not in omit:
        chips.append((270, "Minimum Amount Due", inr(min_due)))
    for (lx, lab, val) in chips:
        d.rectangle([lx, 480, lx + 220, 530], outline=BORDER, width=1)
        d.rectangle([lx, 480, lx + 220, 505], fill=GRAY_BOX)
        d.text((lx + 12, 484), lab, font=f["chip"], fill=(40, 40, 40))
        (d.text((lx + 12, 508), "₹ " + str(val), font=_rup(20), fill=(0, 0, 0))
         if "Due" in lab else d.text((lx + 12, 508), str(val), font=f["value"], fill=(0, 0, 0)))
    total_label = rng.choice(["Your Total Amount Due", "Total Amount Due", "Total Dues"])
    if "total_amount_due" not in omit:
        d.rectangle([900, 480, 1210, 590], fill=color)
        d.text((915, 486), total_label, font=f["chip"], fill=(255, 255, 255))
        d.text((915, 520), "₹ " + inr(total_due), font=_rup(34), fill=(255, 255, 255))
    # statement summary
    d.rectangle([40, 545, 860, 600], outline=BORDER, width=1)
    d.rectangle([40, 545, 150, 600], fill=color)
    d.text((50, 558), "Statement", font=f["chip"], fill=(255, 255, 255))
    d.text((50, 578), "Summary", font=f["chip"], fill=(255, 255, 255))
    prev = round(rng.uniform(2000, 6000), 2); purch = round(rng.uniform(3000, 9000), 2)
    pays = round(rng.uniform(2000, 6000), 2)
    for i, (lab, val) in enumerate([("Previous Balance", inr(prev)), ("Purchases/ Charges", inr(purch)),
                                    ("Cash Advances,", "0.00"), ("Payments/Credits", inr(pays))]):
        lx = 160 + i * 172
        d.text((lx, 552), lab, font=f["small"], fill=(40, 40, 40))
        d.text((lx, 575), val, font=f["value"], fill=(0, 0, 0))
    # credit summary
    d.rectangle([40, 610, 150, 660], fill=color)
    d.text((50, 622), "Credit", font=f["chip"], fill=(255, 255, 255))
    d.text((50, 642), "Summary", font=f["chip"], fill=(255, 255, 255))
    d.rectangle([150, 610, 860, 660], outline=BORDER, width=1)
    d.text((170, 616), "Credit Limit", font=f["small"], fill=(40, 40, 40))
    d.text((170, 638), inr(credit_limit), font=f["value"], fill=(0, 0, 0))
    d.text((500, 616), "Available Credit", font=f["small"], fill=(40, 40, 40))
    d.text((500, 638), inr(avail), font=f["value"], fill=(0, 0, 0))
    d.rectangle([870, 600, 1210, 660], outline=BORDER, width=1)
    if "due_date" not in omit:
        d.text((882, 606), f"Due Date :   {due_date}", font=f["chip"], fill=(0, 0, 0))
    d.text((882, 632), "Interest will be levied if Total Amount", font=f["small"], fill=(0, 0, 0))
    # rewards
    d.rectangle([40, 675, 150, 775], outline=BORDER, width=1)
    d.text((52, 690), bank[0].replace(" Bank", "")[:12], font=f["small"], fill=(0, 0, 0))
    d.text((52, 712), "Rewards", font=f["small"], fill=(0, 0, 0))
    pts = rng.randint(0, 300)
    d.rectangle([150, 675, 1210, 725], outline=BORDER, width=1)
    for i, (lab, val) in enumerate([("Opening Balance", "0"), ("Earned During the month", str(pts)),
                                    ("Redeemed during the month", "0"), ("Closing Balance", str(pts))]):
        lx = 165 + i * 260
        d.text((lx, 681), lab, font=f["small"], fill=(40, 40, 40))
        d.text((lx, 702), val, font=f["value"], fill=(0, 0, 0))
    d.rectangle([150, 725, 1210, 775], outline=BORDER, width=1)
    d.text((300, 731), "Points Earned", font=f["small"], fill=(40, 40, 40))
    d.text((300, 752), str(pts), font=f["value"], fill=(0, 0, 0))
    payback = _digits(rng, 19)
    d.text((700, 731), "Points Transferred to PAYBACK (Acc:" + payback + ")", font=f["small"], fill=(40, 40, 40))
    d.text((700, 752), str(pts), font=f["value"], fill=(0, 0, 0))
    d.text((40, 792), "For detailed summary of Reward Points, please click here", font=f["small"], fill=(30, 60, 160))
    d.text((40, 815), f"Convert your Big shopping bills into small, manageable EMI's! If any of the below transaction is highlighted, you may login to {bank[0]}",
           font=f["small"], fill=(170, 30, 30))
    d.text((40, 835), "Infinity and convert to EMI now ! TnC apply, visit www.icicibank.com/emoncall  for details.", font=f["small"], fill=(170, 30, 30))
    invoice = _digits(rng, 19)
    d.rectangle([40, 860, 1210, 900], fill=color)
    d.text((330, 868), "TRANSACTION DETAILS", font=f["banner"], fill=(255, 255, 255))
    if "invoice_no" not in omit:
        d.text((850, 872), f"INVOICE_NO :  {invoice}", font=f["small"], fill=(255, 255, 255))
    card4 = card.split()[0] + " XXXX XXXX " + card.split()[-1]
    if "card_masked" not in omit:
        d.text((40, 915), "Card Number :", font=f["value"], fill=(0, 0, 0))
        name_tail = "" if "customer_name" in omit else \
            f"  -  {name.replace('MR ', '').replace('MRS ', '')}"
        d.text((190, 915), f"{card4}{name_tail}", font=F_BOLD, fill=(0, 0, 0))
    # transactions table
    cols = [40, 148, 340, 672, 775, 850, 985, 1210]
    ty = 950
    heads = ["Date", "Ref. Number", "Transaction Details", "Reward", "Currency", "International", "Amount(In ₹)"]
    d.rectangle([40, ty, 1210, ty + 40], outline=BORDER, width=1)
    for i, h in enumerate(heads):
        d.text((cols[i] + 6, ty + 10), h, font=f["small"], fill=(0, 0, 0))
        if i > 0: d.line([cols[i], ty, cols[i], ty + 40], fill=BORDER, width=1)
    txs = []
    y = ty + 40
    n = rng.randint(6, 11)
    for i in range(n):
        ref = _digits(rng, 20)
        desc = rng.choice(CARD_DESCRIPTORS)
        pts_i = rng.choice([0.0, 0.0, 2.0, 3.0, 5.0])
        intl = "0.00"
        amt = round(rng.uniform(70, 1500), 2)
        cr = " CR" if rng.rand() < 0.18 else ""
        txs.append({"date": stmt_month_date(), "ref": ref, "detail": desc,
                    "reward_points": pts_i, "amount": amt, "credit": bool(cr)})
        hcell = 44
        d.rectangle([40, y, 1210, y + hcell], outline=BORDER, width=1)
        for ci in range(1, 7):
            d.line([cols[ci], y, cols[ci], y + hcell], fill=BORDER, width=1)
        d.text((cols[0] + 6, y + 13), txs[-1]["date"], font=f["small"], fill=(0, 0, 0))
        d.text((cols[1] + 6, y + 13), ref[:19], font=f["small"], fill=(0, 0, 0))
        d.text((cols[2] + 6, y + 13), desc[:30], font=f["small"], fill=(0, 0, 0))
        d.text((cols[3] + 6, y + 13), f"{pts_i:.2f}", font=f["small"], fill=(0, 0, 0))
        d.text((cols[5] + 6, y + 13), intl, font=f["small"], fill=(0, 0, 0))
        d.text((cols[6] + 6, y + 13), f"{amt:.2f}{cr}", font=f["small"], fill=(0, 0, 0))
        y += hcell
    gt = {k: v for k, v in
          {"family": "bank_card_estatement", "issuer": bank[0], "issuer_index": seed % len(BANKS),
           "customer_name": name, "company": company,
           "card_masked": card,
           "card_number_row": None if "card_masked" in omit else card4,
           "statement_date": stmt_date,
           "due_date": due_date, "min_amount_due": min_due, "total_amount_due": total_due,
           "credit_limit": credit_limit, "available_credit": avail, "payback_account": payback,
           "invoice_no": invoice, "previous_balance": prev, "purchases_charges": purch,
           "payments_credits": pays, "reward_points": pts, "n_transactions": n,
           "transactions": txs}.items()
          if v is not None and k not in omit}
    return np.asarray(img, float), gt

if __name__ == "__main__":
    import os
    OUT = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.RandomState(11)
    recipes = {}
    for fam, fn in (("savings", gen_statement), ("card", gen_card)):
        page, gt = fn(rng)
        Image.fromarray(np.clip(page, 0, 255).astype(np.uint8)).save(f"{OUT}/content_{fam}.png")
        recipes[fam] = gt
        with open(f"{OUT}/content_{fam}_gt.json", "w") as fp:
            json.dump(gt, fp, indent=1)
    print("PAGES DONE")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'transactions'}
                      for k, v in recipes.items()}, indent=1))
