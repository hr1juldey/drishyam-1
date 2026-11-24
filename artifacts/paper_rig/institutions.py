"""Institution/layout-grammar diversity pools + seed-derived profile draws.
Diversity doctrine (user): overfitting to bills-as-a-class is NOT fine either —
balance across institutions AND document types (electricity, telecom, salary,
CSV/tabular renders). Every profile draw is seed-derived (provenance law):
same seed -> same institution -> same pixels.

LOIO (leave-one-institution-out): profile index = seed % len(POOL). Bucket 7
(seed % 10 == 7) is the generalization holdout: training rows exclude it, the
generalization eval uses it. Measured, not assumed.
"""
import numpy as np

# (name, brand color, logo style, tagline/sub-brand)  — logo styles drawn by generator
BANKS = [("ICICI Bank", (230, 90, 20), "oval_mark", "Credit Cards"),
         ("HDFC BANK", (0, 60, 90), "square_mark", "We understand your world"),
         ("State Bank of India", (0, 80, 160), "circle_mark", "SBI"),
         ("Axis Bank", (150, 20, 30), "wordmark", "Badhti Ka Naam Zindagi"),
         ("Kotak Mahindra Bank", (190, 30, 45), "wordmark", "Safety First"),
         ("Punjab National Bank", (200, 130, 30), "circle_mark", "PNB"),
         ("Bank of Baroda", (170, 30, 90), "circle_mark", "Bob"),
         ("Canara Bank", (30, 90, 60), "wordmark", "Canara"),
         ("IndusInd Bank", (40, 40, 80), "wordmark", "IndusInd"),
         ("YES BANK", (60, 30, 100), "wordmark", "YES")]

DISCOMS = [("TANGEDCO", (60, 30, 120), "Chennai, Tamil Nadu"),
           ("MSEDCL", (30, 60, 140), "Mumbai, Maharashtra"),
           ("BSES Rajdhani Power", (150, 30, 30), "New Delhi"),
           ("Adani Electricity", (40, 110, 170), "Mumbai, Maharashtra"),
           ("CESC Limited", (30, 90, 170), "Kolkata, West Bengal"),
           ("Torrent Power", (230, 120, 30), "Ahmedabad, Gujarat"),
           ("TSSPDCL", (50, 110, 60), "Hyderabad, Telangana"),
           ("UGVCL", (90, 40, 130), "Mehsana, Gujarat")]

TELCO = [("Airtel", (230, 30, 30), "airtel"),
         ("Jio", (30, 60, 160), "Jio"),
         ("Vi (Vodafone Idea)", (170, 40, 150), "Vi"),
         ("BSNL", (20, 90, 50), "BSNL"),
         ("ACT Fibernet", (240, 130, 20), "ACT"),
         ("Tata Play Fiber", (60, 60, 70), "TATA")]

# GSTIN state codes (correct: first 2 digits of GSTIN)
STATES = [("24", "Gujarat"), ("27", "Maharashtra"), ("29", "Karnataka"),
          ("32", "Kerala"), ("33", "Tamil Nadu"), ("36", "Telangana"),
          ("37", "Andhra Pradesh"), ("07", "Delhi"), ("09", "Uttar Pradesh"),
          ("18", "West Bengal"), ("06", "Haryana"), ("23", "Madhya Pradesh")]

DATE_FMTS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%d-%b-%Y", "%Y-%m-%d"]

ACC_LABELS = ["Account Number:", "Account No:", "A/c No:", "Customer ID:"]
ACC_LABELS_CARD = ["Card Account No", "Card Number", "Card No"]

CONSUMER_LABELS = ["Consumer No.:", "Consumer Number:", "RR No.:", "Service Connection No.:"]
BILLPERIOD_LABELS = ["Billing Period:", "Bill Period:", "Consumption Period:"]
AMOUNT_DUE_LABELS = ["Amount Payable", "Net Amount Payable", "Total Amount Due", "Payable Amount"]
DUE_LABELS = ["Due Date :", "Payment Due Date :", "Last Date for Payment :"]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def draw_date(rng):
    """(printable string, y, m, d) — format varies by seed-derived style."""
    y, m, d = rng.randint(2019, 2025), rng.randint(1, 13), rng.randint(1, 29)
    fmt = DATE_FMTS[rng.randint(0, len(DATE_FMTS))]
    s = {"%d/%m/%Y": f"{d:02d}/{m:02d}/{y}", "%d-%m-%Y": f"{d:02d}-{m:02d}-{y}",
         "%d.%m.%Y": f"{d:02d}.{m:02d}.{y}", "%d %b %Y": f"{d:02d} {MONTHS[m-1]} {y}",
         "%d-%b-%Y": f"{d:02d}-{MONTHS[m-1]}-{y}", "%Y-%m-%d": f"{y}-{m:02d}-{d:02d}"}[fmt]
    return s, y, m, d

def inr_group(x):
    """Indian grouping: last 3, then 2-2-2 (12,34,567.89)."""
    s = f"{x:.2f}"
    whole, frac = s.split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.append(head[-2:]); head = head[:-2]
        if head:
            parts.append(head)
        whole = ",".join(reversed(parts)) + "," + tail
    return f"{whole}.{frac}"

def fmt_amount(rng, x):
    """Printed amount string: prefix style + grouping style vary (gt stores printed)."""
    pre = ["Rs ", "\u20b9 ", "INR ", ""][rng.randint(0, 4)]
    body = inr_group(x)
    return f"{pre}{body}"

def holdout(seed):
    """Generalization holdout bucket: exclude from training manifests."""
    return seed % 10 == 7
