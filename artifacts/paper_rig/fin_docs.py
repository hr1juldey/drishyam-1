"""New financial-document families (diversity directive): electricity bills
(discoms), telecom invoices (telcos), salary slips, and CSV/Excel-style tabular
renders. Same contract as the other generators: deterministic (rng, seed) ->
(PIL-page np float, gt dict with PRINTED strings). Structure varies, arithmetic
is internally consistent, answers are image-only. A4 portrait 1241x1754.
Family tags: electricity_bill / telecom_invoice / salary_slip / table_sheet."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from institutions import (DISCOMS, TELCO, draw_date, fmt_amount, inr_group,
                          CONSUMER_LABELS, BILLPERIOD_LABELS, AMOUNT_DUE_LABELS,
                          DUE_LABELS, STATES)

W, H = 1241, 1754
FD = "/usr/share/fonts/truetype/liberation"
def _f(name, size):
    try: return ImageFont.truetype(f"{FD}/{name}", size)
    except Exception:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
def fonts():
    return {"logo": _f("LiberationSans-Bold.ttf", 40), "label": _f("LiberationSans-Regular.ttf", 20),
            "value": _f("LiberationSans-Regular.ttf", 20), "th": _f("LiberationSans-Regular.ttf", 22),
            "cell": _f("LiberationSans-Regular.ttf", 20), "small": _f("LiberationSans-Regular.ttf", 18),
            "title": _f("LiberationSerif-BoldItalic.ttf", 40), "banner": _f("LiberationSans-Bold.ttf", 26),
            "big": _f("LiberationSans-Bold.ttf", 44)}
BORDER = (90, 90, 90)
COMPANIES = ["THOMSON REUTERS INTERNATIONAL SERVICES PVT LTD", "INFOSYS LIMITED",
             "WIPRO TECHNOLOGIES LTD", "TATA CONSULTANCY SERVICES", "FLIPKART INTERNET PVT LTD",
             "ZOHAR TECHNOLOGIES PVT LTD", "MERCEDES BENZ R&D INDIA", "AMAZON DEVELOPMENT CENTRE"]
EMPLOYEES = ["MR RAHUL SHARMA", "MRS KAVITHA RAO", "MR ABHISHEK JHA", "MS FATIMA SHEIKH",
             "MR DEBDATTA SEN", "MRS MEERA IYER", "MR PRADEEP KUMAR T", "MS ANANYA GHOSH"]

def _digits(rng, n):
    return "".join(str(rng.randint(0, 10)) for _ in range(n))

# ---------------- electricity bill ----------------
def gen_electricity(rng, seed=0):
    f = fonts()
    discom = DISCOMS[seed % len(DISCOMS)]            # LOIO-derivable
    name, color, city = discom
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, 120, 100], fill=color)      # brand block logo
    d.text((44, 50), name[0], font=f["logo"], fill=(255, 255, 255))
    d.text((130, 42), name, font=f["banner"], fill=(0, 0, 0))
    d.text((130, 80), city, font=f["small"], fill=(60, 60, 60))
    d.text((880, 55), "ELECTRICITY BILL", font=f["banner"], fill=color)
    cons = _digits(rng, rng.choice([10, 11, 12]))
    cons_label = CONSUMER_LABELS[rng.randint(0, len(CONSUMER_LABELS))]
    period_label = BILLPERIOD_LABELS[rng.randint(0, len(BILLPERIOD_LABELS))]
    due_label = DUE_LABELS[rng.randint(0, len(DUE_LABELS))]
    bill_date, y, m, dd = draw_date(rng)
    units = rng.randint(80, 920)
    slabs = [(100, 3.5), (200, 5.2), (400, 7.8), (10**9, 9.6)]     # slab arithmetic
    rem, prev_tier, charge = units, 0, 0.0
    for lim, rate in slabs:
        take = min(rem, lim - prev_tier)
        if take > 0:
            charge += take * rate; rem -= take; prev_tier = lim
    fixed = rng.choice([70.0, 90.0, 110.0, 125.0])
    duty = round(units * 0.06, 2)
    subtotal = round(charge + fixed + duty, 2)
    gst = round(subtotal * 0.05, 2)
    total = round(subtotal + gst, 2)
    due_off = rng.randint(10, 25)
    due, _, _, _ = draw_date(rng)
    pt = fmt_amount(rng, total)                      # ONE draw: box and gt must match
    rows = [("Consumer No.", cons), ("Meter No.", _digits(rng, 8)),
            (period_label.rstrip(":"), f"{bill_date} - {due}"),
            ("Units Consumed", str(units)),
            ("Energy Charges", fmt_amount(rng, round(charge, 2))),
            ("Fixed Charges", fmt_amount(rng, fixed)),
            ("Electricity Duty", fmt_amount(rng, duty)),
            ("GST (5%)", fmt_amount(rng, gst))]
    yy = 190
    for lab, val in rows:
        d.text((110, yy), lab, font=f["label"], fill=(40, 40, 40))
        d.text((480, yy), str(val), font=f["value"], fill=(0, 0, 0))
        d.line([100, yy + 34, 1140, yy + 34], fill=(200, 200, 200), width=1)
        yy += 52
    d.rectangle([100, yy + 30, 760, yy + 100], fill=color)
    d.text((120, yy + 42), AMOUNT_DUE_LABELS[rng.randint(0, len(AMOUNT_DUE_LABELS))],
           font=f["banner"], fill=(255, 255, 255))
    d.text((120, yy + 72), pt, font=_f("LiberationSans-Bold.ttf", 30),
           fill=(255, 255, 255))
    d.text((800, yy + 44), due_label + f" {due}", font=f["label"], fill=(0, 0, 0))
    d.text((800, yy + 76), f"Bill Date: {bill_date}", font=f["small"], fill=(0, 0, 0))
    d.text((110, H - 120), "This is a computer generated bill and does not require signature.",
           font=f["small"], fill=(60, 60, 60))
    gt = {"family": "electricity_bill", "provider": name, "provider_index": seed % len(DISCOMS),
          "consumer_no": cons, "meter_no": rows[1][1], "bill_date": bill_date,
          "due_date": due, "units_consumed": units, "energy_charges": round(charge, 2),
          "fixed_charges": fixed, "electricity_duty": duty, "gst": gst,
          "total_amount": total, "printed_total": pt}
    return np.asarray(img, float), gt

# ---------------- telecom invoice ----------------
def gen_telecom(rng, seed=0):
    f = fonts()
    telco = TELCO[seed % len(TELCO)]                 # LOIO-derivable
    name, color, brand = telco
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((80, 50), brand, font=_f("LiberationSans-Bold.ttf", 56), fill=color)
    d.text((820, 60), "TELECOM INVOICE", font=f["banner"], fill=(0, 0, 0))
    mobile = f"+91 {_digits(rng, 5)} {_digits(rng, 5)}"
    inv_no = _digits(rng, 9)
    bill_date, y, m, dd = draw_date(rng)
    plan_amt = float(rng.choice([199.0, 239.0, 299.0, 399.0, 549.0, 666.0, 999.0]))
    plan_name = rng.choice(["Unlimited Data Plan", "Family Postpaid Plan", "Work-from-home Plan",
                            "Digital Pack", "Premium Unlimited"])
    data_gb = rng.choice([1.5, 2.0, 3.0, 5.0, 8.0])
    extra = round(rng.uniform(0, 220), 2)
    subtotal = plan_amt + extra
    gst = round(subtotal * 0.18, 2)
    total = round(subtotal + gst, 2)
    due, _, _, _ = draw_date(rng)
    pt = fmt_amount(rng, total)                      # ONE draw: box and gt must match
    rows = [("Mobile Number", mobile), ("Invoice No.", inv_no),
            ("Plan", f"{plan_name} ({fmt_amount(rng, plan_amt)})"),
            ("Data Used", f"{data_gb:.1f} GB"), ("Additional Charges", fmt_amount(rng, extra)),
            ("Sub Total", fmt_amount(rng, subtotal)), ("GST 18%", fmt_amount(rng, gst)),
            ("Bill Date", bill_date)]
    yy = 200
    for lab, val in rows:
        d.text((110, yy), lab, font=f["label"], fill=(40, 40, 40))
        d.text((480, yy), str(val), font=f["value"], fill=(0, 0, 0))
        d.line([100, yy + 34, 1140, yy + 34], fill=(200, 200, 200), width=1)
        yy += 52
    d.rectangle([100, yy + 30, 760, yy + 100], fill=color)
    d.text((120, yy + 42), AMOUNT_DUE_LABELS[rng.randint(0, len(AMOUNT_DUE_LABELS))],
           font=f["banner"], fill=(255, 255, 255))
    d.text((120, yy + 72), pt, font=_f("LiberationSans-Bold.ttf", 30),
           fill=(255, 255, 255))
    d.text((800, yy + 44), DUE_LABELS[rng.randint(0, len(DUE_LABELS))] + f" {due}",
           font=f["label"], fill=(0, 0, 0))
    d.text((110, H - 120), "GST invoice for telecom services. Retain for your records.",
           font=f["small"], fill=(60, 60, 60))
    gt = {"family": "telecom_invoice", "provider": name, "provider_index": seed % len(TELCO),
          "mobile": mobile, "invoice_no": inv_no, "bill_date": bill_date, "due_date": due,
          "plan_name": plan_name, "plan_amount": plan_amt, "data_used_gb": data_gb,
          "subtotal": subtotal, "gst": gst, "total_amount": total,
          "printed_total": pt}
    return np.asarray(img, float), gt

# ---------------- salary slip ----------------
def gen_salary(rng, seed=0):
    f = fonts()
    company = COMPANIES[seed % len(COMPANIES)]       # LOIO-derivable
    emp = EMPLOYEES[rng.randint(0, len(EMPLOYEES))]
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, W - 30, 110], fill=(240, 240, 245))
    d.text((50, 50), company, font=f["banner"], fill=(0, 0, 0))
    d.text((900, 55), "SALARY SLIP", font=f["banner"], fill=(120, 30, 30))
    d.text((50, 80), rng.choice(["PAN India Operations", "Corporate Office", "Technology Division"]),
           font=f["small"], fill=(60, 60, 60))
    m = rng.randint(1, 13)
    period = rng.choice(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]) \
        if False else ["January","February","March","April","May","June","July","August",
                       "September","October","November","December"][m - 1 if m < 12 else 11]
    year = rng.randint(2022, 2025)
    basic = float(rng.choice([32000, 41500, 50000, 58500, 67000, 75000]))
    hra = round(basic * 0.40, 2)
    conv = float(rng.choice([1600.0, 2000.0, 2400.0]))
    medical = float(rng.choice([750.0, 1000.0, 1250.0]))
    pf = round(basic * 0.12, 2)
    tax = round(basic * rng.uniform(0.02, 0.12), 2)
    gross = round(basic + hra + conv + medical, 2)
    net = round(gross - pf - tax, 2)
    d.text((50, 150), f"Employee Name: {emp}", font=f["value"], fill=(0, 0, 0))
    d.text((700, 150), f"Employee ID: EMP{_digits(rng, 5)}", font=f["value"], fill=(0, 0, 0))
    d.text((50, 190), f"Salary Period: {period} {year}", font=f["value"], fill=(0, 0, 0))
    d.text((700, 190), f"Designation: {rng.choice(['ENGINEER','ANALYST','MANAGER','LEAD','EXECUTIVE'])}",
           font=f["value"], fill=(0, 0, 0))
    ty = 260
    d.rectangle([50, ty, 620, ty + 40], fill=(210, 220, 235))
    d.text((62, ty + 8), "EARNINGS", font=f["th"], fill=(0, 0, 0))
    d.rectangle([640, ty, 1190, ty + 40], fill=(235, 210, 210))
    d.text((652, ty + 8), "DEDUCTIONS", font=f["th"], fill=(0, 0, 0))
    yy = ty + 46
    for lab, val in [("Basic", basic), ("House Rent Allowance", hra),
                     ("Conveyance", conv), ("Medical Allowance", medical),
                     ("Gross Earnings", gross)]:
        d.text((62, yy), lab, font=f["cell"], fill=(0, 0, 0))
        d.text((520, yy), fmt_amount(rng, val), font=f["cell"], fill=(0, 0, 0))
        d.line([50, yy + 34, 620, yy + 34], fill=(200, 200, 200), width=1)
        yy += 50
    yy2 = ty + 46
    for lab, val in [("Provident Fund", pf), ("Income Tax (TDS)", tax),
                     ("Net Pay", net)]:
        d.text((652, yy2), lab, font=f["cell"], fill=(0, 0, 0))
        d.text((1110, yy2), fmt_amount(rng, val), font=f["cell"], fill=(0, 0, 0))
        d.line([640, yy2 + 34, 1190, yy2 + 34], fill=(200, 200, 200), width=1)
        yy2 += 50
    d.rectangle([340, 620, 900, 700], fill=(25, 60, 110))
    d.text((360, 632), rng.choice(["Net Pay", "Net Salary", "Take Home Pay"]),
           font=f["banner"], fill=(255, 255, 255))
    pn = fmt_amount(rng, net)                        # ONE draw: box and gt must match
    d.text((360, 664), pn, font=_f("LiberationSans-Bold.ttf", 30),
           fill=(255, 255, 255))
    d.text((50, H - 120), "This is a system generated payslip and does not require a signature.",
           font=f["small"], fill=(60, 60, 60))
    gt = {"family": "salary_slip", "employer": company, "employer_index": seed % len(COMPANIES),
          "employee_name": emp, "employee_id": f"EMP{_digits(rng, 5)}", "period": f"{period} {year}",
          "basic": basic, "hra": hra, "conveyance": conv, "medical": medical,
          "pf": pf, "tax": tax, "gross_earnings": gross, "net_pay": net,
          "printed_net": pn}
    return np.asarray(img, float), gt

# ---------------- CSV/Excel-style tabular render ----------------
def gen_table(rng, seed=0):
    """Spreadsheet-like render: grid lines, header row, plain columns. The
    'don't-look-like-a-bill' family — tabular data as a document."""
    f = fonts()
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    sheets = [("transaction_ledger", ["Date", "Ref", "Description", "Debit", "Credit", "Balance"]),
              ("portfolio_holdings", ["Scrip", "ISIN", "Qty", "Avg Cost", "LTP", "P&L"]),
              ("expense_report", ["Date", "Category", "Vendor", "Amount", "GST", "Notes"]),
              ("loan_schedule", ["Month", "EMI", "Principal", "Interest", "Balance", "Paid %"])]
    kind, cols = sheets[seed % len(sheets)]
    nc = len(cols)
    widths = [40 + int(W * w) for w in ([0.16, 0.22, 0.30, 0.16, 0.16, 0.16] if nc == 6
                                        else [0.18, 0.26, 0.28, 0.28])]
    col_x = [40]
    for w in widths[:nc]:
        col_x.append(min(col_x[-1] + int(w), W - 40))
    col_x[-1] = W - 40
    d.text((40, 40), kind.replace("_", " ").title(), font=f["banner"], fill=(40, 40, 90))
    d.text((950, 40), f"sheet: {kind}.csv", font=f["small"], fill=(120, 120, 120))
    ty = 100
    rh = 44
    n = rng.randint(10, 16)
    bal = 0.0
    scrips = [("RELIANCE", "INE002A01018"), ("TCS", "INE467B01029"), ("INFY", "INE009A01021"),
              ("HDFCBANK", "INE040A01034"), ("ITC", "INE154A01025"), ("TATAMOTORS", "INE155A01022")]
    vendors = ["UBER INDIA", "OLLA CABS", "TAJ HOTELS", "PVR CINEMAS", "BLINKIT", "NAMMA CAFE"]
    cats = ["TRAVEL", "MEALS", "STAY", "SOFTWARE", "OFFICE"]
    data = []
    for i in range(n):
        dt, y_, m_, dd_ = draw_date(rng)
        if kind == "transaction_ledger":
            debit = round(rng.uniform(0, 3000), 2) if rng.rand() < .6 else 0.0
            credit = round(rng.uniform(0, 4000), 2) if debit == 0 else 0.0
            bal = round(bal + credit - debit, 2)
            data.append([dt, _digits(rng, 8), rng.choice(vendors),
                         f"{debit:.2f}" if debit else "-", f"{credit:.2f}" if credit else "-",
                         inr_group(bal)])
        elif kind == "portfolio_holdings":
            s, isin = scrips[rng.randint(0, len(scrips))]
            qty = rng.randint(5, 200)
            avg = round(rng.uniform(200, 3500), 2)
            ltp = round(avg * rng.uniform(0.85, 1.25), 2)
            data.append([s, isin, str(qty), f"{avg:.2f}", f"{ltp:.2f}",
                         f"{(ltp-avg)*qty:+.2f}"])
        elif kind == "expense_report":
            data.append([dt, rng.choice(cats), rng.choice(vendors),
                         fmt_amount(rng, round(rng.uniform(200, 18000), 2)),
                         f"{rng.choice([5, 12, 18])}%", "-"])
        else:
            emi = 25400.0
            interest = round(rng.uniform(3000, 9000), 2)
            principal = round(emi - interest, 2)
            bal = round(max(0.0, rng.uniform(100000, 900000)), 2)
            data.append([f"{i+1:02d}", f"{emi:.2f}", f"{principal:.2f}",
                         f"{interest:.2f}", inr_group(bal), f"{rng.randint(4, 88)}%"])
    # grid: header + rows
    d.rectangle([40, ty, W - 40, ty + rh], fill=(235, 235, 240))
    for ci in range(nc):
        d.text((col_x[ci] + 8, ty + 10), cols[ci], font=f["th"], fill=(0, 0, 0))
        if ci: d.line([col_x[ci], ty, col_x[ci], ty + rh * (n + 1)], fill=BORDER, width=1)
    d.line([40, ty, W - 40, ty], fill=BORDER, width=1)
    d.line([40, ty + rh * (n + 1), W - 40, ty + rh * (n + 1)], fill=BORDER, width=1)
    for ri, rowv in enumerate(data):
        y = ty + rh * (ri + 1)
        d.line([40, y, W - 40, y], fill=BORDER, width=1)
        for ci, v in enumerate(rowv[:nc]):
            d.text((col_x[ci] + 8, y + 10), str(v), font=f["cell"], fill=(0, 0, 0))
    d.text((40, ty + rh * (n + 1) + 30),
           f"{n} rows | generated {kind}", font=f["small"], fill=(120, 120, 120))
    # grounded QA target: a specific cell — lookup questions, not open ones
    if kind == "transaction_ledger":
        qa = {"col": "Balance", "row_idx": n - 1,
              "q": "What is the balance shown in the last row of this ledger?",
              "a": data[n - 1][5]}
    elif kind == "portfolio_holdings":
        qa = {"col": "P&L", "row_idx": 0, "q": f"What is the P&L value for scrip {data[0][0]}?",
              "a": data[0][5]}
    elif kind == "expense_report":
        qa = {"col": "Amount", "row_idx": n - 1,
              "q": "What is the amount of the last expense listed?",
              "a": data[n - 1][3]}
    else:
        qa = {"col": "Balance", "row_idx": 0,
              "q": "What is the outstanding balance after the first EMI in this schedule?",
              "a": data[0][4]}
    gt = {"family": "table_sheet", "sheet_kind": kind, "sheet_index": seed % len(sheets),
          "n_rows": n, "columns": cols, "qa": qa}
    return np.asarray(img, float), gt
