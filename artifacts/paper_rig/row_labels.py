"""gt -> one (question, answer) pair per row. Gold-label doctrine:
a field destroyed by smudge_heavy/tear answers 'unreadable'; fade/band/crease
keep the value (partially readable = still answerable)."""
UNREADABLE_OPS = {"smudge_heavy", "tear"}

def _leg(gt, fid):
    op = gt.get("legibility_applied", {}).get(fid)
    return "unreadable" if op in UNREADABLE_OPS else None

def _inr(x):
    return f"Rs {x:.2f}"

def _thermal_pairs(gt):
    return [
        ("total_amt", "What is the total amount on this receipt?",
         _leg(gt, "total_amt") or _inr(gt["total"])),
        ("bill_no", "What is the bill number on this receipt?",
         _leg(gt, "bill_no") or gt["bill_no"]),
        ("shop_name", "Which shop issued this receipt?",
         _leg(gt, "shop_name") or gt["shop"]),
        ("date", "What is the date printed on this receipt?",
         _leg(gt, "date") or gt["date"]),
        ("gstin", "What is the GSTIN printed on this receipt?",
         _leg(gt, "gstin") or gt["gstin"]),
        ("n_items", "How many items were billed on this receipt?", str(len(gt["items"]))),
    ]

def _savings_pairs(gt):
    return [
        ("account_number", "What is the account number on this statement?", gt["account_number"]),
        ("from_date", "What is the from-date on this statement?", gt["from_date"]),
        ("n_transactions", "How many transactions are listed on this statement?",
         str(gt["n_transactions"])),
        ("bank", "What is the bank name?", "ICICI Bank"),
    ]

def _card_pairs(gt):
    return [
        ("total_due", "What is the total amount due on this credit card statement?",
         _inr(gt["total_amount_due"])),
        ("min_due", "What is the minimum amount due on this credit card statement?",
         _inr(gt["min_amount_due"])),
        ("due_date", "What is the due date on this statement?", gt["due_date"]),
        ("card_masked", "What is the masked card number on this statement?", gt["card_masked"]),
        ("customer_name", "Who is the customer on this statement?", gt["customer_name"]),
        ("invoice_no", "What is the invoice number on this statement?", gt["invoice_no"]),
    ]

TABLES = {"thermal": _thermal_pairs, "savings": _savings_pairs, "card": _card_pairs}

def build_qa(family, gt, seed):
    """One (field, question, answer) dict per call. Field rotates deterministically by seed."""
    pairs = TABLES[family](gt)
    f, q, a = pairs[seed % len(pairs)]
    return [{"field": f, "q": q, "a": a}]
