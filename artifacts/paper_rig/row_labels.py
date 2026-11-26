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
    # K5 twins: omitted fields are absent from gt — .get + None-filter degrades
    # gracefully; non-twin pages have every key, so behaviour is unchanged there.
    pairs = [
        ("account_number", "What is the account number on this statement?", gt.get("account_number")),
        ("from_date", "What is the from-date on this statement?", gt.get("from_date")),
        ("n_transactions", "How many transactions are listed on this statement?",
         str(gt["n_transactions"]) if "n_transactions" in gt else None),
        ("bank", "What is the bank name?", gt.get("bank")),
    ]
    return [(f, q, a) for f, q, a in pairs if a is not None]

def _card_pairs(gt):
    pairs = [
        ("total_due", "What is the total amount due on this credit card statement?",
         _inr(gt["total_amount_due"]) if "total_amount_due" in gt else None),
        ("min_due", "What is the minimum amount due on this credit card statement?",
         _inr(gt["min_amount_due"]) if "min_amount_due" in gt else None),
        ("due_date", "What is the due date on this statement?", gt.get("due_date")),
        ("card_masked", "What is the masked card number on this statement?", gt.get("card_masked")),
        ("customer_name", "Who is the customer on this statement?", gt.get("customer_name")),
        ("invoice_no", "What is the invoice number on this statement?", gt.get("invoice_no")),
        ("issuer", "Which bank issued this card statement?", gt.get("issuer")),
    ]
    return [(f, q, a) for f, q, a in pairs if a is not None]

def _electricity_pairs(gt):
    pairs = [
        ("total_amount", "What is the total amount payable on this electricity bill?",
         gt.get("printed_total")),
        ("units_consumed", "How many units were consumed on this electricity bill?",
         str(gt["units_consumed"]) if "units_consumed" in gt else None),
        ("due_date", "What is the due date on this electricity bill?", gt.get("due_date")),
        ("consumer_no", "What is the consumer number on this bill?", gt.get("consumer_no")),
        ("provider", "Which electricity board issued this bill?", gt.get("provider")),
    ]
    return [(f, q, a) for f, q, a in pairs if a is not None]

def _telecom_pairs(gt):
    pairs = [
        ("total_amount", "What is the total amount payable on this telecom invoice?",
         gt.get("printed_total")),
        ("due_date", "What is the due date on this telecom invoice?", gt.get("due_date")),
        ("invoice_no", "What is the invoice number on this telecom invoice?", gt.get("invoice_no")),
        ("mobile", "What is the mobile number on this invoice?", gt.get("mobile")),
        ("provider", "Which telecom operator issued this invoice?", gt.get("provider")),
    ]
    return [(f, q, a) for f, q, a in pairs if a is not None]

def _salary_pairs(gt):
    # K5 twins: omitted fields are absent from gt — .get + None-filter degrades
    # gracefully; non-twin pages have every key, so behaviour is unchanged there.
    pairs = [
        ("net_pay", "What is the net pay on this salary slip?", gt.get("printed_net")),
        ("gross_earnings", "What are the gross earnings on this salary slip?",
         _inr(gt["gross_earnings"]) if "gross_earnings" in gt else None),
        ("employee_name", "Who is the employee on this salary slip?", gt.get("employee_name")),
        ("employer", "Which company issued this salary slip?", gt.get("employer")),
        ("period", "Which salary period does this slip cover?", gt.get("period")),
    ]
    return [(f, q, a) for f, q, a in pairs if a is not None]

def _table_pairs(gt):
    qa = gt["qa"]
    return [(qa["col"], qa["q"], qa["a"])]

TABLES = {"thermal": _thermal_pairs, "savings": _savings_pairs, "card": _card_pairs,
          "electricity": _electricity_pairs, "telecom": _telecom_pairs,
          "salary": _salary_pairs, "table": _table_pairs}

def build_qa(family, gt, seed):
    """One (field, question, answer) dict per call. Field rotates deterministically by seed."""
    pairs = TABLES[family](gt)
    f, q, a = pairs[seed % len(pairs)]
    return [{"field": f, "q": q, "a": a}]
