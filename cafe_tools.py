# cafe_tools.py
# The "live numbers" half of the agent (agent.py), for the DEMO.
#
# In a real setup these tools would read your actual business or finance system,
# read-only. In this public demo they read FICTIONAL numbers for Ember & Oak
# Coffee from sample_data/cafe_numbers.json, so the agent can be shown answering
# questions that need both notes ("what is my emergency fund target?") and
# numbers ("how much is in it now?").
#
# Every tool returns a small dict the model can read. Anything it cannot answer
# comes back as {"error": ...} so the model can say so instead of guessing.

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "sample_data" / "cafe_numbers.json"


def _data():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _month(data, month):
    """Pick the month to report: the one asked for, or the latest if none given.
    Returns (month, row) or (None, error dict)."""
    months = sorted(data["months"])
    if not month:
        month = months[-1]
    if month not in data["months"]:
        return None, {"error": f"no café numbers for {month}", "available_months": months}
    return month, data["months"][month]


def cafe_sales(month=None):
    """Café sales for one month (YYYY-MM): total, by category, and drinks sold."""
    month, row = _month(_data(), month)
    if month is None:
        return row
    return {
        "month": month,
        "total_sales": sum(row["sales"].values()),
        "by_category": row["sales"],
        "drinks_sold": row["drinks_sold"],
    }


def cafe_expenses(month=None, category=None):
    """Café costs for one month. Give a category (e.g. 'green beans') for just that line."""
    month, row = _month(_data(), month)
    if month is None:
        return row
    exp = row["expenses"]
    if category:
        key = next((k for k in exp if k.lower() == category.strip().lower()), None)
        if key is None:
            return {"error": f"no cost category '{category}'", "categories": sorted(exp)}
        return {"month": month, "category": key, "amount": exp[key]}
    return {"month": month, "total_expenses": sum(exp.values()), "by_category": exp}


def cafe_profit(month=None):
    """Café profit for one month: sales minus costs."""
    month, row = _month(_data(), month)
    if month is None:
        return row
    sales = sum(row["sales"].values())
    costs = sum(row["expenses"].values())
    return {"month": month, "total_sales": sales, "total_expenses": costs, "profit": sales - costs}


def cafe_cash_position():
    """Current balances: operating account, emergency fund, expansion fund, and total."""
    cp = _data()["cash_position"]
    return {"as_of": cp["as_of"], "accounts": cp["accounts"], "total": sum(cp["accounts"].values())}
