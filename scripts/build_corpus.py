import json
from pathlib import Path
from collections import Counter

OUT_CORPUS = Path("data/corpus/te_policy_corpus.jsonl")
OUT_TABLES = Path("data/policy_tables")
OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. CANONICAL POLICY TABLES (Single Source of Truth)
# =========================================================
REGIONS = {
    "IN": {"currency": "INR", "fx_to_usd": 0.012, "mileage_unit": "km", "car_rate": 14.0, "bike_rate": 7.0, "ev_rate": 16.0, "receipt_threshold": 500, "tax_doc": "GST Tax Invoice"},
    "US": {"currency": "USD", "fx_to_usd": 1.0, "mileage_unit": "mile", "car_rate": 0.67, "bike_rate": 0.35, "ev_rate": 0.72, "receipt_threshold": 25, "tax_doc": "Itemized Merchant Receipt"},
    "EU": {"currency": "EUR", "fx_to_usd": 1.08, "mileage_unit": "km", "car_rate": 0.35, "bike_rate": 0.20, "ev_rate": 0.40, "receipt_threshold": 20, "tax_doc": "VAT-Compliant Invoice"},
    "UK": {"currency": "GBP", "fx_to_usd": 1.27, "mileage_unit": "mile", "car_rate": 0.45, "bike_rate": 0.24, "ev_rate": 0.50, "receipt_threshold": 20, "tax_doc": "UK VAT Receipt"},
    "APAC": {"currency": "SGD", "fx_to_usd": 0.74, "mileage_unit": "km", "car_rate": 0.60, "bike_rate": 0.30, "ev_rate": 0.68, "receipt_threshold": 30, "tax_doc": "GST/Tax Receipt"},
}

CITY_TIERS = {
    "IN": {
        "Tier-1": {"cities": ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Kolkata"], "hotel": 9500, "per_diem": 2200, "peak_months": ["Nov", "Dec", "Jan", "Feb"], "peak_mult": 1.15},
        "Tier-2": {"cities": ["Jaipur", "Ahmedabad", "Kochi", "Chandigarh", "Indore", "Patna", "Lucknow"], "hotel": 6500, "per_diem": 1600, "peak_months": ["Nov", "Dec", "Jan"], "peak_mult": 1.10},
        "Tier-3": {"cities": ["Nagpur", "Coimbatore", "Bhopal", "Surat", "Vadodara", "Ranchi"], "hotel": 4500, "per_diem": 1200, "peak_months": ["Dec", "Jan"], "peak_mult": 1.10},
    },
    "US": {
        "Tier-1": {"cities": ["New York", "San Francisco", "Boston", "Seattle", "Washington DC", "Los Angeles"], "hotel": 320, "per_diem": 95, "peak_months": ["May", "Jun", "Sep", "Oct"], "peak_mult": 1.15},
        "Tier-2": {"cities": ["Austin", "Chicago", "Denver", "Atlanta", "Dallas", "Miami"], "hotel": 220, "per_diem": 75, "peak_months": ["Mar", "Apr", "Oct"], "peak_mult": 1.10},
        "Tier-3": {"cities": ["Columbus", "Kansas City", "Pittsburgh", "Raleigh", "Salt Lake City"], "hotel": 160, "per_diem": 60, "peak_months": ["Sep", "Oct"], "peak_mult": 1.05},
    },
    "EU": {
        "Tier-1": {"cities": ["Paris", "Munich", "Amsterdam", "Zurich", "Frankfurt", "Dublin"], "hotel": 260, "per_diem": 85, "peak_months": ["Jun", "Jul", "Sep"], "peak_mult": 1.15},
        "Tier-2": {"cities": ["Berlin", "Madrid", "Milan", "Vienna", "Brussels", "Barcelona"], "hotel": 190, "per_diem": 65, "peak_months": ["May", "Jun", "Sep"], "peak_mult": 1.10},
    },
    "UK": {
        "Tier-1": {"cities": ["London", "Edinburgh"], "hotel": 250, "per_diem": 80, "peak_months": ["Jun", "Jul", "Aug"], "peak_mult": 1.15},
        "Tier-2": {"cities": ["Manchester", "Birmingham", "Bristol", "Leeds"], "hotel": 160, "per_diem": 60, "peak_months": ["Jun", "Jul"], "peak_mult": 1.10},
    },
    "APAC": {
        "Tier-1": {"cities": ["Singapore", "Tokyo", "Sydney", "Hong Kong"], "hotel": 300, "per_diem": 95, "peak_months": ["Mar", "Apr", "Sep", "Oct"], "peak_mult": 1.15},
        "Tier-2": {"cities": ["Melbourne", "Seoul", "Kuala Lumpur", "Bangkok"], "hotel": 190, "per_diem": 65, "peak_months": ["Nov", "Dec"], "peak_mult": 1.10},
    },
}

GRADE_BANDS = {
    "L1-L4": {"grades": ["L1", "L2", "L3", "L4"], "hotel_mult": 1.0, "flight_dom": "Economy", "flight_intl_short": "Economy", "flight_intl_long": "Premium Economy", "adv_days": 14, "intl_travel_requires_vp": False},
    "L5-L6": {"grades": ["L5", "L6"], "hotel_mult": 1.25, "flight_dom": "Economy", "flight_intl_short": "Premium Economy", "flight_intl_long": "Business", "adv_days": 7, "intl_travel_requires_vp": False},
    "L7-L8": {"grades": ["L7", "L8"], "hotel_mult": 1.6, "flight_dom": "Business", "flight_intl_short": "Business", "flight_intl_long": "Business", "adv_days": 3, "intl_travel_requires_vp": False},
}

ELIGIBILITY_RULES = {
    "allowed_statuses": ["ACTIVE"],
    "probation_travel": "SOFT_FLAG",
    "contractor_travel": "HUMAN_REVIEW",
    "restricted_destinations": ["North Korea", "Syria", "Iran", "Crimea", "Sanctioned-Region"],
    "valid_project_prefixes": ["PRJ-", "CLI-", "INT-", "RND-"],
    "departments": ["Engineering", "Sales", "Consulting", "Finance", "Operations", "HR", "Marketing", "Legal"],
}

EXPENSE_CAPS_USD_EQ = {
    "ground_transport": {"per_day": 60, "soft_overage_pct": 0.20},
    "client_entertainment": {"per_attendee": 75, "soft_overage_pct": 0.15},
    "telecom_roaming": {"per_trip": 50, "soft_overage_pct": 0.25},
    "laundry": {"per_trip_min_4_nights": 40, "soft_overage_pct": 0.15},
    "airport_parking": {"per_day": 25, "soft_overage_pct": 0.20},
    "visa_and_passport": {"per_trip": 350, "soft_overage_pct": 0.10},
}

PROHIBITED_ITEMS = [
    "alcohol_non_client", "minibar", "in_room_movies", "spa_and_wellness",
    "personal_grooming", "traffic_fines", "airline_lounge_personal", "companion_travel",
    "personal_luggage_purchase", "luxury_car_rental", "gym_membership", "pet_boarding"
]

APPROVAL_MATRIX = [
    {"violation_type": "PASS", "grade_band": "L1-L4", "max_amount_usd_eq": 1000, "approver": "Direct Manager"},
    {"violation_type": "PASS", "grade_band": "L1-L4", "max_amount_usd_eq": 5000, "approver": "Department Head"},
    {"violation_type": "PASS", "grade_band": "L5-L6", "max_amount_usd_eq": 2500, "approver": "Direct Manager"},
    {"violation_type": "PASS", "grade_band": "L5-L6", "max_amount_usd_eq": 10000, "approver": "VP / Business Unit Head"},
    {"violation_type": "PASS", "grade_band": "L7-L8", "max_amount_usd_eq": 15000, "approver": "Finance Controller"},
    {"violation_type": "SOFT_FLAG", "grade_band": "L1-L4", "max_amount_usd_eq": 1500, "approver": "Direct Manager"},
    {"violation_type": "SOFT_FLAG", "grade_band": "L1-L4", "max_amount_usd_eq": 5000, "approver": "Department Head"},
    {"violation_type": "SOFT_FLAG", "grade_band": "L5-L6", "max_amount_usd_eq": 5000, "approver": "Department Head"},
    {"violation_type": "SOFT_FLAG", "grade_band": "ALL", "max_amount_usd_eq": 999999, "approver": "Finance Controller"},
    {"violation_type": "HARD_FLAG", "grade_band": "ALL", "max_amount_usd_eq": 999999, "approver": "BLOCKED (Requires VP + CFO Written Exception)"},
    {"violation_type": "HUMAN_REVIEW", "grade_band": "ALL", "max_amount_usd_eq": 999999, "approver": "Finance Policy Desk"},
]

VENDORS = {
    "IN": {
        "portal": "Concur-IN / MakeMyTrip Corporate",
        "airlines": ["IndiGo Corporate", "Air India Corporate", "Akasa Air Business"],
        "hotels": ["Taj Group (IHCL)", "Lemon Tree Hotels", "Marriott India", "ITC Hotels"],
        "cabs": ["Uber for Business", "BluSmart Corporate", "Ola Corporate"],
    },
    "US": {
        "portal": "SAP Concur Travel",
        "airlines": ["Delta Air Lines", "United Airlines", "American Airlines"],
        "hotels": ["Marriott Bonvoy Corporate", "Hilton for Business", "Hyatt Business"],
        "cabs": ["Uber for Business", "Lyft Business", "Enterprise Rent-A-Car"],
    },
    "EU": {
        "portal": "SAP Concur EU",
        "airlines": ["Lufthansa Group", "Air France-KLM", "Iberia Corporate"],
        "hotels": ["Accor Business", "NH Hotels", "Marriott Europe"],
        "cabs": ["FreeNow Business", "Uber for Business", "Sixt Corporate"],
    },
    "UK": {
        "portal": "SAP Concur UK",
        "airlines": ["British Airways On Business", "Virgin Atlantic Corporate", "EasyJet Business"],
        "hotels": ["Premier Inn Business", "Hilton UK", "IHG Business Edge"],
        "cabs": ["Uber for Business", "Addison Lee Corporate", "National Rail Business"],
    },
    "APAC": {
        "portal": "SAP Concur APAC",
        "airlines": ["Singapore Airlines HighFlyer", "ANA Corporate", "Qantas Business"],
        "hotels": ["Pan Pacific Corporate", "Shangri-La Business", "Marriott APAC"],
        "cabs": ["Grab for Business", "Uber for Business", "Gojek Corporate"],
    },
}

# =========================================================
# 2. DOCUMENT SYNTHESIZER
# =========================================================
docs = []
doc_counter = 1
ALL_GRADES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]

def add_doc(region, grade_scope, category, section_id, title, text, effective_date="2026-01-01", status="ACTIVE", doc_type="policy"):
    global doc_counter
    prefix = "SUP" if status == "SUPERSEDED" else "POL"
    doc_id = f"{prefix}-{region}-{category[:3].upper()}-{doc_counter:04d}"
    doc_counter += 1
    docs.append({
        "doc_id": doc_id,
        "section_id": section_id,
        "policy_ref": f"{doc_id} §{section_id}",
        "title": title,
        "region": region,
        "grade_scope": grade_scope,
        "expense_category": category,
        "effective_date": effective_date,
        "status": status,
        "doc_type": doc_type,
        "text": text.strip(),
    })

# A. Travel Eligibility & Project Code Rules (Resp 1)
for dept in ELIGIBILITY_RULES["departments"]:
    for reg in REGIONS:
        add_doc(
            region=reg, grade_scope=ALL_GRADES, category="eligibility", section_id="1.1",
            title=f"Travel Eligibility & Project Code Mandate — {dept} ({reg})",
            text=(
                f"Active full-time employees in the {dept} department ({reg}) are eligible for business travel only when "
                f"assigned to a valid active project code prefixed with {', '.join(ELIGIBILITY_RULES['valid_project_prefixes'])}. "
                f"Requests missing employee_id, grade, department, or project_code must return NEEDS_INFORMATION. "
                f"Employees on probation require Direct Manager pre-approval (SOFT_FLAG); contractors require Finance Policy Desk clearance (HUMAN_REVIEW)."
            )
        )

for dest in ELIGIBILITY_RULES["restricted_destinations"]:
    add_doc(
        region="GLOBAL", grade_scope=ALL_GRADES, category="eligibility", section_id="1.4",
        title=f"Restricted Travel Destination — {dest}",
        text=(
            f"Corporate travel to {dest} is strictly prohibited due to international sanctions and corporate duty-of-care restrictions. "
            f"Any travel or expense request involving {dest} must be immediately classified as HARD_FLAG and blocked."
        )
    )

# B. City-Tier Hotel & Per Diem Rate Tables (Resp 3 & 4)
for reg, tiers in CITY_TIERS.items():
    cur = REGIONS[reg]["currency"]
    for tier, info in tiers.items():
        peak_str = ", ".join(info["peak_months"])
        for city in info["cities"]:
            for band_name, band in GRADE_BANDS.items():
                cap = round(info["hotel"] * band["hotel_mult"], 2)
                peak_cap = round(cap * info["peak_mult"], 2)
                soft_max = round(cap * 1.15, 2)
                add_doc(
                    region=reg, grade_scope=band["grades"], category="hotel", section_id="3.1",
                    title=f"{city} ({tier}, {reg}) Hotel Cap — {band_name}",
                    doc_type="rate_table",
                    text=(
                        f"Employees in {band_name} travelling to {city} ({tier}, {reg}) have a standard nightly hotel cap of "
                        f"{cur} {cap} excluding taxes. During peak season ({peak_str}), the seasonal cap increases to {cur} {peak_cap}. "
                        f"Rates exceeding the applicable cap by up to 15% (up to {cur} {soft_max} standard) trigger a SOFT_FLAG requiring "
                        f"written justification and approver review. Rates exceeding 15% above cap trigger a HARD_FLAG."
                    )
                )
            add_doc(
                region=reg, grade_scope=ALL_GRADES, category="per_diem", section_id="4.1",
                title=f"{city} ({tier}, {reg}) Daily Meal & Incidental Per Diem",
                doc_type="rate_table",
                text=(
                    f"The standard daily meal and incidental per diem for {city} ({tier}, {reg}) is {cur} {info['per_diem']} per full travel day. "
                    f"First and last travel days are reimbursed at 75% ({cur} {round(info['per_diem']*0.75, 2)}). "
                    f"Complimentary or host-provided meals must be deducted from the daily allowance (Breakfast 20%, Lunch 30%, Dinner 50%)."
                )
            )

# C. Flight Policy Rules (Resp 2)
for reg in REGIONS:
    for band_name, band in GRADE_BANDS.items():
        for route_type, cabin_key, dur in [
            ("Domestic", "flight_dom", "any duration"),
            ("International Short-Haul", "flight_intl_short", "under 6 hours"),
            ("International Long-Haul", "flight_intl_long", "6 hours or more"),
        ]:
            add_doc(
                region=reg, grade_scope=band["grades"], category="flight", section_id="2.1",
                title=f"{reg} {route_type} Flight Entitlement — {band_name}",
                text=(
                    f"For {route_type} flights ({dur}) originating in {reg}, {band_name} employees are entitled to "
                    f"{band[cabin_key]} class. Bookings must be completed at least {band['adv_days']} days before departure. "
                    f"Bookings inside the {band['adv_days']}-day window trigger a SOFT_FLAG requiring business urgency justification. "
                    f"Booking a higher cabin class than entitled without pre-approved exception triggers a HARD_FLAG."
                )
            )

# D. Mileage Reimbursement Circulars (Resp 7)
for reg, rinfo in REGIONS.items():
    cur, unit = rinfo["currency"], rinfo["mileage_unit"]
    for vtype, rate, cap_dist in [
        ("Personal Car", rinfo["car_rate"], 400),
        ("Two-Wheeler", rinfo["bike_rate"], 150),
        ("Personal Electric Vehicle (EV)", rinfo["ev_rate"], 400),
    ]:
        add_doc(
            region=reg, grade_scope=ALL_GRADES, category="mileage", section_id="7.1",
            title=f"{reg} Mileage Reimbursement Circular — {vtype} (2026)",
            doc_type="circular",
            text=(
                f"Effective 2026-01-01, business travel by {vtype} in {reg} is reimbursed at {cur} {rate} per {unit}. "
                f"Routine commute between home and primary office is strictly excluded. Single-day claims exceeding {cap_dist} {unit} "
                f"trigger a SOFT_FLAG for route verification; claims missing origin, destination, or distance logs return NEEDS_INFORMATION."
            )
        )

# E. Receipt Threshold & Tax Rules (Resp 6)
for reg, rinfo in REGIONS.items():
    cur = rinfo["currency"]
    for cat in ["meals", "ground_transport", "hotel", "flight", "client_entertainment", "telecom_roaming", "airport_parking"]:
        thresh = 0 if cat in ["hotel", "flight", "client_entertainment"] else rinfo["receipt_threshold"]
        rule = (
            f"Mandatory itemized {rinfo['tax_doc']} required for all amounts."
            if thresh == 0
            else f"Itemized {rinfo['tax_doc']} mandatory for claims >= {cur} {thresh}."
        )
        add_doc(
            region=reg, grade_scope=ALL_GRADES, category="receipt", section_id="6.1",
            title=f"{reg} Receipt & Tax Rule — {cat.replace('_', ' ').title()}",
            doc_type="receipt_rule",
            text=(
                f"For {cat.replace('_', ' ')} claims in {reg}: {rule} "
                f"Credit card slips without itemization are insufficient for hotel and client entertainment claims. "
                f"Missing receipts above threshold require a signed Lost Receipt Affidavit (max 2 per fiscal year) and trigger a SOFT_FLAG; "
                f"a 3rd missing receipt in the same fiscal year triggers a HARD_FLAG."
            )
        )

# F. Expense Claim Caps & Prohibited Items (Resp 5)
for reg, rinfo in REGIONS.items():
    cur = rinfo["currency"]
    for cat, cinfo in EXPENSE_CAPS_USD_EQ.items():
        basis_key = [k for k in cinfo if k != "soft_overage_pct"][0]
        local_cap = round(cinfo[basis_key] / rinfo["fx_to_usd"], 2)
        pct = int(cinfo["soft_overage_pct"] * 100)
        add_doc(
            region=reg, grade_scope=ALL_GRADES, category="expense_claim", section_id="5.1",
            title=f"{reg} Expense Cap — {cat.replace('_', ' ').title()}",
            text=(
                f"In {reg}, {cat.replace('_', ' ')} claims are capped at {cur} {local_cap} ({basis_key.replace('_', ' ')}). "
                f"Claims exceeding the cap by up to {pct}% trigger a SOFT_FLAG; claims exceeding {pct}% above the cap trigger a HARD_FLAG."
            )
        )

for item in PROHIBITED_ITEMS:
    add_doc(
        region="GLOBAL", grade_scope=ALL_GRADES, category="expense_claim", section_id="5.4",
        title=f"Prohibited Expense Rule — {item.replace('_', ' ').title()}",
        text=(
            f"Expenses categorized as '{item.replace('_', ' ')}' are strictly non-reimbursable across all grades and regions. "
            f"Any claim containing this line item must be classified as HARD_FLAG and excluded from reimbursement."
        )
    )

# G. Approval Hierarchy Matrices (Resp 8)
for reg, rinfo in REGIONS.items():
    cur = rinfo["currency"]
    for row in APPROVAL_MATRIX:
        local_amt = round(row["max_amount_usd_eq"] / rinfo["fx_to_usd"], 2)
        g_scope = ALL_GRADES if row["grade_band"] == "ALL" else GRADE_BANDS[row["grade_band"]]["grades"]
        add_doc(
            region=reg, grade_scope=g_scope, category="approval", section_id="8.1",
            title=f"{reg} Approval Hierarchy — {row['violation_type']} ({row['grade_band']}, up to {cur} {local_amt})",
            doc_type="approval_matrix",
            text=(
                f"In {reg}, for {row['grade_band']} claims classified as {row['violation_type']} up to {cur} {local_amt} "
                f"(USD {row['max_amount_usd_eq']} equivalent), the mandatory human approver is: {row['approver']}. "
                f"The AI copilot may recommend actions but cannot substitute for human authorization."
            )
        )

# H. Vendor Addenda & Preferred Partners
for reg, vinfo in VENDORS.items():
    portal = vinfo["portal"]
    for airline in vinfo["airlines"]:
        add_doc(
            region=reg, grade_scope=ALL_GRADES, category="flight", section_id="2.4",
            title=f"Vendor Addendum ({reg}) — Preferred Airline: {airline}",
            doc_type="vendor_addendum",
            text=(
                f"{airline} is a contracted preferred carrier for {reg} travel booked via {portal}. "
                f"Corporate fares include free seat selection, 1 checked bag, and zero change fees. "
                f"Booking a non-preferred airline is permitted only when the fare is at least 15% cheaper or no preferred flight operates within 3 hours; "
                f"otherwise, off-portal or non-preferred bookings trigger a SOFT_FLAG."
            )
        )
    for hotel in vinfo["hotels"]:
        add_doc(
            region=reg, grade_scope=ALL_GRADES, category="hotel", section_id="3.4",
            title=f"Vendor Addendum ({reg}) — Preferred Hotel Chain: {hotel}",
            doc_type="vendor_addendum",
            text=(
                f"{hotel} is a corporate preferred hotel partner in {reg} accessible through {portal}. "
                f"Negotiated corporate rates include complimentary breakfast and Wi-Fi (do not claim breakfast per diem when included). "
                f"Booking outside {portal} without documented unavailability triggers a SOFT_FLAG."
            )
        )
    for cab in vinfo["cabs"]:
        add_doc(
            region=reg, grade_scope=ALL_GRADES, category="ground_transport", section_id="5.6",
            title=f"Vendor Addendum ({reg}) — Preferred Ground Transport: {cab}",
            doc_type="vendor_addendum",
            text=(
                f"{cab} is an approved corporate ground transport partner in {reg}. "
                f"Employees must select standard sedan/hatchback tiers; luxury or black-car tiers are prohibited (HARD_FLAG) "
                f"unless transporting external executive clients with pre-approval."
            )
        )

# I. Exception FAQs & Edge Cases (Resp 9 & 10)
FAQ_TEMPLATES = [
    ("bleisure_extension", "flight", "SOFT_FLAG",
     "Can I extend a business trip for personal leisure (bleisure) over the weekend?",
     "Yes, provided the return flight costs equal to or less than a standard business-only itinerary (comparison screenshot mandatory). All hotel, meal, and transport expenses during personal days are strictly non-reimbursable. Missing fare-comparison evidence triggers NEEDS_INFORMATION; higher airfare without paying the difference triggers a SOFT_FLAG."),
    ("client_alcohol", "expense_claim", "SOFT_FLAG",
     "Is alcohol reimbursable during a client dinner?",
     "Alcohol is strictly prohibited for solo or internal team meals (HARD_FLAG). During external client dinners, moderate alcohol (max 2 drinks per attendee, within the per-attendee client entertainment cap) is allowed only with itemized receipt, attendee list, and Department Head review (SOFT_FLAG)."),
    ("citywide_sold_out", "hotel", "SOFT_FLAG",
     "What if every hotel in my destination city exceeds the nightly cap due to a major conference?",
     "If a citywide event causes all corporate-portal hotels to exceed the city-tier cap by more than 15%, attach a portal search screenshot showing no compliant rooms within 5 km. This converts a standard HARD_FLAG into a SOFT_FLAG routed to Department Head + Finance review."),
    ("late_flight_urgent_client", "flight", "SOFT_FLAG",
     "My client scheduled an emergency meeting 3 days before travel, violating the advance booking window.",
     "Bookings made inside the grade-based advance booking window trigger a SOFT_FLAG (not a HARD_FLAG). Attach the client email or incident ticket as justification for Direct Manager approval."),
    ("split_receipt_avoid_threshold", "receipt", "HARD_FLAG",
     "Can I split a single meal or vendor bill into two smaller claims below the receipt threshold?",
     "No. Deliberately splitting expenses to bypass receipt thresholds or approval slabs is a policy violation classified as HARD_FLAG and escalated to Finance Audit."),
    ("expired_project_code", "eligibility", "NEEDS_INFORMATION",
     "What happens if I submit a travel request with an expired or missing project code?",
     "The system cannot validate budget ownership without an active project code. Requests with missing or closed project codes must be classified as NEEDS_INFORMATION until the employee provides an active code."),
    ("ambiguous_medical_upgrade", "flight", "HUMAN_REVIEW",
     "I have a documented medical condition and need Business Class on a domestic flight as an L3 employee.",
     "Medical accommodation cabin upgrades are not auto-approved or auto-rejected by standard rules. Classify as HUMAN_REVIEW and escalate to HR Benefits and the Finance Policy Desk with medical documentation."),
    ("spouse_travel_upgrade", "expense_claim", "HARD_FLAG",
     "Can I expense my spouse's flight or hotel room upgrade on a 2-week work trip?",
     "Companion travel expenses and room upgrades for family members are prohibited across all grades (HARD_FLAG). Only the single-occupancy base room rate is reimbursable."),
    ("lost_hotel_folio", "receipt", "SOFT_FLAG",
     "I lost my itemized hotel folio and only have my credit card statement.",
     "A credit card statement alone is insufficient because hotels may include prohibited items (minibar, movies, spa). Request a duplicate folio from the hotel; if impossible, submit a Lost Receipt Affidavit (SOFT_FLAG, max 2/year)."),
    ("currency_conversion_variance", "expense_claim", "HUMAN_REVIEW",
     "My corporate card FX rate differs by 6% from the standard monthly reimbursement rate.",
     "FX variances up to 3% are auto-accepted with card statement proof (PASS). Variances above 5% without card-issuer documentation require Finance Policy Desk verification (HUMAN_REVIEW)."),
    ("home_to_airport_mileage", "mileage", "PASS",
     "Can I claim mileage from my home to the airport for a business trip?",
     "Yes. Travel from home to the airport or railway station for an approved business trip is reimbursable at the regional per-km/mile rate, minus normal daily commute distance if departing during regular office hours."),
    ("ev_charging_vs_mileage", "mileage", "HARD_FLAG",
     "Can I claim both the per-km EV mileage rate and public EV charging receipts?",
     "No. The official EV mileage reimbursement rate already covers electricity, wear-and-tear, and insurance. Claiming both mileage and fuel/charging receipts is double-dipping and triggers a HARD_FLAG."),
    ("conference_included_meals", "per_diem", "SOFT_FLAG",
     "My conference registration included lunch, can I still claim full daily per diem?",
     "No. Full per diem cannot be claimed when meals are provided by the hotel, airline, or conference. Deduct 20% for breakfast, 30% for lunch, and 50% for dinner. Unadjusted claims trigger a SOFT_FLAG."),
    ("non_preferred_hotel_cheaper", "hotel", "PASS",
     "I booked a non-preferred hotel that is 25% cheaper than the corporate preferred chain.",
     "Compliant (PASS), provided the property meets corporate safety standards, falls within the city-tier cap, and an itemized tax invoice is submitted."),
    ("contractor_per_diem", "eligibility", "HUMAN_REVIEW",
     "Are external contractors eligible for standard employee per diem rates?",
     "Contractor travel reimbursement depends on the Statement of Work (SOW), not employee policy. Classify as HUMAN_REVIEW and route to Procurement / Finance Policy Desk."),
    ("upgraded_flight_with_personal_miles", "flight", "PASS",
     "Can I upgrade my Economy ticket to Business Class using my personal frequent flyer miles?",
     "Yes (PASS), provided the company is only billed for the policy-compliant base fare and no extra change/upgrade fees are expensed."),
    ("ride_share_surge_pricing", "ground_transport", "SOFT_FLAG",
     "Heavy rain caused Uber surge pricing that pushed my daily ground transport 18% over the cap.",
     "Ground transport overages up to 20% due to surge pricing or airport congestion trigger a SOFT_FLAG with trip receipt and brief note for Direct Manager approval. Overages above 20% require Department Head review."),
    ("visa_expedite_fee", "expense_claim", "SOFT_FLAG",
     "Can I expense expedited visa processing fees for an urgent client visit?",
     "Standard visa fees are reimbursable (PASS). Expedited processing fees exceeding the regional visa cap by up to 10% trigger a SOFT_FLAG requiring confirmation that travel urgency was business-driven."),
    ("coworking_day_pass", "expense_claim", "HUMAN_REVIEW",
     "Can I expense a day pass at a coworking space during a remote client trip?",
     "Coworking passes are not explicitly covered under standard hotel or telecom categories. Classify as HUMAN_REVIEW for Finance Policy Desk determination unless pre-approved in the project budget."),
    ("missing_travel_dates", "flight", "NEEDS_INFORMATION",
     "What if a user asks whether a INR 14,000 hotel in Mumbai is compliant without giving travel dates or grade?",
     "Nightly caps depend on both employee grade band and peak/off-peak travel dates. The agent must never assume grade or dates and must return NEEDS_INFORMATION requesting both fields."),
    ("weekend_stay_cheaper_flight", "hotel", "SOFT_FLAG",
     "Staying over Saturday night reduces my airfare by USD 600, more than the extra hotel night.",
     "Permitted as a cost-saving exception (SOFT_FLAG) if total trip cost (flight + extra hotel + per diem) is lower than returning Friday. Attach fare comparison for Direct Manager sign-off."),
    ("cash_tip_no_receipt", "receipt", "PASS",
     "How are small cash tips for hotel porters or cab drivers handled without receipts?",
     "Reasonable cash gratuities below the regional receipt threshold are reimbursable within the daily incidental per diem without a receipt (PASS), provided they are itemized in the claim description."),
    ("red_eye_early_checkin", "hotel", "SOFT_FLAG",
     "My international flight lands at 5 AM; can I book an extra half-day early check-in?",
     "Early check-in fees up to 50% of the nightly cap are permitted after overnight flights exceeding 6 hours, routed as SOFT_FLAG with boarding pass proof."),
    ("cancellation_fee_client_reschedule", "flight", "PASS",
     "A client rescheduled our meeting and the airline charged a cancellation/rebooking fee.",
     "Business-driven cancellation or rebooking fees are reimbursable (PASS) with the original ticket, new itinerary, and client rescheduling email attached."),
    ("personal_card_usage", "expense_claim", "SOFT_FLAG",
     "I used my personal credit card instead of the corporate card for a hotel booking.",
     "Employees issued a corporate card must use it for all flight and hotel bookings. Using a personal card for points accumulation triggers a SOFT_FLAG on the first occurrence and Finance escalation if repeated."),
    ("group_team_dinner_cap", "expense_claim", "SOFT_FLAG",
     "I paid for a dinner for 5 internal team members during an offsite.",
     "Internal team meals must be billed under the most senior employee's corporate card and stay within the regional daily meal cap per attendee. Submitting an internal meal as 'client entertainment' is a HARD_FLAG; overages up to 15% trigger a SOFT_FLAG."),
    ("missing_itemized_gst_in", "receipt", "SOFT_FLAG",
     "My Indian hotel invoice does not include the company's GSTIN.",
     "For Indian domestic hotel stays, employees must provide the company's state GSTIN at check-in so input tax credit can be claimed. Invoices missing GSTIN above INR 5,000 trigger a SOFT_FLAG and Finance tax review."),
    ("airport_lounge_long_layover", "expense_claim", "SOFT_FLAG",
     "Can I expense a one-time airport lounge pass during a 6-hour international layover?",
     "Personal airline lounge memberships are strictly prohibited (HARD_FLAG). However, a single-use lounge pass for involuntary layovers exceeding 4 hours may be reviewed as a SOFT_FLAG with boarding pass proof."),
    ("dual_per_diem_and_actuals", "per_diem", "HARD_FLAG",
     "Can I claim daily meal per diem AND submit individual restaurant receipts on the same day?",
     "No. Mixing per diem and actual meal receipts on the same travel day is double reimbursement and triggers a HARD_FLAG. Use per diem for personal meals and actuals only for documented client entertainment (with meal deduction)."),
    ("unmapped_crypto_payment", "expense_claim", "HUMAN_REVIEW",
     "A foreign vendor only accepted stablecoin/crypto payment for a conference booth.",
     "Non-fiat/cryptocurrency reimbursements fall outside standard T&E rails. Classify as HUMAN_REVIEW and escalate to Treasury and the Finance Policy Desk."),
]

for reg in REGIONS:
    for slug, cat, decision, q, a in FAQ_TEMPLATES:
        add_doc(
            region=reg, grade_scope=ALL_GRADES, category=cat, section_id="9.2",
            title=f"Exception FAQ ({reg}) [{decision}] — {q}",
            doc_type="exception_faq",
            text=f"Region: {reg} | Expected Classification: {decision}. Question: {q} Answer: {a}"
        )

# J. Superseded 2024/2025 Circulars (Temporal Filtering Tests)
for yr, factor in [("2024-01-01", 0.85), ("2025-01-01", 0.92)]:
    for reg, rinfo in REGIONS.items():
        cur, unit = rinfo["currency"], rinfo["mileage_unit"]
        old_car = round(rinfo["car_rate"] * factor, 2)
        old_bike = round(rinfo["bike_rate"] * factor, 2)
        add_doc(
            region=reg, grade_scope=ALL_GRADES, category="mileage", section_id="7.1",
            title=f"[SUPERSEDED {yr[:4]}] {reg} Mileage Reimbursement Circular — Personal Car",
            effective_date=yr, status="SUPERSEDED", doc_type="circular",
            text=(
                f"[HISTORICAL / SUPERSEDED BY 2026-01-01 CIRCULAR] Effective {yr}, business travel by Personal Car in {reg} "
                f"was reimbursed at {cur} {old_car} per {unit}. Do not apply this rate to travel on or after 2026-01-01."
            )
        )
        add_doc(
            region=reg, grade_scope=ALL_GRADES, category="mileage", section_id="7.1",
            title=f"[SUPERSEDED {yr[:4]}] {reg} Mileage Reimbursement Circular — Two-Wheeler",
            effective_date=yr, status="SUPERSEDED", doc_type="circular",
            text=(
                f"[HISTORICAL / SUPERSEDED BY 2026-01-01 CIRCULAR] Effective {yr}, business travel by Two-Wheeler in {reg} "
                f"was reimbursed at {cur} {old_bike} per {unit}. Do not apply this rate to travel on or after 2026-01-01."
            )
        )

for reg, tiers in CITY_TIERS.items():
    cur = REGIONS[reg]["currency"]
    t1 = tiers["Tier-1"]
    for city in t1["cities"]:
        old_cap = round(t1["hotel"] * 0.85, 2)
        add_doc(
            region=reg, grade_scope=GRADE_BANDS["L1-L4"]["grades"], category="hotel", section_id="3.1",
            title=f"[SUPERSEDED 2024] {city} (Tier-1, {reg}) Hotel Cap — L1-L4",
            effective_date="2024-01-01", status="SUPERSEDED", doc_type="rate_table",
            text=(
                f"[HISTORICAL / SUPERSEDED BY 2026-01-01 POLICY] Effective 2024-01-01, L1-L4 employees travelling to {city} "
                f"had a nightly hotel cap of {cur} {old_cap}. Superseded by current 2026 Tier-1 rates."
            )
        )

for reg, rinfo in REGIONS.items():
    cur = rinfo["currency"]
    old_thresh = round(rinfo["receipt_threshold"] * 0.6, 2)
    for cat in ["meals", "ground_transport", "telecom_roaming"]:
        add_doc(
            region=reg, grade_scope=ALL_GRADES, category="receipt", section_id="6.1",
            title=f"[SUPERSEDED 2024] {reg} Receipt Threshold — {cat.replace('_', ' ').title()}",
            effective_date="2024-01-01", status="SUPERSEDED", doc_type="receipt_rule",
            text=(
                f"[HISTORICAL / SUPERSEDED BY 2026-01-01 POLICY] Prior to 2026, itemized receipts for {cat.replace('_', ' ')} "
                f"in {reg} were required above {cur} {old_thresh}. Refer to active 2026 receipt circulars."
            )
        )

# =========================================================
# 3. WRITE OUTPUTS & PRINT STATS
# =========================================================
with OUT_CORPUS.open("w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

policy_tables = {
    "regions": REGIONS,
    "city_tiers": CITY_TIERS,
    "grade_bands": GRADE_BANDS,
    "eligibility_rules": ELIGIBILITY_RULES,
    "expense_caps_usd_eq": EXPENSE_CAPS_USD_EQ,
    "prohibited_items": PROHIBITED_ITEMS,
    "approval_matrix": APPROVAL_MATRIX,
    "vendors": VENDORS,
}
with (OUT_TABLES / "policy_tables.json").open("w", encoding="utf-8") as f:
    json.dump(policy_tables, f, indent=2)

print(f"Total documents: {len(docs)}")
print("By doc_type:", dict(Counter(d["doc_type"] for d in docs)))
print("By status:", dict(Counter(d["status"] for d in docs)))