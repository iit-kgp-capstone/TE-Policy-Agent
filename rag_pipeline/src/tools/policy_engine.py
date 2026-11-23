import json
from pathlib import Path
from typing import Any

# Adjust parents[2] or parents[3] to match where data/policy_tables/ lives
PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_TABLES_PATH = PROJECT_ROOT / "data/policy_tables/policy_tables.json"


class SymbolicPolicyEngine:
    """
    Deterministic Symbolic Policy Engine covering all 10 T&E responsibilities.
    Reads exclusively from the Single Source of Truth (policy_tables.json).
    """

    def __init__(self, tables_path: Path = POLICY_TABLES_PATH):
        with tables_path.open("r", encoding="utf-8") as f:
            self.tables: dict[str, Any] = json.load(f)

        self.regions = self.tables["regions"]
        self.city_tiers = self.tables["city_tiers"]
        self.grade_bands = self.tables["grade_bands"]
        self.eligibility_rules = self.tables["eligibility_rules"]
        self.expense_caps_usd = self.tables["expense_caps_usd_eq"]
        self.prohibited_items = set(self.tables["prohibited_items"])
        self.approval_matrix = self.tables["approval_matrix"]
        self.vendors = self.tables["vendors"]

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------
    def _get_grade_band(self, grade: str | None) -> tuple[str | None, dict[str, Any] | None]:
        if not grade:
            return None, None
        g = grade.strip().upper()
        for band_name, info in self.grade_bands.items():
            if g in info["grades"]:
                return band_name, info
        return None, None

    def _find_city(self, city: str | None, region: str | None = None) -> tuple[str | None, str | None, dict[str, Any] | None]:
        if not city:
            return None, None, None
        target = city.strip().lower()
        search_regions = [region.upper()] if region and region.upper() in self.city_tiers else list(self.city_tiers.keys())
        for reg in search_regions:
            for tier_name, tier_info in self.city_tiers[reg].items():
                if any(c.lower() == target for c in tier_info["cities"]):
                    return reg, tier_name, tier_info
        return None, None, None

    @staticmethod
    def _result(
        decision: str,
        tool_name: str,
        explanation: str,
        violation_category: str = "NONE",
        allowed_amount: float | None = None,
        claimed_amount: float | None = None,
        currency: str | None = None,
        missing_fields: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        variance_amt = None
        variance_pct = None
        if allowed_amount is not None and claimed_amount is not None:
            variance_amt = round(claimed_amount - allowed_amount, 2)
            variance_pct = round((variance_amt / allowed_amount) * 100, 2) if allowed_amount > 0 else 0.0

        return {
            "tool": tool_name,
            "decision_category": decision,
            "violation_category": violation_category,
            "allowed_amount": allowed_amount,
            "claimed_amount": claimed_amount,
            "currency": currency,
            "variance_amount": variance_amt,
            "variance_pct": variance_pct,
            "missing_fields": missing_fields or [],
            "explanation": explanation,
            "metadata": metadata or {},
        }

    # ---------------------------------------------------------
    # 1. TRAVEL ELIGIBILITY
    # ---------------------------------------------------------
    def check_travel_eligibility(
        self,
        employee_id: str | None = None,
        grade: str | None = None,
        department: str | None = None,
        employment_status: str | None = None,
        destination: str | None = None,
        project_code: str | None = None,
    ) -> dict[str, Any]:
        missing = [
            k for k, v in {
                "employee_id": employee_id,
                "employee_grade": grade,
                "department": department,
                "travel_destination": destination,
                "project_code": project_code,
            }.items() if not v
        ]
        if missing:
            return self._result(
                "NEEDS_INFORMATION", "check_travel_eligibility",
                f"Missing required eligibility fields: {', '.join(missing)}.",
                violation_category="MISSING_REQUIRED_FIELDS", missing_fields=missing,
            )

        if any(destination.strip().lower() == r.lower() for r in self.eligibility_rules["restricted_destinations"]):
            return self._result(
                "HARD_FLAG", "check_travel_eligibility",
                f"Destination '{destination}' is on the restricted/sanctioned travel list.",
                violation_category="RESTRICTED_DESTINATION",
            )

        valid_prefixes = tuple(self.eligibility_rules["valid_project_prefixes"])
        if not project_code.strip().upper().startswith(valid_prefixes):
            return self._result(
                "NEEDS_INFORMATION", "check_travel_eligibility",
                f"Project code '{project_code}' is invalid. Must start with {', '.join(valid_prefixes)}.",
                violation_category="INVALID_PROJECT_CODE", missing_fields=["project_code"],
            )

        status = (employment_status or "ACTIVE").strip().upper()
        if status == "CONTRACTOR":
            return self._result(
                "HUMAN_REVIEW", "check_travel_eligibility",
                "External contractor travel depends on SOW terms; escalate to Finance Policy Desk.",
                violation_category="CONTRACTOR_SOW_REVIEW",
            )
        if status == "PROBATION":
            return self._result(
                "SOFT_FLAG", "check_travel_eligibility",
                "Employee on probation requires Direct Manager pre-approval for business travel.",
                violation_category="PROBATION_TRAVEL",
            )
        if status not in self.eligibility_rules["allowed_statuses"]:
            return self._result(
                "HARD_FLAG", "check_travel_eligibility",
                f"Employment status '{status}' is not eligible for corporate travel.",
                violation_category="INELIGIBLE_STATUS",
            )

        return self._result(
            "PASS", "check_travel_eligibility",
            f"Employee {employee_id} ({grade}, {department}) is eligible for travel to {destination} under {project_code}.",
        )

    # ---------------------------------------------------------
    # 2. FLIGHT POLICY
    # ---------------------------------------------------------
    def validate_flight_booking(
        self,
        grade: str | None = None,
        route_type: str | None = None,      # "domestic" | "intl_short" | "intl_long"
        booked_cabin: str | None = None,    # "Economy" | "Premium Economy" | "Business" | "First"
        days_in_advance: int | None = None,
        region: str = "IN",
        is_preferred_airline: bool = True,
    ) -> dict[str, Any]:
        missing = [
            k for k, v in {
                "employee_grade": grade,
                "route_type": route_type,
                "booked_cabin": booked_cabin,
                "days_in_advance": days_in_advance,
            }.items() if v is None or v == ""
        ]
        if missing:
            return self._result(
                "NEEDS_INFORMATION", "validate_flight_booking",
                f"Missing required flight fields: {', '.join(missing)}.",
                violation_category="MISSING_REQUIRED_FIELDS", missing_fields=missing,
            )

        band_name, band = self._get_grade_band(grade)
        if not band:
            return self._result("HUMAN_REVIEW", "validate_flight_booking", f"Unmapped employee grade '{grade}'.", "UNMAPPED_GRADE")

        cabin_rank = {"economy": 1, "premium economy": 2, "business": 3, "first": 4}
        route_map = {"domestic": "flight_dom", "intl_short": "flight_intl_short", "intl_long": "flight_intl_long"}
        r_key = route_map.get(route_type.lower())
        if not r_key:
            return self._result("HUMAN_REVIEW", "validate_flight_booking", f"Unmapped route_type '{route_type}'.", "UNMAPPED_ROUTE")

        entitled_cabin = band[r_key]
        booked_norm = booked_cabin.strip().lower()
        entitled_norm = entitled_cabin.lower()

        if booked_norm not in cabin_rank:
            return self._result("HUMAN_REVIEW", "validate_flight_booking", f"Unrecognized cabin class '{booked_cabin}'.", "UNMAPPED_CABIN")

        if cabin_rank[booked_norm] > cabin_rank[entitled_norm]:
            return self._result(
                "HARD_FLAG", "validate_flight_booking",
                f"Grade {grade} ({band_name}) is entitled to {entitled_cabin} for {route_type} flights; '{booked_cabin}' violates cabin policy.",
                violation_category="CABIN_CLASS_VIOLATION",
                metadata={"entitled_cabin": entitled_cabin, "booked_cabin": booked_cabin, "region": region},
            )

        adv_required = band["adv_days"]
        if days_in_advance < adv_required:
            return self._result(
                "SOFT_FLAG", "validate_flight_booking",
                f"Booked {days_in_advance} days in advance; {band_name} requires at least {adv_required} days.",
                violation_category="ADVANCE_BOOKING_WINDOW_BREACH",
                metadata={"required_days": adv_required, "actual_days": days_in_advance, "region": region},
            )

        if not is_preferred_airline:
            return self._result(
                "SOFT_FLAG", "validate_flight_booking",
                f"Non-preferred airline booked in {region}; requires >=15% fare savings or schedule justification.",
                violation_category="NON_PREFERRED_AIRLINE",
                metadata={"region": region},
            )

        return self._result(
            "PASS", "validate_flight_booking",
            f"Flight booking ({booked_cabin}, {days_in_advance}d advance) complies with {band_name} policy.",
            metadata={"entitled_cabin": entitled_cabin, "region": region},
        )

    # ---------------------------------------------------------
    # 3. HOTEL POLICY
    # ---------------------------------------------------------
    def validate_hotel_limit(
        self,
        grade: str | None = None,
        city: str | None = None,
        nightly_rate: float | None = None,
        nights: int = 1,
        travel_month: str | None = None,   # e.g., "Jan", "Nov"
        region: str | None = None,
    ) -> dict[str, Any]:
        missing = [
            k for k, v in {
                "employee_grade": grade,
                "travel_destination": city,
                "expense_amount": nightly_rate,
                "travel_dates": travel_month,
            }.items() if v is None or v == ""
        ]
        if missing:
            return self._result(
                "NEEDS_INFORMATION", "validate_hotel_limit",
                f"Missing required hotel validation fields: {', '.join(missing)}.",
                violation_category="MISSING_REQUIRED_FIELDS", missing_fields=missing,
            )

        band_name, band = self._get_grade_band(grade)
        reg, tier_name, tier_info = self._find_city(city, region)
        if not band or not tier_info or not reg:
            return self._result(
                "HUMAN_REVIEW", "validate_hotel_limit",
                f"No mapped city-tier hotel cap found for city='{city}', grade='{grade}'.",
                violation_category="UNMAPPED_DESTINATION_OR_GRADE",
            )

        cur = self.regions[reg]["currency"]
        base_cap = tier_info["hotel"] * band["hotel_mult"]
        is_peak = travel_month[:3].title() in tier_info["peak_months"]
        effective_cap = round(base_cap * (tier_info["peak_mult"] if is_peak else 1.0), 2)
        soft_limit = round(effective_cap * 1.15, 2)

        total_allowed = round(effective_cap * nights, 2)
        total_claimed = round(nightly_rate * nights, 2)

        if nightly_rate <= effective_cap:
            return self._result(
                "PASS", "validate_hotel_limit",
                f"Nightly rate {cur} {nightly_rate} is within the {city} ({tier_name}) cap of {cur} {effective_cap} ({'peak' if is_peak else 'standard'}).",
                allowed_amount=total_allowed, claimed_amount=total_claimed, currency=cur,
                metadata={"city": city, "tier": tier_name, "region": reg, "is_peak": is_peak, "nightly_cap": effective_cap},
            )
        if nightly_rate <= soft_limit:
            return self._result(
                "SOFT_FLAG", "validate_hotel_limit",
                f"Nightly rate {cur} {nightly_rate} exceeds {city} cap ({cur} {effective_cap}) by <=15% (soft limit {cur} {soft_limit}). Justification required.",
                violation_category="HOTEL_CAP_SOFT_OVERAGE",
                allowed_amount=total_allowed, claimed_amount=total_claimed, currency=cur,
                metadata={"city": city, "tier": tier_name, "region": reg, "is_peak": is_peak, "nightly_cap": effective_cap},
            )
        return self._result(
            "HARD_FLAG", "validate_hotel_limit",
            f"Nightly rate {cur} {nightly_rate} exceeds {city} cap ({cur} {effective_cap}) by >15% (hard breach above {cur} {soft_limit}).",
            violation_category="HOTEL_CAP_HARD_BREACH",
            allowed_amount=total_allowed, claimed_amount=total_claimed, currency=cur,
            metadata={"city": city, "tier": tier_name, "region": reg, "is_peak": is_peak, "nightly_cap": effective_cap},
        )

    # ---------------------------------------------------------
    # 4. MEAL & PER DIEM
    # ---------------------------------------------------------
    def calculate_meal_per_diem(
        self,
        city: str | None = None,
        total_days: int | None = None,
        region: str | None = None,
        provided_meals: dict[str, int] | None = None,  # e.g., {"breakfast": 2, "lunch": 1, "dinner": 0}
        claimed_amount: float | None = None,
    ) -> dict[str, Any]:
        missing = [k for k, v in {"travel_destination": city, "travel_dates": total_days}.items() if not v]
        if missing:
            return self._result(
                "NEEDS_INFORMATION", "calculate_meal_per_diem",
                f"Missing required per diem fields: {', '.join(missing)}.",
                violation_category="MISSING_REQUIRED_FIELDS", missing_fields=missing,
            )

        reg, tier_name, tier_info = self._find_city(city, region)
        if not tier_info or not reg:
            return self._result(
                "HUMAN_REVIEW", "calculate_meal_per_diem",
                f"No per diem rate found for destination '{city}'.",
                violation_category="UNMAPPED_DESTINATION",
            )

        cur = self.regions[reg]["currency"]
        daily_rate = tier_info["per_diem"]
        if total_days == 1:
            gross = daily_rate * 0.75
        else:
            gross = (daily_rate * 0.75 * 2) + (daily_rate * (total_days - 2))

        deductions = 0.0
        if provided_meals:
            deductions += provided_meals.get("breakfast", 0) * daily_rate * 0.20
            deductions += provided_meals.get("lunch", 0) * daily_rate * 0.30
            deductions += provided_meals.get("dinner", 0) * daily_rate * 0.50

        allowed = round(max(0.0, gross - deductions), 2)

        if claimed_amount is not None and claimed_amount > allowed:
            return self._result(
                "SOFT_FLAG", "calculate_meal_per_diem",
                f"Claimed per diem {cur} {claimed_amount} exceeds computed allowance {cur} {allowed} for {total_days}d in {city}.",
                violation_category="PER_DIEM_EXCEEDED",
                allowed_amount=allowed, claimed_amount=claimed_amount, currency=cur,
                metadata={"daily_rate": daily_rate, "deductions": round(deductions, 2), "region": reg},
            )

        return self._result(
            "PASS", "calculate_meal_per_diem",
            f"Allowed meal & incidental per diem for {total_days}d in {city} ({tier_name}): {cur} {allowed} (daily rate {cur} {daily_rate}, deductions {cur} {round(deductions, 2)}).",
            allowed_amount=allowed, claimed_amount=claimed_amount, currency=cur,
            metadata={"daily_rate": daily_rate, "deductions": round(deductions, 2), "region": reg},
        )

    # ---------------------------------------------------------
    # 5. EXPENSE CLAIM
    # ---------------------------------------------------------
    def validate_expense_claim(
        self,
        category: str | None = None,
        amount: float | None = None,
        region: str = "IN",
        units: int = 1,                  # days or attendees
        item_tag: str | None = None,     # optional prohibited item tag
    ) -> dict[str, Any]:
        missing = [k for k, v in {"expense_category": category, "expense_amount": amount}.items() if v is None or v == ""]
        if missing:
            return self._result(
                "NEEDS_INFORMATION", "validate_expense_claim",
                f"Missing required expense fields: {', '.join(missing)}.",
                violation_category="MISSING_REQUIRED_FIELDS", missing_fields=missing,
            )

        if item_tag and item_tag.strip().lower() in self.prohibited_items:
            return self._result(
                "HARD_FLAG", "validate_expense_claim",
                f"Item '{item_tag}' is strictly prohibited and non-reimbursable.",
                violation_category="PROHIBITED_EXPENSE_ITEM",
                allowed_amount=0.0, claimed_amount=amount, currency=self.regions.get(region.upper(), {}).get("currency"),
            )

        cat_norm = category.strip().lower()
        if cat_norm in self.prohibited_items:
            return self._result(
                "HARD_FLAG", "validate_expense_claim",
                f"Category '{category}' is strictly prohibited and non-reimbursable.",
                violation_category="PROHIBITED_EXPENSE_ITEM",
                allowed_amount=0.0, claimed_amount=amount, currency=self.regions.get(region.upper(), {}).get("currency"),
            )

        reg = region.upper()
        if reg not in self.regions or cat_norm not in self.expense_caps_usd:
            return self._result(
                "HUMAN_REVIEW", "validate_expense_claim",
                f"Expense category '{category}' in region '{region}' is not mapped to a standard cap.",
                violation_category="UNMAPPED_EXPENSE_CATEGORY",
            )

        rinfo = self.regions[reg]
        cinfo = self.expense_caps_usd[cat_norm]
        basis_key = [k for k in cinfo if k != "soft_overage_pct"][0]
        local_unit_cap = cinfo[basis_key] / rinfo["fx_to_usd"]
        allowed = round(local_unit_cap * max(1, units), 2)
        soft_limit = round(allowed * (1.0 + cinfo["soft_overage_pct"]), 2)
        cur = rinfo["currency"]

        if amount <= allowed:
            return self._result(
                "PASS", "validate_expense_claim",
                f"{category} claim of {cur} {amount} is within the allowed cap of {cur} {allowed}.",
                allowed_amount=allowed, claimed_amount=amount, currency=cur, metadata={"region": reg},
            )
        if amount <= soft_limit:
            return self._result(
                "SOFT_FLAG", "validate_expense_claim",
                f"{category} claim of {cur} {amount} exceeds cap ({cur} {allowed}) within soft band (up to {cur} {soft_limit}).",
                violation_category="EXPENSE_CAP_SOFT_OVERAGE",
                allowed_amount=allowed, claimed_amount=amount, currency=cur, metadata={"region": reg},
            )
        return self._result(
            "HARD_FLAG", "validate_expense_claim",
            f"{category} claim of {cur} {amount} exceeds hard limit ({cur} {soft_limit}).",
            violation_category="EXPENSE_CAP_HARD_BREACH",
            allowed_amount=allowed, claimed_amount=amount, currency=cur, metadata={"region": reg},
        )

    # ---------------------------------------------------------
    # 6. RECEIPT POLICY
    # ---------------------------------------------------------
    def verify_receipt_requirement(
        self,
        category: str | None = None,
        amount: float | None = None,
        region: str = "IN",
        has_itemized_receipt: bool | None = None,
        has_gstin: bool = True,
        lost_receipt_count_ytd: int = 0,
    ) -> dict[str, Any]:
        missing = [
            k for k, v in {
                "expense_category": category,
                "expense_amount": amount,
                "receipt_info": has_itemized_receipt,
            }.items() if v is None or v == ""
        ]
        if missing:
            return self._result(
                "NEEDS_INFORMATION", "verify_receipt_requirement",
                f"Missing required receipt fields: {', '.join(missing)}.",
                violation_category="MISSING_REQUIRED_FIELDS", missing_fields=missing,
            )

        reg = region.upper()
        rinfo = self.regions.get(reg)
        if not rinfo:
            return self._result("HUMAN_REVIEW", "verify_receipt_requirement", f"Unmapped region '{region}'.", "UNMAPPED_REGION")

        cur = rinfo["currency"]
        cat_norm = category.strip().lower()
        threshold = 0 if cat_norm in {"hotel", "flight", "client_entertainment"} else rinfo["receipt_threshold"]

        if amount >= threshold and not has_itemized_receipt:
            if lost_receipt_count_ytd >= 2:
                return self._result(
                    "HARD_FLAG", "verify_receipt_requirement",
                    f"Missing itemized {rinfo['tax_doc']} for {cur} {amount} (threshold {cur} {threshold}); annual limit of 2 Lost Receipt Affidavits already reached.",
                    violation_category="MISSING_RECEIPT_LIMIT_EXCEEDED", claimed_amount=amount, currency=cur,
                )
            return self._result(
                "SOFT_FLAG", "verify_receipt_requirement",
                f"Itemized {rinfo['tax_doc']} is mandatory for {category} >= {cur} {threshold}. Submit a Lost Receipt Affidavit (used {lost_receipt_count_ytd}/2 this year).",
                violation_category="MISSING_RECEIPT_AFFIDAVIT_REQUIRED", claimed_amount=amount, currency=cur,
            )

        if reg == "IN" and cat_norm == "hotel" and amount > 5000 and not has_gstin:
            return self._result(
                "SOFT_FLAG", "verify_receipt_requirement",
                "Indian domestic hotel invoice above INR 5,000 is missing company GSTIN; requires Finance tax review.",
                violation_category="MISSING_GSTIN", claimed_amount=amount, currency=cur,
            )

        return self._result(
            "PASS", "verify_receipt_requirement",
            f"Receipt requirement satisfied for {category} ({cur} {amount}, threshold {cur} {threshold}).",
            claimed_amount=amount, currency=cur,
        )

    # ---------------------------------------------------------
    # 7. MILEAGE POLICY
    # ---------------------------------------------------------
    def validate_mileage_claim(
        self,
        region: str = "IN",
        vehicle_type: str | None = None,    # "car" | "bike" | "ev"
        distance: float | None = None,
        origin: str | None = None,
        destination: str | None = None,
        is_daily_commute: bool = False,
        claimed_amount: float | None = None,
    ) -> dict[str, Any]:
        missing = [
            k for k, v in {
                "vehicle_type": vehicle_type,
                "distance": distance,
                "origin": origin,
                "travel_destination": destination,
            }.items() if v is None or v == ""
        ]
        if missing:
            return self._result(
                "NEEDS_INFORMATION", "validate_mileage_claim",
                f"Missing required mileage log fields: {', '.join(missing)}.",
                violation_category="MISSING_REQUIRED_FIELDS", missing_fields=missing,
            )

        if is_daily_commute:
            return self._result(
                "HARD_FLAG", "validate_mileage_claim",
                "Routine commute between home and primary office is strictly non-reimbursable.",
                violation_category="COMMUTE_EXCLUDED", allowed_amount=0.0, claimed_amount=claimed_amount,
            )

        reg = region.upper()
        rinfo = self.regions.get(reg)
        if not rinfo:
            return self._result("HUMAN_REVIEW", "validate_mileage_claim", f"Unmapped region '{region}'.", "UNMAPPED_REGION")

        v_norm = vehicle_type.strip().lower()
        rate_map = {"car": ("car_rate", 400), "bike": ("bike_rate", 150), "two-wheeler": ("bike_rate", 150), "ev": ("ev_rate", 400)}
        if v_norm not in rate_map:
            return self._result("HUMAN_REVIEW", "validate_mileage_claim", f"Unmapped vehicle_type '{vehicle_type}'.", "UNMAPPED_VEHICLE")

        rate_key, max_daily_dist = rate_map[v_norm]
        rate = rinfo[rate_key]
        unit = rinfo["mileage_unit"]
        cur = rinfo["currency"]
        allowed = round(distance * rate, 2)

        if distance > max_daily_dist:
            return self._result(
                "SOFT_FLAG", "validate_mileage_claim",
                f"Claimed distance {distance} {unit} exceeds single-day bound ({max_daily_dist} {unit}). Route verification required; computed allowance {cur} {allowed}.",
                violation_category="MILEAGE_DISTANCE_BOUND_EXCEEDED",
                allowed_amount=allowed, claimed_amount=claimed_amount or allowed, currency=cur,
            )

        if claimed_amount is not None and claimed_amount > allowed:
            return self._result(
                "SOFT_FLAG", "validate_mileage_claim",
                f"Claimed {cur} {claimed_amount} exceeds official mileage rate ({cur} {rate}/{unit} x {distance} = {cur} {allowed}).",
                violation_category="MILEAGE_RATE_EXCEEDED",
                allowed_amount=allowed, claimed_amount=claimed_amount, currency=cur,
            )

        return self._result(
            "PASS", "validate_mileage_claim",
            f"Reimbursable mileage: {distance} {unit} x {cur} {rate}/{unit} = {cur} {allowed}.",
            allowed_amount=allowed, claimed_amount=claimed_amount or allowed, currency=cur,
        )

    # ---------------------------------------------------------
    # 8. APPROVAL HIERARCHY
    # ---------------------------------------------------------
    def resolve_approval_hierarchy(
        self,
        grade: str | None,
        amount: float | None,
        region: str = "IN",
        violation_type: str = "PASS",
    ) -> dict[str, Any]:
        if violation_type == "NEEDS_INFORMATION":
            return self._result(
                "NEEDS_INFORMATION", "resolve_approval_hierarchy",
                "Cannot assign approver until missing required fields are supplied.",
                metadata={"required_approval_level": "NONE"},
            )

        band_name, _ = self._get_grade_band(grade)
        rinfo = self.regions.get(region.upper(), {"fx_to_usd": 1.0, "currency": "USD"})
        usd_eq = round((amount or 0.0) * rinfo["fx_to_usd"], 2)

        for row in self.approval_matrix:
            if row["violation_type"] != violation_type:
                continue
            if row["grade_band"] not in (band_name, "ALL"):
                continue
            if usd_eq <= row["max_amount_usd_eq"]:
                return self._result(
                    violation_type, "resolve_approval_hierarchy",
                    f"For {violation_type} ({band_name or 'ALL'}, USD {usd_eq} eq.), mandatory human approver is: {row['approver']}.",
                    metadata={"required_approval_level": row["approver"], "amount_usd_eq": usd_eq},
                )

        return self._result(
            "HUMAN_REVIEW", "resolve_approval_hierarchy",
            "No matching approval slab found; escalate to Finance Policy Desk.",
            metadata={"required_approval_level": "Finance Policy Desk", "amount_usd_eq": usd_eq},
        )

    # ---------------------------------------------------------
    # 9. EXCEPTION HANDLING (AGGREGATOR & HARD-FLAG LOCK)
    # ---------------------------------------------------------
    def classify_exception_route(self, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Deterministically aggregates all symbolic tool outputs.
        Precedence: HARD_FLAG > NEEDS_INFORMATION > HUMAN_REVIEW > SOFT_FLAG > PASS.
        """
        decisions = [r["decision_category"] for r in tool_results]
        missing = sorted({f for r in tool_results for f in r.get("missing_fields", [])})
        violations = [r["violation_category"] for r in tool_results if r.get("violation_category") not in (None, "NONE")]

        if "HARD_FLAG" in decisions:
            final = "HARD_FLAG"
            route = "BLOCKED (Requires VP + CFO Written Exception)"
        elif "NEEDS_INFORMATION" in decisions:
            final = "NEEDS_INFORMATION"
            route = "RETURN_TO_EMPLOYEE"
        elif "HUMAN_REVIEW" in decisions:
            final = "HUMAN_REVIEW"
            route = "Finance Policy Desk"
        elif "SOFT_FLAG" in decisions:
            final = "SOFT_FLAG"
            route = "MANAGER_OR_DEPT_HEAD_EXCEPTION_QUEUE"
        else:
            final = "PASS"
            route = "STANDARD_APPROVAL_WORKFLOW"

        return self._result(
            final, "classify_exception_route",
            f"Final exception route: {final} -> {route}. Violations: {violations or ['NONE']}.",
            violation_category=violations[0] if violations else "NONE",
            missing_fields=missing,
            metadata={"route": route, "all_violations": violations},
        )

    # ---------------------------------------------------------
    # 10. POLICY RECOMMENDATIONS (SYMBOLIC HINTS FOR LLM)
    # ---------------------------------------------------------
    def build_policy_recommendation_hints(self, tool_result: dict[str, Any], region: str = "IN") -> dict[str, Any]:
        """
        Generates structured, non-binding recommendation hints from symbolic gap data
        so the <=8B LLM clearly separates mandatory rules from compliant alternatives.
        """
        decision = tool_result["decision_category"]
        vcat = tool_result.get("violation_category", "NONE")
        meta = tool_result.get("metadata", {})
        reg = (meta.get("region") or region or "IN").upper()
        vinfo = self.vendors.get(reg, {})
        hints: list[str] = []

        if decision == "NEEDS_INFORMATION":
            hints.append(f"Request the employee to provide: {', '.join(tool_result.get('missing_fields', []))}.")
        elif vcat in ("HOTEL_CAP_SOFT_OVERAGE", "HOTEL_CAP_HARD_BREACH"):
            cap = meta.get("nightly_cap")
            cur = tool_result.get("currency")
            hotels = ", ".join(vinfo.get("hotels", []))
            hints.append(f"Target a nightly rate <= {cur} {cap} via {vinfo.get('portal', 'corporate portal')}.")
            if hotels:
                hints.append(f"Check negotiated corporate rates with preferred chains: {hotels}.")
            if decision == "SOFT_FLAG":
                hints.append("Attach written justification (e.g., conference proximity) for exception review.")
        elif vcat == "CABIN_CLASS_VIOLATION":
            hints.append(f"Rebook in entitled cabin class ({meta.get('entitled_cabin')}) or use personal miles for upgrade.")
        elif vcat == "ADVANCE_BOOKING_WINDOW_BREACH":
            hints.append(f"Attach client email or urgency ticket explaining why booking missed the {meta.get('required_days')}-day window.")
        elif vcat == "NON_PREFERRED_AIRLINE":
            airlines = ", ".join(vinfo.get("airlines", []))
            hints.append(f"Prefer contracted carriers ({airlines}) via {vinfo.get('portal', 'corporate portal')}, or attach >=15% savings screenshot.")
        elif vcat == "MISSING_RECEIPT_AFFIDAVIT_REQUIRED":
            hints.append("Request a duplicate itemized invoice from the merchant, or submit a signed Lost Receipt Affidavit.")
        elif decision == "PASS":
            hints.append("No corrective action required; submit through standard approval workflow.")
        else:
            hints.append("Escalate with supporting documentation to the designated human reviewer.")

        return {
            "tool": "build_policy_recommendation_hints",
            "mandatory_decision": decision,
            "symbolic_gap": {
                "allowed_amount": tool_result.get("allowed_amount"),
                "claimed_amount": tool_result.get("claimed_amount"),
                "variance_amount": tool_result.get("variance_amount"),
                "variance_pct": tool_result.get("variance_pct"),
            },
            "non_binding_recommendations": hints,
            "advisory_notice": "AI assessment only. Final approval requires authorized human sign-off.",
        }


if __name__ == "__main__":
    engine = SymbolicPolicyEngine()

    r1 = engine.check_travel_eligibility("EMP104", "L3", "Engineering", "ACTIVE", "Mumbai", "PRJ-2026-09")
    r2 = engine.validate_flight_booking("L3", "domestic", "Economy", days_in_advance=10, region="IN")
    r3 = engine.validate_hotel_limit("L3", "Mumbai", nightly_rate=12000, nights=2, travel_month="Jan", region="IN")
    r4 = engine.calculate_meal_per_diem("Mumbai", total_days=3, region="IN", provided_meals={"breakfast": 2})
    r5 = engine.validate_expense_claim("ground_transport", amount=4200, region="IN", units=1)
    r6 = engine.verify_receipt_requirement("hotel", amount=24000, region="IN", has_itemized_receipt=True, has_gstin=True)
    r7 = engine.validate_mileage_claim("IN", "car", distance=85, origin="Home", destination="Airport")

    route = engine.classify_exception_route([r1, r2, r3, r4, r5, r6, r7])
    approver = engine.resolve_approval_hierarchy("L3", amount=24000, region="IN", violation_type=route["decision_category"])
    hints = engine.build_policy_recommendation_hints(r3, region="IN")

    for res in [r1, r2, r3, r4, r5, r6, r7, route, approver]:
        print(f"[{res['decision_category']:<17}] {res['tool']}: {res['explanation']}")
    print("Recommendations:", hints["non_binding_recommendations"])