import pytest
from src.tools.policy_engine import SymbolicPolicyEngine


@pytest.fixture(scope="module")
def engine() -> SymbolicPolicyEngine:
    return SymbolicPolicyEngine()


# =========================================================
# 1. TRAVEL ELIGIBILITY (check_travel_eligibility)
# =========================================================
class TestTravelEligibility:
    def test_pass_active_employee(self, engine):
        res = engine.check_travel_eligibility(
            employee_id="EMP001", grade="L3", department="Engineering",
            employment_status="ACTIVE", destination="Mumbai", project_code="PRJ-2026-01"
        )
        assert res["decision_category"] == "PASS"

    def test_needs_information_missing_fields(self, engine):
        res = engine.check_travel_eligibility(employee_id="EMP001", grade=None, department="Sales")
        assert res["decision_category"] == "NEEDS_INFORMATION"
        assert "employee_grade" in res["missing_fields"]
        assert "project_code" in res["missing_fields"]

    def test_needs_information_invalid_project_prefix(self, engine):
        res = engine.check_travel_eligibility(
            employee_id="EMP001", grade="L3", department="Sales",
            employment_status="ACTIVE", destination="Delhi", project_code="BAD-999"
        )
        assert res["decision_category"] == "NEEDS_INFORMATION"
        assert res["violation_category"] == "INVALID_PROJECT_CODE"

    def test_hard_flag_restricted_destination(self, engine):
        res = engine.check_travel_eligibility(
            employee_id="EMP001", grade="L5", department="Sales",
            employment_status="ACTIVE", destination="North Korea", project_code="CLI-101"
        )
        assert res["decision_category"] == "HARD_FLAG"
        assert res["violation_category"] == "RESTRICTED_DESTINATION"

    def test_soft_flag_probation(self, engine):
        res = engine.check_travel_eligibility(
            employee_id="EMP002", grade="L2", department="Engineering",
            employment_status="PROBATION", destination="Bengaluru", project_code="RND-42"
        )
        assert res["decision_category"] == "SOFT_FLAG"
        assert res["violation_category"] == "PROBATION_TRAVEL"

    def test_human_review_contractor(self, engine):
        res = engine.check_travel_eligibility(
            employee_id="CON009", grade="L4", department="Consulting",
            employment_status="CONTRACTOR", destination="Pune", project_code="CLI-88"
        )
        assert res["decision_category"] == "HUMAN_REVIEW"
        assert res["violation_category"] == "CONTRACTOR_SOW_REVIEW"


# =========================================================
# 2. FLIGHT POLICY (validate_flight_booking)
# =========================================================
class TestFlightPolicy:
    def test_pass_compliant_booking(self, engine):
        res = engine.validate_flight_booking(
            grade="L3", route_type="domestic", booked_cabin="Economy",
            days_in_advance=15, region="IN", is_preferred_airline=True
        )
        assert res["decision_category"] == "PASS"

    def test_hard_flag_cabin_violation(self, engine):
        res = engine.validate_flight_booking(
            grade="L3", route_type="domestic", booked_cabin="Business",
            days_in_advance=20, region="IN"
        )
        assert res["decision_category"] == "HARD_FLAG"
        assert res["violation_category"] == "CABIN_CLASS_VIOLATION"

    def test_soft_flag_late_booking(self, engine):
        res = engine.validate_flight_booking(
            grade="L3", route_type="domestic", booked_cabin="Economy",
            days_in_advance=5, region="IN"
        )
        assert res["decision_category"] == "SOFT_FLAG"
        assert res["violation_category"] == "ADVANCE_BOOKING_WINDOW_BREACH"

    def test_soft_flag_non_preferred_airline(self, engine):
        res = engine.validate_flight_booking(
            grade="L5", route_type="intl_long", booked_cabin="Business",
            days_in_advance=10, region="US", is_preferred_airline=False
        )
        assert res["decision_category"] == "SOFT_FLAG"
        assert res["violation_category"] == "NON_PREFERRED_AIRLINE"

    def test_needs_information_missing_days(self, engine):
        res = engine.validate_flight_booking(grade="L4", route_type="domestic", booked_cabin="Economy", days_in_advance=None)
        assert res["decision_category"] == "NEEDS_INFORMATION"
        assert "days_in_advance" in res["missing_fields"]


# =========================================================
# 3. HOTEL POLICY (validate_hotel_limit)
# =========================================================
class TestHotelPolicy:
    def test_pass_standard_season_mumbai_l3(self, engine):
        # Mumbai Tier-1 L1-L4 standard cap = INR 9,500
        res = engine.validate_hotel_limit(grade="L3", city="Mumbai", nightly_rate=9500, nights=2, travel_month="Jun", region="IN")
        assert res["decision_category"] == "PASS"
        assert res["allowed_amount"] == 19000.0
        assert res["currency"] == "INR"

    def test_pass_peak_season_mumbai_l3(self, engine):
        # Mumbai Tier-1 peak (Jan) = 9,500 * 1.15 = INR 10,925
        res = engine.validate_hotel_limit(grade="L3", city="Mumbai", nightly_rate=10900, nights=1, travel_month="Jan", region="IN")
        assert res["decision_category"] == "PASS"
        assert res["metadata"]["is_peak"] is True
        assert res["allowed_amount"] == 10925.0

    def test_soft_flag_within_15_pct_band(self, engine):
        # Peak cap 10,925; +15% soft limit = 12,563.75
        res = engine.validate_hotel_limit(grade="L3", city="Mumbai", nightly_rate=12000, nights=2, travel_month="Jan", region="IN")
        assert res["decision_category"] == "SOFT_FLAG"
        assert res["violation_category"] == "HOTEL_CAP_SOFT_OVERAGE"
        assert res["allowed_amount"] == 21850.0
        assert res["claimed_amount"] == 24000.0

    def test_hard_flag_above_15_pct_band(self, engine):
        res = engine.validate_hotel_limit(grade="L3", city="Mumbai", nightly_rate=14000, nights=1, travel_month="Jun", region="IN")
        assert res["decision_category"] == "HARD_FLAG"
        assert res["violation_category"] == "HOTEL_CAP_HARD_BREACH"

    def test_human_review_unmapped_city(self, engine):
        res = engine.validate_hotel_limit(grade="L3", city="Thimphu", nightly_rate=8000, nights=1, travel_month="Jun")
        assert res["decision_category"] == "HUMAN_REVIEW"
        assert res["violation_category"] == "UNMAPPED_DESTINATION_OR_GRADE"


# =========================================================
# 4. MEAL & PER DIEM (calculate_meal_per_diem)
# =========================================================
class TestMealPerDiem:
    def test_pass_single_day_75_pct(self, engine):
        # Mumbai daily = 2,200; 1-day = 75% = 1,650
        res = engine.calculate_meal_per_diem(city="Mumbai", total_days=1, region="IN")
        assert res["decision_category"] == "PASS"
        assert res["allowed_amount"] == 1650.0

    def test_pass_multiday_with_meal_deductions(self, engine):
        # 3 days in Mumbai: (2200*0.75*2) + 2200 = 5,500
        # Deductions: 2 breakfasts (2 * 440 = 880) + 1 lunch (660) = 1,540 -> Net = 3,960
        res = engine.calculate_meal_per_diem(
            city="Mumbai", total_days=3, region="IN",
            provided_meals={"breakfast": 2, "lunch": 1, "dinner": 0}
        )
        assert res["decision_category"] == "PASS"
        assert res["allowed_amount"] == 3960.0

    def test_soft_flag_overclaimed_per_diem(self, engine):
        res = engine.calculate_meal_per_diem(city="New York", total_days=2, region="US", claimed_amount=200.0)
        # 2 days in NY = 95 * 0.75 * 2 = 142.5
        assert res["decision_category"] == "SOFT_FLAG"
        assert res["allowed_amount"] == 142.5


# =========================================================
# 5. EXPENSE CLAIM (validate_expense_claim)
# =========================================================
class TestExpenseClaim:
    def test_pass_within_cap(self, engine):
        # IN ground_transport cap = 60 / 0.012 = INR 5,000
        res = engine.validate_expense_claim(category="ground_transport", amount=4500, region="IN", units=1)
        assert res["decision_category"] == "PASS"
        assert res["allowed_amount"] == 5000.0

    def test_soft_flag_within_overage_band(self, engine):
        # +20% soft band = up to INR 6,000
        res = engine.validate_expense_claim(category="ground_transport", amount=5800, region="IN", units=1)
        assert res["decision_category"] == "SOFT_FLAG"
        assert res["violation_category"] == "EXPENSE_CAP_SOFT_OVERAGE"

    def test_hard_flag_prohibited_item(self, engine):
        res = engine.validate_expense_claim(category="hotel", amount=1200, region="IN", item_tag="minibar")
        assert res["decision_category"] == "HARD_FLAG"
        assert res["violation_category"] == "PROHIBITED_EXPENSE_ITEM"
        assert res["allowed_amount"] == 0.0

    def test_human_review_unmapped_category(self, engine):
        res = engine.validate_expense_claim(category="coworking_day_pass", amount=1500, region="IN")
        assert res["decision_category"] == "HUMAN_REVIEW"


# =========================================================
# 6. RECEIPT POLICY (verify_receipt_requirement)
# =========================================================
class TestReceiptPolicy:
    def test_pass_below_threshold_no_receipt(self, engine):
        # IN threshold = INR 500
        res = engine.verify_receipt_requirement(category="meals", amount=400, region="IN", has_itemized_receipt=False)
        assert res["decision_category"] == "PASS"

    def test_soft_flag_missing_receipt_affidavit(self, engine):
        res = engine.verify_receipt_requirement(category="meals", amount=800, region="IN", has_itemized_receipt=False, lost_receipt_count_ytd=1)
        assert res["decision_category"] == "SOFT_FLAG"
        assert res["violation_category"] == "MISSING_RECEIPT_AFFIDAVIT_REQUIRED"

    def test_hard_flag_third_lost_receipt(self, engine):
        res = engine.verify_receipt_requirement(category="meals", amount=800, region="IN", has_itemized_receipt=False, lost_receipt_count_ytd=2)
        assert res["decision_category"] == "HARD_FLAG"
        assert res["violation_category"] == "MISSING_RECEIPT_LIMIT_EXCEEDED"

    def test_soft_flag_missing_gstin_india_hotel(self, engine):
        res = engine.verify_receipt_requirement(category="hotel", amount=9000, region="IN", has_itemized_receipt=True, has_gstin=False)
        assert res["decision_category"] == "SOFT_FLAG"
        assert res["violation_category"] == "MISSING_GSTIN"


# =========================================================
# 7. MILEAGE POLICY (validate_mileage_claim)
# =========================================================
class TestMileagePolicy:
    def test_pass_india_car_mileage(self, engine):
        # IN car rate = INR 14/km -> 100 km = INR 1,400
        res = engine.validate_mileage_claim(region="IN", vehicle_type="car", distance=100, origin="Home", destination="Airport")
        assert res["decision_category"] == "PASS"
        assert res["allowed_amount"] == 1400.0

    def test_hard_flag_daily_commute(self, engine):
        res = engine.validate_mileage_claim(
            region="IN", vehicle_type="car", distance=30,
            origin="Home", destination="Office", is_daily_commute=True
        )
        assert res["decision_category"] == "HARD_FLAG"
        assert res["violation_category"] == "COMMUTE_EXCLUDED"

    def test_soft_flag_distance_bound_exceeded(self, engine):
        # Bike daily bound = 150 km
        res = engine.validate_mileage_claim(region="IN", vehicle_type="bike", distance=180, origin="Pune", destination="Mumbai")
        assert res["decision_category"] == "SOFT_FLAG"
        assert res["violation_category"] == "MILEAGE_DISTANCE_BOUND_EXCEEDED"

    def test_needs_information_missing_logs(self, engine):
        res = engine.validate_mileage_claim(region="US", vehicle_type="car", distance=50, origin=None, destination="Client")
        assert res["decision_category"] == "NEEDS_INFORMATION"
        assert "origin" in res["missing_fields"]


# =========================================================
# 8. APPROVAL HIERARCHY (resolve_approval_hierarchy)
# =========================================================
class TestApprovalHierarchy:
    def test_pass_l3_direct_manager(self, engine):
        res = engine.resolve_approval_hierarchy(grade="L3", amount=20000, region="IN", violation_type="PASS")
        assert res["metadata"]["required_approval_level"] == "Direct Manager"

    def test_soft_flag_l3_dept_head(self, engine):
        # INR 200,000 * 0.012 = USD 2,400 (> 1,500, <= 5,000) -> Department Head
        res = engine.resolve_approval_hierarchy(grade="L3", amount=200000, region="IN", violation_type="SOFT_FLAG")
        assert res["metadata"]["required_approval_level"] == "Department Head"

    def test_hard_flag_blocked(self, engine):
        res = engine.resolve_approval_hierarchy(grade="L3", amount=5000, region="IN", violation_type="HARD_FLAG")
        assert "BLOCKED" in res["metadata"]["required_approval_level"]


# =========================================================
# 9. EXCEPTION HANDLING (classify_exception_route)
# =========================================================
class TestExceptionRoute:
    def test_hard_flag_overrides_everything(self, engine):
        results = [
            {"decision_category": "PASS", "violation_category": "NONE", "missing_fields": []},
            {"decision_category": "SOFT_FLAG", "violation_category": "ADVANCE_BOOKING_WINDOW_BREACH", "missing_fields": []},
            {"decision_category": "HARD_FLAG", "violation_category": "PROHIBITED_EXPENSE_ITEM", "missing_fields": []},
        ]
        route = engine.classify_exception_route(results)
        assert route["decision_category"] == "HARD_FLAG"

    def test_needs_info_precedes_soft_flag(self, engine):
        results = [
            {"decision_category": "SOFT_FLAG", "violation_category": "PROBATION_TRAVEL", "missing_fields": []},
            {"decision_category": "NEEDS_INFORMATION", "violation_category": "MISSING_REQUIRED_FIELDS", "missing_fields": ["travel_dates"]},
        ]
        route = engine.classify_exception_route(results)
        assert route["decision_category"] == "NEEDS_INFORMATION"
        assert "travel_dates" in route["missing_fields"]


# =========================================================
# 10. POLICY RECOMMENDATIONS (build_policy_recommendation_hints)
# =========================================================
class TestPolicyRecommendations:
    def test_hints_include_advisory_notice_and_gap(self, engine):
        hotel_res = engine.validate_hotel_limit(grade="L3", city="Mumbai", nightly_rate=12000, nights=2, travel_month="Jan", region="IN")
        hints = engine.build_policy_recommendation_hints(hotel_res, region="IN")
        assert hints["mandatory_decision"] == "SOFT_FLAG"
        assert hints["symbolic_gap"]["variance_amount"] == 2150.0
        assert len(hints["non_binding_recommendations"]) >= 2
        assert "Final approval requires authorized human sign-off" in hints["advisory_notice"]