"""
tests/test_verify_normalizers.py
----------------------------------
Unit tests for verify/normalizers.py — 30 normalization cases.

Run with:
    pytest tests/test_verify_normalizers.py -v
"""
import pytest
from datetime import date

from verify.normalizers import (
    norm_amount, norm_pct, norm_date, norm_section,
    dates_equal, sections_equal, numerically_equal,
)


# ---------------------------------------------------------------------------
# norm_amount — rupee / monetary amounts
# ---------------------------------------------------------------------------

class TestNormAmount:
    def test_plain_number(self):
        assert norm_amount("500000") == 500000.0

    def test_lakh(self):
        assert norm_amount("₹5 lakh") == 500000.0

    def test_lakhs_plural(self):
        assert norm_amount("Rs. 10 lakhs") == 1_000_000.0

    def test_crore(self):
        assert norm_amount("₹1.5 crore") == 15_000_000.0

    def test_crores(self):
        assert norm_amount("INR 2 crores") == 20_000_000.0

    def test_comma_separated(self):
        assert norm_amount("5,00,000") == 500_000.0

    def test_thousand(self):
        assert norm_amount("Rs 50 thousand") == 50_000.0

    def test_no_currency_symbol(self):
        assert norm_amount("5 lakh") == 500_000.0

    def test_none_for_text(self):
        assert norm_amount("no amount here") is None


# ---------------------------------------------------------------------------
# norm_pct — percentages
# ---------------------------------------------------------------------------

class TestNormPct:
    def test_percent_symbol(self):
        assert norm_pct("5%") == 5.0

    def test_per_cent(self):
        assert norm_pct("5 per cent") == 5.0

    def test_percent_word(self):
        assert norm_pct("7.5percent") == 7.5

    def test_with_pa(self):
        assert norm_pct("10% p.a.") == 10.0

    def test_per_annum(self):
        assert norm_pct("8 per cent per annum") == 8.0

    def test_none_for_text(self):
        assert norm_pct("five rupees") is None

    def test_zero_pct(self):
        assert norm_pct("0%") == 0.0


# ---------------------------------------------------------------------------
# norm_date — Indian date formats
# ---------------------------------------------------------------------------

class TestNormDate:
    def test_ddmmyyyy_slash(self):
        assert norm_date("12/03/2024") == date(2024, 3, 12)

    def test_ddmmyyyy_dash(self):
        assert norm_date("01-04-2023") == date(2023, 4, 1)

    def test_word_date(self):
        assert norm_date("12 March 2024") == date(2024, 3, 12)

    def test_word_date_short_month(self):
        assert norm_date("5 Jan 2022") == date(2022, 1, 5)

    def test_iso_date(self):
        assert norm_date("2024-03-12") == date(2024, 3, 12)

    def test_fy(self):
        assert norm_date("FY 2023-24") == date(2023, 4, 1)

    def test_fy_dash(self):
        assert norm_date("FY 2022-23") == date(2022, 4, 1)

    def test_none_for_garbage(self):
        assert norm_date("not a date") is None


# ---------------------------------------------------------------------------
# dates_equal
# ---------------------------------------------------------------------------

class TestDatesEqual:
    def test_equal_formats(self):
        assert dates_equal("12/03/2024", "12 March 2024") is True

    def test_not_equal(self):
        assert dates_equal("12/03/2024", "13/03/2024") is False

    def test_unparseable(self):
        assert dates_equal("12/03/2024", "no date") is None


# ---------------------------------------------------------------------------
# norm_section
# ---------------------------------------------------------------------------

class TestNormSection:
    def test_basic(self):
        assert norm_section("Section 80C") == "section80c"

    def test_with_sub(self):
        assert norm_section("Section 80C(2)") == "section80c(2)"

    def test_regulation(self):
        assert norm_section("Regulation 52(4)(a)") == "regulation52(4)(a)"

    def test_lowercase_input(self):
        assert norm_section("section 10") == "section10"


# ---------------------------------------------------------------------------
# sections_equal
# ---------------------------------------------------------------------------

class TestSectionsEqual:
    def test_equal(self):
        assert sections_equal("Section 80C(2)", "section 80C(2)") is True

    def test_not_equal(self):
        assert sections_equal("Section 80C(2)", "Section 80D(2)") is False


# ---------------------------------------------------------------------------
# numerically_equal
# ---------------------------------------------------------------------------

class TestNumericallyEqual:
    def test_equal(self):
        assert numerically_equal(500_000.0, 500_000.0) is True

    def test_not_equal(self):
        assert numerically_equal(5.0, 10.0) is False

    def test_none_input(self):
        assert numerically_equal(None, 5.0) is None

    def test_zero_zero(self):
        assert numerically_equal(0.0, 0.0) is True

    def test_floating_point_close(self):
        assert numerically_equal(5.000000001, 5.0) is True
