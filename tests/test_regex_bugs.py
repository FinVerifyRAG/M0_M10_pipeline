"""
tests/test_regex_bugs.py
------------------------
Regression tests for Bugs 1–3 and related design fixes.

Bug 1  PCT regex misses almost every percentage.
Bug 2  SEC regex truncates or misses section references.
Bug 3  INR regex misses common Indian currency forms.
Also covers:
  - Basis-points normalization (Section 4 / bps requirement).
  - Date parser extensions: AY, dd.mm.yyyy, ordinal forms (Issue 18).
  - norm_amount extended forms (Bug 3).
  - extract_section_list for list forms (Bug 2).

Run with:
    pytest tests/test_regex_bugs.py -v
"""
import pytest
from datetime import date

from verify.normalizers import (
    norm_pct,
    norm_amount,
    norm_bps,
    norm_date,
    norm_section,
    extract_section_list,
    is_bps,
    PCT,
    SEC,
    _BPS_RE,
    fy_from_ay,
    _clean_pdf_currency,
)


# ===========================================================================
# BUG 1 - PCT regex
# ===========================================================================

class TestPCTBug1:
    """
    Regression tests for Bug 1: PCT word-boundary failure.

    Required forms from the spec:
        5%          5 %         5% of income    10%     10.5%
        TDS at 10%. 5 per cent  5 percent       5percent
    """

    # ---------- bare % forms ----------
    def test_bare_5_pct(self):
        """'5%' must be extracted (was failing before fix)."""
        assert norm_pct("5%") == pytest.approx(5.0)

    def test_bare_5_pct_with_space(self):
        """'5 %' (space before percent symbol) must be extracted."""
        assert norm_pct("5 %") == pytest.approx(5.0)

    def test_10_pct(self):
        assert norm_pct("10%") == pytest.approx(10.0)

    def test_decimal_pct(self):
        assert norm_pct("10.5%") == pytest.approx(10.5)

    # ---------- in context ----------
    def test_pct_of_income(self):
        """'5% of income' - trailing words must not prevent extraction."""
        assert norm_pct("5% of income") == pytest.approx(5.0)

    def test_tds_pct_with_punct(self):
        """'TDS at 10%.' - preceding text and trailing period must be ignored."""
        assert norm_pct("TDS at 10%.") == pytest.approx(10.0)

    # ---------- word forms ----------
    def test_per_cent_spaced(self):
        assert norm_pct("5 per cent") == pytest.approx(5.0)

    def test_percent_word(self):
        assert norm_pct("5 percent") == pytest.approx(5.0)

    def test_percent_nospace(self):
        """'5percent' - no space between digit and word form."""
        assert norm_pct("5percent") == pytest.approx(5.0)

    # ---------- per-annum suffix ----------
    def test_pct_pa(self):
        assert norm_pct("5% p.a.") == pytest.approx(5.0)

    def test_pct_per_annum(self):
        assert norm_pct("8 per cent per annum") == pytest.approx(8.0)

    # ---------- sentence-boundary ----------
    def test_pct_end_of_sentence(self):
        """Percentage at end of sentence (no trailing space)."""
        assert norm_pct("The TDS rate is 10%") == pytest.approx(10.0)

    def test_pct_followed_by_comma(self):
        assert norm_pct("10%, subject to conditions") == pytest.approx(10.0)

    # ---------- zero / edge cases ----------
    def test_zero_pct(self):
        assert norm_pct("0%") == pytest.approx(0.0)

    def test_large_pct(self):
        assert norm_pct("100%") == pytest.approx(100.0)

    def test_no_pct(self):
        assert norm_pct("rupees five hundred") is None

    # ---------- PCT regex findall ----------
    def test_pct_findall_in_text(self):
        """PCT.findall should find all percentage tokens in a block of text."""
        text = "TDS at 5% and surcharge at 10% applies."
        matches = PCT.findall(text)
        assert "5" in matches and "10" in matches

    def test_pct_does_not_match_plain_digit(self):
        assert PCT.search("just 5 rupees") is None


# ===========================================================================
# BUG 2 - SEC regex
# ===========================================================================

class TestSECBug2:
    """
    Regression tests for Bug 2: SEC regex truncation/missing forms.
    """

    # ---------- hyphenated identifiers ----------
    def test_section_194_ia_not_truncated(self):
        """'section 194-IA' must NOT be captured as 'section 194'."""
        ids = extract_section_list("section 194-IA")
        assert ids, "Expected at least one ID"
        assert ids[0] == "194-IA", f"Got {ids[0]!r} instead of '194-IA'"

    def test_section_194_ia_norm(self):
        """norm_section preserves the hyphen."""
        assert "194-ia" in norm_section("section 194-IA")

    # ---------- list forms ----------
    def test_sections_and_list(self):
        """'Sections 80C and 80D' must yield ['80C', '80D']."""
        ids = extract_section_list("Sections 80C and 80D")
        assert "80C" in ids and "80D" in ids, f"Got {ids}"

    def test_sections_comma_list(self):
        """'Sections 80C, 80D and 80G' -> ['80C', '80D', '80G']."""
        ids = extract_section_list("Sections 80C, 80D and 80G")
        assert set(ids) >= {"80C", "80D", "80G"}, f"Got {ids}"

    # ---------- u/s form ----------
    def test_us_form(self):
        """'u/s 80C' must be extracted."""
        ids = extract_section_list("u/s 80C")
        assert ids and ids[0] == "80C", f"Got {ids}"

    # ---------- subclauses ----------
    def test_section_10_1(self):
        """'Section 10(1)' must be captured."""
        ids = extract_section_list("Section 10(1)")
        assert ids and ids[0].startswith("10"), f"Got {ids}"

    def test_rule_6_2(self):
        """'Rule 6(2)' must be captured."""
        ids = extract_section_list("Rule 6(2)")
        assert ids, f"Expected a match, got {ids}"

    def test_regulation_18a(self):
        """'Regulation 18A' must be captured."""
        ids = extract_section_list("Regulation 18A")
        assert ids and "18A" in ids[0], f"Got {ids}"

    def test_clause_4b(self):
        """'Clause 4(b)' must be captured."""
        ids = extract_section_list("Clause 4(b)")
        assert ids, f"Expected a match, got {ids}"

    # ---------- SEC.search returns a match ----------
    def test_sec_pattern_matches_us(self):
        assert SEC.search("u/s 80C") is not None

    def test_sec_pattern_matches_sections_plural(self):
        assert SEC.search("sections 80C and 80D") is not None

    def test_sec_pattern_matches_regulation(self):
        assert SEC.search("Regulation 18A") is not None

    def test_sec_pattern_does_not_match_plain_text(self):
        assert SEC.search("no references here") is None


# ===========================================================================
# BUG 3 - INR regex + bps
# ===========================================================================

class TestINRBug3:
    """
    Regression tests for Bug 3: extended Indian currency forms.
    """

    # ---------- standard forms ----------
    def test_rupee_symbol_lakh(self):
        assert norm_amount("5,00,000") == pytest.approx(500000.0)

    def test_rs_dot_comma(self):
        assert norm_amount("Rs. 5,00,000") == pytest.approx(500000.0)

    def test_rs_dot_no_space_slash(self):
        """'Rs.5,00,000/-' must be parsed."""
        assert norm_amount("Rs.5,00,000/-") == pytest.approx(500000.0)

    def test_inr_plain(self):
        assert norm_amount("INR 500000") == pytest.approx(500000.0)

    def test_5_lakh_rupees(self):
        """'5 lakh rupees' - unit before currency word."""
        assert norm_amount("5 lakh rupees") == pytest.approx(500000.0)

    def test_rupee_symbol_lakh_space(self):
        assert norm_amount("5 lakh") == pytest.approx(500000.0)

    def test_rs_space_lakh(self):
        assert norm_amount("Rs 5 lakh") == pytest.approx(500000.0)

    def test_backtick_as_rupee(self):
        """`5,00,000 - backtick as PDF-corrupted rupee symbol."""
        assert norm_amount("`5,00,000") == pytest.approx(500000.0)

    def test_crore_form(self):
        assert norm_amount("1.5 crore") == pytest.approx(15_000_000.0)

    def test_indian_comma_format(self):
        """5,00,000 comma-separated Indian number."""
        assert norm_amount("5,00,000") == pytest.approx(500000.0)

    def test_10_lakh(self):
        assert norm_amount("10 lakh") == pytest.approx(1_000_000.0)

    def test_2_crores(self):
        assert norm_amount("2 crores") == pytest.approx(20_000_000.0)

    # ---------- existing forms still work ----------
    def test_plain_inr_no_symbol(self):
        assert norm_amount("500000") == pytest.approx(500000.0)

    def test_rs_lakhs_plural(self):
        assert norm_amount("Rs. 10 lakhs") == pytest.approx(1_000_000.0)


class TestCleanPDFCurrency:
    def test_backtick_replaced(self):
        """_clean_pdf_currency converts backtick to rupee symbol (₹)."""
        result = _clean_pdf_currency("`5,00,000")
        # Backtick is the PDF corruption of ₹; it gets converted to ₹
        assert result == "\u20b95,00,000", f"Got {result!r}"

    def test_trailing_slash_removed(self):
        assert _clean_pdf_currency("5,00,000/-") == "5,00,000"


class TestBPS:
    """
    Tests for basis-points normalization (25 bps = 0.25%).
    """

    def test_bps_numeric(self):
        assert norm_bps("25 bps") == pytest.approx(0.25)

    def test_bps_basis_points(self):
        assert norm_bps("25 basis points") == pytest.approx(0.25)

    def test_bps_100(self):
        assert norm_bps("100 bps") == pytest.approx(1.0)

    def test_bps_decimal(self):
        assert norm_bps("12.5 bps") == pytest.approx(0.125)

    def test_bps_singular(self):
        assert norm_bps("1 basis point") == pytest.approx(0.01)

    def test_no_bps(self):
        assert norm_bps("10% interest") is None

    def test_is_bps_true(self):
        assert is_bps("rate cut of 25 bps") is True

    def test_is_bps_false(self):
        assert is_bps("rate cut of 5%") is False


# ===========================================================================
# Issue 18 - Date parser extensions
# ===========================================================================

class TestDateExtensions:
    """
    Tests for Issue 18: AY, dd.mm.yyyy, ordinal dates.
    """

    def test_ay_2024_25(self):
        """AY 2024-25 -> 01 Apr 2024."""
        d = norm_date("AY 2024-25")
        assert d == date(2024, 4, 1), f"Got {d}"

    def test_ay_2023_24(self):
        """AY 2023-24 -> 01 Apr 2023."""
        d = norm_date("AY 2023-24")
        assert d == date(2023, 4, 1)

    def test_fy_from_ay_2024_25(self):
        """AY 2024-25 -> FY 2023-24."""
        result = fy_from_ay("AY 2024-25")
        assert result == (2023, 2024)

    def test_dot_separator_date(self):
        """'12.03.2024' -> 12 March 2024."""
        d = norm_date("12.03.2024")
        assert d == date(2024, 3, 12), f"Got {d}"

    def test_ordinal_day_of_month(self):
        """'1st day of April, 2026' -> 1 April 2026."""
        d = norm_date("1st day of April, 2026")
        assert d == date(2026, 4, 1), f"Got {d}"

    def test_2nd_day_of_january(self):
        """'2nd day of January, 2025' -> 2 Jan 2025."""
        d = norm_date("2nd day of January, 2025")
        assert d == date(2025, 1, 2)

    def test_fy_still_works(self):
        """Existing FY form must still parse."""
        d = norm_date("FY 2023-24")
        assert d == date(2023, 4, 1)

    def test_iso_still_works(self):
        d = norm_date("2024-03-12")
        assert d == date(2024, 3, 12)

    def test_word_date_still_works(self):
        d = norm_date("12 March 2024")
        assert d == date(2024, 3, 12)


# ===========================================================================
# Section identifier normalizer (Bug 2 - norm_section)
# ===========================================================================

class TestNormSectionBug2:
    def test_section_194_ia(self):
        """Bug 2: '194-IA' must not be truncated to '194'."""
        result = norm_section("section 194-IA")
        assert "194-ia" in result, f"Got {result!r}"

    def test_section_80c(self):
        assert "80c" in norm_section("Section 80C")

    def test_section_80c_2(self):
        assert "80c(2)" in norm_section("Section 80C(2)")

    def test_us_80c(self):
        assert "80c" in norm_section("u/s 80C")

    def test_regulation_52_4(self):
        assert "52(4)" in norm_section("Regulation 52(4)")
