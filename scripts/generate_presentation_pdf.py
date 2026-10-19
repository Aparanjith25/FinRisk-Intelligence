"""Script to generate high-resolution docs/presentation.pdf (7 slides) using Pillow.

Reflects the exact implementation of FinRisk-Intelligence:
Slide 1: Title & Candidate / Team Identification Placeholders
Slide 2: Problem Statement & Technical Approach
Slide 3: System Architecture & Data Flow
Slide 4: Implementation & Technology Choices
Slide 5: Experimental Results & Validated Workflow
Slide 6: Domain Impact & Module B Stress-Testing Example
Slide 7: Limitations, Modeling Approximations & Future Work
"""

import os
from PIL import Image, ImageDraw, ImageFont

def get_font(name: str, size: int):
    windir = os.environ.get("WINDIR", "C:\\Windows")
    font_path = os.path.join(windir, "Fonts", name)
    if os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()

def draw_header(draw, title: str, slide_num: int, total_slides: int = 7):
    # Top banner
    draw.rectangle([(0, 0), (1920, 110)], fill=(15, 23, 42)) # Slate 900
    f_title = get_font("segoeuib.ttf", 36)
    f_num = get_font("segoeui.ttf", 20)
    draw.text((60, 32), title, font=f_title, fill=(255, 255, 255))
    draw.text((1760, 42), f"{slide_num} / {total_slides}", font=f_num, fill=(148, 163, 184))

    # Bottom footer
    draw.rectangle([(0, 1030), (1920, 1080)], fill=(241, 245, 249))
    f_footer = get_font("segoeui.ttf", 15)
    draw.text((60, 1045), "FinRisk Intelligence | S&P Global & CRISIL Campus Hackathon 2026 | DoSelect Phase III", font=f_footer, fill=(100, 116, 139))
    draw.text((1540, 1045), "Confidential — Hackathon Submission", font=f_footer, fill=(100, 116, 139))

def create_slide_1() -> Image.Image:
    """Slide 1: Title and candidate/institution placeholders."""
    img = Image.new("RGB", (1920, 1080), (15, 23, 42)) # Deep Navy Background
    draw = ImageDraw.Draw(img)

    f_super = get_font("segoeui.ttf", 24)
    f_title = get_font("segoeuib.ttf", 60)
    f_sub = get_font("segoeui.ttf", 28)
    f_card_hdr = get_font("segoeuib.ttf", 22)
    f_card_txt = get_font("segoeui.ttf", 18)

    draw.text((120, 160), "S&P GLOBAL & CRISIL CAMPUS HACKATHON 2026 | DOSELECT PHASE III", font=f_super, fill=(56, 189, 248))
    draw.text((120, 210), "FinRisk Intelligence", font=f_title, fill=(255, 255, 255))
    draw.text((120, 290), "AI-Powered Financial Risk Engine & Strategic Multi-Asset Portfolio Stress Testing", font=f_sub, fill=(203, 213, 225))

    # Decorative line
    draw.rectangle([(120, 360), (450, 366)], fill=(56, 189, 248))

    # Project Scope pill
    draw.rounded_rectangle([(120, 400), (620, 445)], radius=8, fill=(30, 41, 59), outline=(56, 189, 248), width=1)
    draw.text((140, 412), "Selected Downstream Track: Module B — Portfolio Stress Testing", font=get_font("segoeuib.ttf", 16), fill=(224, 242, 254))

    # Cards for Team & Project Information
    # Left Card: Candidate Placeholders
    draw.rounded_rectangle([(120, 480), (980, 880)], radius=12, fill=(30, 41, 59), outline=(71, 85, 105), width=2)
    draw.text((150, 510), "Candidate & Team Identification", font=f_card_hdr, fill=(56, 189, 248))

    info_lines = [
        ("Candidate Name:", "Ramisetty Aparanjith"),
        ("College / University:", "IIT (ISM) Dhanbad"),
        ("Official Email:", "23je0787@iitism.ac.in"),
        ("Degree & Department:", "B.Tech / Computer Science and Engineering"),
        ("Public GitHub Repository:", "https://github.com/Aparanjith25/FinRisk-Intelligence"),
        ("Unlisted Demo Video:", "https://youtu.be/hfqHQLqod7w"),
    ]

    y_pos = 560
    for label, val in info_lines:
        draw.text((150, y_pos), label, font=get_font("segoeuib.ttf", 17), fill=(226, 232, 240))
        draw.text((430, y_pos), val, font=get_font("consola.ttf", 16), fill=(147, 197, 253))
        y_pos += 44

    # Right Card: Project Overview & Core Metrics
    draw.rounded_rectangle([(1020, 480), (1800, 880)], radius=12, fill=(30, 41, 59), outline=(71, 85, 105), width=2)
    draw.text((1050, 510), "Technical Submission Highlights", font=f_card_hdr, fill=(52, 211, 153))

    tech_points = [
        "• End-to-End Operational Pipeline: Unstructured narrative ingestion to multi-asset valuation.",
        "• Genuine Dual-Source Ingestion: Live regulatory news (SEC, ECB, Fed) + synthetic social CSV.",
        "• Financial NLP Risk Scoring: ProsusAI/finbert continuous sentiment [-1, 1] + 6 event categories.",
        "• Explainable Impact Scoring: Transparent 1-10 integer score with additive audit rationales.",
        "• High-Impact Event Alerts: Configurable threshold (> 7) with human-confirmed scenario staging.",
        "• Module B Wholesale Portfolio: 10 assets ($17.0M baseline) including Pay-Fixed IRS with DV01.",
        "• Engineering Rigor: 147 passed tests (0 failed, 0 skipped), FastAPI REST + Streamlit UI.",
    ]

    y_pos = 565
    for pt in tech_points:
        draw.text((1050, y_pos), pt, font=f_card_txt, fill=(226, 232, 240))
        y_pos += 42

    # Bottom Footer
    draw.rectangle([(0, 1030), (1920, 1080)], fill=(15, 23, 42))
    draw.text((120, 1045), "S&P Global & CRISIL Campus Hackathon 2026 | Slide 1 of 7", font=get_font("segoeui.ttf", 16), fill=(100, 116, 139))
    return img

def create_slide_2() -> Image.Image:
    """Slide 2: Problem and approach."""
    img = Image.new("RGB", (1920, 1080), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw_header(draw, "Problem Statement & Technical Approach", 2)

    f_card_hdr = get_font("segoeuib.ttf", 24)
    f_sub = get_font("segoeuib.ttf", 18)
    f_body = get_font("segoeui.ttf", 17)
    f_code = get_font("consola.ttf", 15)

    # Left Column: The Problem (Challenges in Financial Risk)
    draw.rounded_rectangle([(80, 150), (920, 990)], radius=12, fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.rounded_rectangle([(80, 150), (920, 210)], radius=12, fill=(254, 242, 242), outline=(254, 202, 202), width=1)
    draw.text((110, 168), "Financial Risk Industry Challenges", font=f_card_hdr, fill=(185, 28, 28))

    problems = [
        ("Deluge of Unstructured Text:", "Risk analysts must continuously monitor central bank speeches, SEC filings, and fast-moving market narratives. Manual tracking is slow and error-prone."),
        ("Opaque Black-Box NLP:", "Off-the-shelf generative models hallucinate entities, confuse publication dates, and provide unexplainable predictions unacceptable in regulated banking."),
        ("Lagging Risk Indicators:", "Traditional VaR models rely on historical covariance matrices that lag during sudden macroeconomic regime shifts and geopolitical conflicts."),
        ("Disconnection Between News & Portfolios:", "Existing systems either generate news sentiment without portfolio links or run stress tests without narrative context. No automated bridge exists."),
        ("Uncontrolled Execution Risk:", "Automated execution of algorithmic trades or shock parameters without human validation introduces systemic risk and regulatory non-compliance."),
    ]

    y_pos = 235
    for title, desc in problems:
        draw.text((110, y_pos), title, font=f_sub, fill=(153, 27, 27))
        y_pos += 26
        draw.text((110, y_pos), desc, font=f_body, fill=(71, 85, 105))
        y_pos += 68

    # Right Column: The FinRisk Intelligence Approach
    draw.rounded_rectangle([(980, 150), (1840, 990)], radius=12, fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.rounded_rectangle([(980, 150), (1840, 210)], radius=12, fill=(240, 253, 244), outline=(187, 247, 208), width=1)
    draw.text((1010, 168), "FinRisk Intelligence Approach & Principles", font=f_card_hdr, fill=(22, 101, 52))

    solutions = [
        ("Multi-Stream Ingestion & Normalization:", "Ingests live RSS feeds (Fed, SEC, ECB) alongside synthetic social streams into a canonical 7-field schema with deterministic SHA-256 fingerprinting."),
        ("Continuous Sentiment in [-1.0, +1.0]:", "Computes P(pos) - P(neg) via ProsusAI/finbert with graceful offline deterministic baseline fallback and strict metadata attribution."),
        ("Explainable Rule-Based Impact Scoring:", "Combines event categorization (6 types) with an additive 1-10 impact scoring heuristic. Every score emits a human-readable mathematical audit string."),
        ("Human-in-the-Loop Decision Support:", "High-impact events (> 7) trigger alerts that stage recommended macro scenarios. ZERO automatic executions; the portfolio manager confirms all tests."),
        ("Module B Quantitative Portfolio Valuation:", "Simulates multi-factor shocks on a $17.0M wholesale portfolio (Equities, Corp/Govt Bonds, Loans, and an Interest Rate Swap with DV01 pricing)."),
    ]

    y_pos = 235
    for title, desc in solutions:
        draw.text((1010, y_pos), title, font=f_sub, fill=(21, 128, 61))
        y_pos += 26
        draw.text((1010, y_pos), desc, font=f_body, fill=(51, 65, 85))
        y_pos += 68

    return img

def create_slide_3() -> Image.Image:
    """Slide 3: Architecture."""
    img = Image.new("RGB", (1920, 1080), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw_header(draw, "System Architecture & End-to-End Pipeline", 3)

    # Embed scaled architecture diagram if available
    arch_png = "c:\\FinRisk-Intelligence\\docs\\architecture.png"
    if os.path.exists(arch_png):
        try:
            arch_img = Image.open(arch_png)
            # Resize maintaining aspect ratio to fit inside (1760, 820)
            arch_img.thumbnail((1760, 830), Image.Resampling.LANCZOS)
            x_offset = (1920 - arch_img.width) // 2
            img.paste(arch_img, (x_offset, 140))
        except Exception:
            draw.text((100, 200), "Architecture Diagram at docs/architecture.png", font=get_font("segoeui.ttf", 24), fill=(0, 0, 0))
    else:
        draw.text((100, 200), "Architecture diagram file docs/architecture.png not found", font=get_font("segoeui.ttf", 24), fill=(255, 0, 0))

    # Caption Box below
    draw.rounded_rectangle([(80, 980), (1840, 1020)], radius=6, fill=(241, 245, 249), outline=(203, 213, 225), width=1)
    draw.text((100, 992), "Strict Architectural Separation: Real regulatory news (Fed, SEC, ECB) is isolated from synthetic test fixtures. All outputs flow to FastAPI and Streamlit.", font=get_font("segoeui.ttf", 15), fill=(71, 85, 105))
    return img

def create_slide_4() -> Image.Image:
    """Slide 4: Implementation and technology choices."""
    img = Image.new("RGB", (1920, 1080), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw_header(draw, "Implementation Details & Technology Stack", 4)

    f_card_hdr = get_font("segoeuib.ttf", 22)
    f_sub = get_font("segoeuib.ttf", 16)
    f_body = get_font("segoeui.ttf", 15)

    # 4 Cards Layout
    cards = [
        ("1. Ingestion & Normalization Layer", (80, 150, 920, 540), (239, 246, 255), (59, 130, 246), (30, 64, 175), [
            ("Dual Stream Parsers:", "XML/RSS parser for news + CSV tabular parser for social feeds."),
            ("Public Syndication Feeds:", "Direct live fetching from SEC and ECB RSS feeds (10s bounded timeout)."),
            ("SHA-256 Fingerprinting:", "Deterministic record IDs eliminate duplicate stories across multiple runs."),
            ("Zero Hallucination:", "Missing company names or publication dates are preserved as None, never fabricated."),
            ("Provenance Isolation:", "Real public records and synthetic records are stored in isolated file paths."),
        ]),
        ("2. NLP & Explainable Risk Scoring", (980, 150, 1840, 540), (245, 243, 255), (139, 92, 246), (91, 33, 182), [
            ("Sentiment Engine:", "ProsusAI/finbert yielding continuous P(pos) - P(neg) in [-1.0, +1.0]."),
            ("Offline Fallback:", "Deterministic domain lexicon fallback when torch/transformers are omitted."),
            ("Event Categorization:", "6 categories: Geopolitical, Macroeconomic, Credit, Regulatory, M&A, Product."),
            ("Impact Scoring Heuristic:", "Additive integer score (1-10) factoring base severity, sentiment, and metrics."),
            ("Audit Trail:", "Every risk signal generates a transparent, human-readable rationale explanation."),
        ]),
        ("3. Stress Testing Valuation (Module B)", (80, 570, 920, 970), (240, 253, 244), (34, 197, 94), (20, 83, 45), [
            ("Multi-Asset Coverage:", "Equities, Corporate Bonds, Sovereign Bonds, Loans, and Interest Rate Swaps."),
            ("Derivative Support:", "5Y Pay-Fixed Macro Hedge Swap ($10M notional, DV01 = $4,200/bp, D=4.2)."),
            ("Directional Consistency:", "Rate hikes produce swap gains; rate cuts produce swap losses."),
            ("Loss Attribution:", "Asset-level contributions sum exactly to 100.0% of total portfolio loss."),
            ("Standardized Scenarios:", "4 preconfigured macro shocks (Stagflation, Geopolitical, Credit, Recession)."),
        ]),
        ("4. API, Dashboard & Testing Stack", (980, 570, 1840, 970), (254, 242, 242), (239, 68, 68), (153, 27, 27), [
            ("FastAPI Microservice:", "High-performance async REST backend with OpenAPI Swagger docs (/docs)."),
            ("Streamlit Analytics UI:", "Interactive 3-tab dashboard with real-time feed switching and Plotly charts."),
            ("Robust Fallback Pattern:", "Dashboard operates seamlessly with live API or direct local engine fallback."),
            ("Automated Test Suite:", "147 unit, integration, and UI AppTests passing with 100% pass rate in ~41s."),
            ("Python Stack:", "Built on Python 3.13, Pydantic, Pandas, NumPy, HTTPX, and Pytest."),
        ]),
    ]

    for title, bbox, fill, outline, hdr_fill, items in cards:
        draw.rounded_rectangle(bbox, radius=12, fill=fill, outline=outline, width=2)
        draw.text((bbox[0] + 25, bbox[1] + 20), title, font=f_card_hdr, fill=hdr_fill)
        y = bbox[1] + 65
        for item_title, item_desc in items:
            draw.text((bbox[0] + 25, y), item_title, font=f_sub, fill=hdr_fill)
            y += 22
            draw.text((bbox[0] + 25, y), item_desc, font=f_body, fill=(51, 65, 85))
            y += 42

    return img

def create_slide_5() -> Image.Image:
    """Slide 5: Results and tested workflow."""
    img = Image.new("RGB", (1920, 1080), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw_header(draw, "Empirical Results & Validated Workflow", 5)

    f_card_hdr = get_font("segoeuib.ttf", 22)
    f_sub = get_font("segoeuib.ttf", 16)
    f_body = get_font("segoeui.ttf", 15)
    f_code = get_font("consola.ttf", 14)

    # Top Row: 3 Key Metrics Cards
    metrics = [
        ("Automated Tests Passing", "147 / 147", "100% Pass Rate (0 failed, 0 skipped)", (240, 253, 244), (34, 197, 94), (21, 128, 61)),
        ("Real Public News Ingested", "63 Records", "26 SEC, 20 Fed, 17 ECB Press Releases", (239, 246, 255), (59, 130, 246), (30, 64, 175)),
        ("Portfolio Market Value", "$17,000,000", "10 Assets Across 5 Distinct Classes", (254, 243, 199), (245, 158, 11), (180, 83, 9)),
    ]

    for i, (m_title, m_val, m_sub, bg, border, txt_c) in enumerate(metrics):
        bx1 = 80 + i * 600
        bx2 = bx1 + 560
        draw.rounded_rectangle([(bx1, 150), (bx2, 280)], radius=10, fill=bg, outline=border, width=2)
        draw.text((bx1 + 25, 168), m_title, font=f_sub, fill=txt_c)
        draw.text((bx1 + 25, 200), m_val, font=get_font("segoeuib.ttf", 36), fill=txt_c)
        draw.text((bx1 + 25, 250), m_sub, font=get_font("segoeui.ttf", 14), fill=(71, 85, 105))

    # Bottom Left: End-to-End Execution Trace on Real Public Data
    draw.rounded_rectangle([(80, 310), (1020, 980)], radius=12, fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.text((110, 335), "Validated End-to-End Pipeline Trace (Real Data)", font=f_card_hdr, fill=(15, 23, 42))

    trace_steps = [
        ("Step 1: Ingestion", "63 genuine regulatory records loaded from data/processed/public_news_records.csv"),
        ("Step 2: Normalization", "100% verified with SHA-256 IDs, provenance='real_public_news'"),
        ("Step 3: Sentiment Scoring", "ProsusAI/finbert continuous scores range: [-0.9208, +0.9131] (strictly within [-1, 1])"),
        ("Step 4: Event Classification", "Macroeconomic (26), Regulatory (18), Geopolitical (1), Corporate (18)"),
        ("Step 5: Impact Scoring", "Impact scores range: [1, 7] strictly within [1, 10] with additive audit rationales"),
        ("Step 6: High-Impact Alerting", "At threshold=6: 4 alerts detected and staged; at threshold=7: 0 alerts (boundary honored)"),
        ("Step 7: Scenario Mapping", "Signals recommended 'Macroeconomic Stagflation' (Composite score: 0.48, 12 triggers)"),
        ("Step 8: Portfolio Stress Test", "$17.0M portfolio stressed to $15,960,740.00 (-$1,039,260.00 / -6.11% total loss)"),
        ("Step 9: Delivery", "Results rendered in Streamlit UI & FastAPI (/stress-test) with holding-level waterfall"),
    ]

    y = 385
    for st_title, st_desc in trace_steps:
        draw.text((110, y), st_title, font=get_font("segoeuib.ttf", 15), fill=(30, 41, 59))
        draw.text((290, y), st_desc, font=f_body, fill=(71, 85, 105))
        y += 62

    # Bottom Right: NLP Benchmark Comparison Results
    draw.rounded_rectangle([(1060, 310), (1840, 980)], radius=12, fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.text((1090, 335), "Empirical NLP Sentiment Evaluation", font=f_card_hdr, fill=(15, 23, 42))

    draw.text((1090, 385), "Preliminary Curated Regulatory Benchmark (N = 30):", font=f_sub, fill=(51, 65, 85))
    draw.text((1090, 415), "• ProsusAI/finbert: 76.67% Accuracy | Macro F1: 0.7548", font=f_body, fill=(21, 128, 61))
    draw.text((1090, 445), "• Deterministic Baseline: 70.00% Accuracy | Macro F1: 0.6849", font=f_body, fill=(71, 85, 105))
    draw.text((1090, 475), "• Held-out ECB subset (n=8): Baseline 87.5% vs FinBERT 75.0% (due to hedged central-bank neutral wording)", font=f_body, fill=(100, 116, 139))

    draw.line([(1090, 525), (1810, 525)], fill=(226, 232, 240), width=1)

    draw.text((1090, 550), "Financial PhraseBank Evaluation (N = 339 test split):", font=f_sub, fill=(51, 65, 85))
    draw.text((1090, 580), "• ProsusAI/finbert: 97.35% Accuracy | Macro F1: 0.9628", font=f_body, fill=(21, 128, 61))
    draw.text((1090, 610), "• Fine-tuned candidate on corporate news: 98.53% Accuracy | Macro F1: 0.9746", font=f_body, fill=(71, 85, 105))

    draw.line([(1090, 660), (1810, 660)], fill=(226, 232, 240), width=1)

    draw.text((1090, 685), "Advanced NLP Experimentation & Routing Finding:", font=f_sub, fill=(185, 28, 28))
    draw.text((1090, 715), "• Fine-tuning on corporate news caused catastrophic forgetting\n  on legal/regulatory text (regulatory negative recall dropped 70% → 60%).\n• Production model strictly remains original ProsusAI/finbert.\n• Built isolated source-aware routing prototype in experiments/;\n  recommends more regulatory training data before production deployment.", font=f_body, fill=(71, 85, 105))

    return img

def create_slide_6() -> Image.Image:
    """Slide 6: Domain impact and Module B stress-testing example."""
    img = Image.new("RGB", (1920, 1080), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw_header(draw, "Domain Impact & Module B Stress-Testing Case Study", 6)

    f_card_hdr = get_font("segoeuib.ttf", 22)
    f_sub = get_font("segoeuib.ttf", 16)
    f_body = get_font("segoeui.ttf", 15)
    f_code = get_font("consola.ttf", 14)

    # Top Narrative Context
    draw.rounded_rectangle([(80, 140), (1840, 225)], radius=8, fill=(239, 246, 255), outline=(191, 219, 254), width=1)
    draw.text((105, 155), "Case Study: Ingested FOMC & Energy Shocks Triggering 'Macroeconomic Stagflation' Shock", font=f_card_hdr, fill=(30, 64, 175))
    draw.text((105, 190), "Scenario Parameters: Equity -15% | Yield Curve +150 bps | Credit Spreads +100 bps | Commercial Loan Loss +20%", font=f_body, fill=(71, 85, 105))

    # Table of 10 Holdings Stressed Valuation
    draw.rounded_rectangle([(80, 245), (1840, 780)], radius=12, fill=(255, 255, 255), outline=(226, 232, 240), width=2)

    # Table Header
    headers = [("ID", 100), ("Asset Name", 180), ("Asset Class", 500), ("Base Value", 700), ("Stressed Value", 890), ("Loss / Gain ($)", 1100), ("P&L %", 1300), ("Contribution", 1470), ("Valuation Method", 1640)]
    draw.rectangle([(80, 245), (1840, 295)], fill=(241, 245, 249))
    for h_txt, h_x in headers:
        draw.text((h_x, 260), h_txt, font=get_font("segoeuib.ttf", 15), fill=(51, 65, 85))

    rows = [
        ("EQ_001", "AlphaCorp Common Stock", "Equity", "$2,500,000", "$2,125,000", "-$375,000.00", "-15.00%", "36.08%", "Direct Price Shock (-15%)"),
        ("EQ_002", "BetaEnergy Global Shares", "Equity", "$1,800,000", "$1,530,000", "-$270,000.00", "-15.00%", "25.98%", "Direct Price Shock (-15%)"),
        ("EQ_003", "GammaTech Solutions", "Equity", "$1,200,000", "$1,020,000", "-$180,000.00", "-15.00%", "17.32%", "Direct Price Shock (-15%)"),
        ("CB_001", "GammaBank Senior Notes", "Corp Bond", "$2,000,000", "$1,760,000", "-$240,000.00", "-12.00%", "23.09%", "Duration (D=4.8, dy+ds=250bp)"),
        ("CB_002", "DeltaMfg Credit Debentures", "Corp Bond", "$1,000,000", "$920,000", "-$80,000.00", "-8.00%", "7.70%", "Duration (D=3.2, dy+ds=250bp)"),
        ("GB_001", "US Treasury Benchmark 10Y", "Govt Bond", "$3,000,000", "$2,631,000", "-$369,000.00", "-12.30%", "35.51%", "Duration (D=8.2, dy=150bp)"),
        ("GB_002", "German Bund Sovereign 5Y", "Govt Bond", "$2,000,000", "$1,865,000", "-$135,000.00", "-6.75%", "12.99%", "Duration (D=4.5, dy=150bp)"),
        ("LN_001", "CRE Syndicated Senior Loan", "Loan", "$1,500,000", "$1,471,650", "-$28,350.00", "-1.89%", "2.73%", "Credit Migration (PD*LGD*1.20)"),
        ("LN_002", "SME Secured Term Facility", "Loan", "$1,000,000", "$964,000", "-$36,000.00", "-3.60%", "3.46%", "Credit Migration (PD*LGD*1.20)"),
        ("SW_001", "5Y Pay-Fixed IRS Macro Hedge", "Derivative", "$1,000,000", "$1,630,000", "+$630,000.00", "+63.00%", "-60.62%", "Linear DV01 ($4,200/bp * +150bp)"),
    ]

    y = 310
    for r in rows:
        fill_color = (255, 255, 255) if r[0] != "SW_001" else (240, 253, 244)
        if r[0] == "SW_001":
            draw.rectangle([(80, y - 5), (1840, y + 36)], fill=fill_color)
        draw.text((100, y), r[0], font=f_code, fill=(30, 41, 59))
        draw.text((180, y), r[1], font=f_body, fill=(30, 41, 59))
        draw.text((500, y), r[2], font=f_body, fill=(71, 85, 105))
        draw.text((700, y), r[3], font=f_body, fill=(30, 41, 59))
        draw.text((890, y), r[4], font=f_body, fill=(30, 41, 59))
        loss_fill = (185, 28, 28) if "-" in r[5] else (21, 128, 61)
        draw.text((1100, y), r[5], font=get_font("segoeuib.ttf", 15), fill=loss_fill)
        draw.text((1300, y), r[6], font=f_body, fill=loss_fill)
        draw.text((1470, y), r[7], font=f_body, fill=(71, 85, 105))
        draw.text((1640, y), r[8], font=f_body, fill=(71, 85, 105))
        y += 44

    # Portfolio Total Summary Box
    draw.rounded_rectangle([(80, 805), (1840, 980)], radius=10, fill=(15, 23, 42))
    draw.text((110, 830), "CONSOLIDATED PORTFOLIO STRESS RESULTS:", font=f_card_hdr, fill=(56, 189, 248))
    draw.text((110, 880), "Baseline Market Value: $17,000,000.00  →  Stressed Portfolio Value: $15,960,740.00", font=get_font("segoeuib.ttf", 22), fill=(255, 255, 255))
    draw.text((110, 925), "Total Absolute Loss: -$1,039,260.00 (-6.11%)  |  Total Holdings: 10  |  Loss Attribution Check: Sum of Asset Losses = $1,039,260.00 (Exact Match)", font=f_body, fill=(203, 213, 225))

    # Derivative Highlight Note on right
    draw.text((1200, 830), "DERIVATIVE MACRO HEDGE IMPACT:", font=get_font("segoeuib.ttf", 18), fill=(52, 211, 153))
    draw.text((1200, 865), "The Pay-Fixed IRS gained +$630,000 (+63.0%),\nactively offsetting 37.7% of physical bond losses\nand cushioning overall portfolio drawdown.", font=f_body, fill=(224, 242, 254))

    return img

def create_slide_7() -> Image.Image:
    """Slide 7: Limitations and future work."""
    img = Image.new("RGB", (1920, 1080), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw_header(draw, "Model Limitations, Honest AI Principles & Future Roadmap", 7)

    f_card_hdr = get_font("segoeuib.ttf", 22)
    f_sub = get_font("segoeuib.ttf", 16)
    f_body = get_font("segoeui.ttf", 15)

    # Left Column: Documented Limitations & Assumptions
    draw.rounded_rectangle([(80, 150), (920, 980)], radius=12, fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.rounded_rectangle([(80, 150), (920, 210)], radius=12, fill=(254, 243, 199), outline=(253, 230, 138), width=1)
    draw.text((110, 168), "Explicit Modeling Limitations (Non-Exaggerated)", font=f_card_hdr, fill=(180, 83, 9))

    limits = [
        ("Stylized Sensitivity vs. Econometric Forecasting:", "Portfolio stress testing executes deterministic sensitivity shocks. It does NOT predict future market prices, default probabilities, or calibrated Value at Risk (VaR)."),
        ("Qualitative Decision Support vs. Quantitative Calibration:", "News risk signals recommend stylized macro scenarios for human review; they do NOT automatically calibrate quantitative basis-point rate shocks."),
        ("Linear DV01 & Duration Approximations:", "Fixed-income and swap revaluation utilizes first-order linear duration / DV01 without second-order convexity adjustments or yield curve reshaping."),
        ("Social Media Stream is Synthetic:", "Ingested social media posts are local synthetic fixtures ([SYNTHETIC POST]), not a live Twitter/Reddit scraping stream (avoids enterprise API key paywalls)."),
        ("RSS Summary vs. Full Text Scraping:", "News ingestion parses official RSS <title> and <description> fields (~30-80 words) rather than scraping full HTML articles behind publisher paywalls."),
    ]

    y = 235
    for l_title, l_desc in limits:
        draw.text((110, y), l_title, font=f_sub, fill=(146, 64, 14))
        y += 24
        draw.text((110, y), l_desc, font=f_body, fill=(71, 85, 105))
        y += 66

    # Right Column: Engineering Best Practices & Future Roadmap
    draw.rounded_rectangle([(980, 150), (1840, 980)], radius=12, fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.rounded_rectangle([(980, 150), (1840, 210)], radius=12, fill=(240, 253, 244), outline=(187, 247, 208), width=1)
    draw.text((1010, 168), "Engineering Rigor & Post-Hackathon Roadmap", font=f_card_hdr, fill=(22, 101, 52))

    roadmap = [
        ("Source-Aware Sentiment Routing Prototype:", "Deploy the experimentally validated routing prototype (using original FinBERT for regulatory text and fine-tuned models for corporate filings) after gathering larger legal corpora."),
        ("Full Yield Curve Term Structure Repricing:", "Integrate QuantLib or open-source curve bootstrapping to price non-parallel curve twists, steepening, and second-order swap convexity."),
        ("Live Enterprise Social & News Webhooks:", "Connect official enterprise APIs (Bloomberg, FactSet, X v2 API) with rate-limit queuing and audit logging."),
        ("Counterparty Credit Valuation Adjustments (CVA/DVA):", "Incorporate bilateral counterparty exposure profiling into derivative valuation."),
        ("Automated CI/CD Regression Pipeline:", "Maintain the strict 147-test automated verification suite across all future PRs and model updates."),
    ]

    y = 235
    for r_title, r_desc in roadmap:
        draw.text((1010, y), r_title, font=f_sub, fill=(21, 128, 61))
        y += 24
        draw.text((1010, y), r_desc, font=f_body, fill=(51, 65, 85))
        y += 66

    return img

def generate_presentation_pdf(output_path: str):
    slide_generators = [
        create_slide_1,
        create_slide_2,
        create_slide_3,
        create_slide_4,
        create_slide_5,
        create_slide_6,
        create_slide_7,
    ]

    images = []
    for i, gen_func in enumerate(slide_generators):
        print(f"Rendering Slide {i+1} of 7...")
        images.append(gen_func())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(
        output_path,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=images[1:],
    )
    print(f"Presentation PDF successfully created at: {output_path} (Total 7 slides)")

if __name__ == "__main__":
    generate_presentation_pdf("c:\\FinRisk-Intelligence\\docs\\presentation.pdf")
