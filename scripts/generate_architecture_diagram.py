"""Script to generate high-resolution docs/architecture.png using Pillow.

Reflects the exact implementation of FinRisk-Intelligence:
- Multi-Source Ingestion: Real Public News (Fed, SEC, ECB) vs. Synthetic Social CSV
- Normalization & Deduplication: SHA-256 fingerprinting, unified schema, provenance tags
- NLP & Risk Engine: ProsusAI/finbert [-1, 1], Rule-based Event Classifier (6 types), Impact Scorer [1, 10]
- Alerting & Decision Support: High-impact alerts (> threshold), ScenarioMapper
- Human-in-the-Loop Confirmation
- Module B Strategic Stress Testing: ScenarioEngine, 10 holdings ($17.0M), DV01 swap pricing
- Delivery Layer: FastAPI REST Service & Streamlit Interactive Analytics Dashboard
"""

import os
from PIL import Image, ImageDraw, ImageFont

def get_font(name: str, size: int):
    windir = os.environ.get("WINDIR", "C:\\Windows")
    font_path = os.path.join(windir, "Fonts", name)
    if os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()

def draw_rounded_rect(draw, bbox, radius, fill, outline, width=1):
    draw.rounded_rectangle(bbox, radius=radius, fill=fill, outline=outline, width=width)

def draw_arrow(draw, start, end, color=(80, 90, 110), width=3):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if x1 == x2: # vertical
        if y2 > y1: # downward
            draw.polygon([(x2 - 6, y2 - 10), (x2 + 6, y2 - 10), (x2, y2)], fill=color)
        else:
            draw.polygon([(x2 - 6, y2 + 10), (x2 + 6, y2 + 10), (x2, y2)], fill=color)
    elif y1 == y2: # horizontal
        if x2 > x1: # rightward
            draw.polygon([(x2 - 10, y2 - 6), (x2 - 10, y2 + 6), (x2, y2)], fill=color)
        else:
            draw.polygon([(x2 + 10, y2 - 6), (x2 + 10, y2 + 6), (x2, y2)], fill=color)

def generate_architecture_diagram(output_path: str):
    # Canvas dimensions: 2400 x 1450 (Ultra crisp 2K layout)
    w, h = 2400, 1450
    img = Image.new("RGB", (w, h), (248, 250, 252)) # Light Slate 50
    draw = ImageDraw.Draw(img)

    # Fonts
    f_title = get_font("segoeui.ttf", 46)
    f_subtitle = get_font("segoeui.ttf", 22)
    f_col_hdr = get_font("segoeuib.ttf", 24)
    f_card_hdr = get_font("segoeuib.ttf", 20)
    f_card_sub = get_font("segoeuib.ttf", 16)
    f_card_txt = get_font("segoeui.ttf", 15)
    f_badge = get_font("segoeuib.ttf", 13)
    f_legend = get_font("segoeui.ttf", 16)

    # Header Bar
    draw.rectangle([(0, 0), (w, 110)], fill=(15, 23, 42)) # Slate 900
    draw.text((60, 22), "FinRisk Intelligence — System Architecture", font=f_title, fill=(255, 255, 255))
    draw.text((60, 75), "End-to-End Unstructured Market Narrative Ingestion to Strategic Multi-Asset Portfolio Stress Testing", font=f_subtitle, fill=(148, 163, 184))

    # Top Legend: Distinguishing Real vs Synthetic vs Engine
    legend_y = 125
    # Real Public Data badge
    draw.rounded_rectangle([(60, legend_y), (260, legend_y + 32)], radius=6, fill=(219, 234, 254), outline=(59, 130, 246), width=2)
    draw.text((75, legend_y + 7), "● REAL PUBLIC DATA", font=f_badge, fill=(30, 64, 175))

    # Synthetic Fixture badge
    draw.rounded_rectangle([(280, legend_y), (480, legend_y + 32)], radius=6, fill=(254, 243, 199), outline=(245, 158, 11), width=2)
    draw.text((295, legend_y + 7), "▲ SYNTHETIC FIXTURE", font=f_badge, fill=(180, 83, 9))

    # Automated Engine badge
    draw.rounded_rectangle([(500, legend_y), (700, legend_y + 32)], radius=6, fill=(241, 245, 249), outline=(100, 116, 139), width=2)
    draw.text((515, legend_y + 7), "■ PIPELINE ENGINE", font=f_badge, fill=(51, 65, 85))

    # Human Governance badge
    draw.rounded_rectangle([(720, legend_y), (950, legend_y + 32)], radius=6, fill=(254, 226, 226), outline=(239, 68, 68), width=2)
    draw.text((735, legend_y + 7), "◆ HUMAN-IN-THE-LOOP", font=f_badge, fill=(153, 27, 27))

    # Define 6 Columns
    # Col 1: Data Ingestion (x=60..400)
    # Col 2: Normalization (x=450..770)
    # Col 3: NLP & Risk Engine (x=820..1190)
    # Col 4: Alerting & Mapping (x=1240..1580)
    # Col 5: Portfolio Stress Engine (x=1630..1990)
    # Col 6: Delivery Layer (x=2040..2340)

    columns = [
        ("1. Multi-Source Ingestion", 60, 390),
        ("2. Normalization & Isolation", 440, 760),
        ("3. NLP & Risk Engine", 810, 1180),
        ("4. Alerting & Recommendation", 1230, 1580),
        ("5. Portfolio Stress Engine", 1630, 1980),
        ("6. API & Dashboard Delivery", 2030, 2340),
    ]

    for title, x1, x2 in columns:
        draw.rounded_rectangle([(x1, 175), (x2, 1380)], radius=12, fill=(255, 255, 255), outline=(226, 232, 240), width=2)
        # Column title banner
        draw.rounded_rectangle([(x1, 175), (x2, 230)], radius=12, fill=(241, 245, 249), outline=(226, 232, 240), width=1)
        draw.text((x1 + 18, 190), title, font=f_col_hdr, fill=(30, 41, 59))

    # -------------------------------------------------------------
    # COLUMN 1: Ingestion Sources
    # -------------------------------------------------------------
    # Box 1.1: Federal Reserve RSS (Real)
    draw_rounded_rect(draw, (80, 260, 370, 410), 8, (239, 246, 255), (59, 130, 246), 2)
    draw.text((95, 275), "U.S. Federal Reserve", font=f_card_hdr, fill=(30, 64, 175))
    draw.text((95, 305), "● REAL PUBLIC REGULATORY NEWS", font=f_badge, fill=(37, 99, 235))
    draw.text((95, 335), "• Endpoint: /feeds/press_all.xml\n• Monetary policy & FOMC actions\n• Parsed: <title> + <description>", font=f_card_txt, fill=(71, 85, 105))

    # Box 1.2: SEC Press Releases (Real)
    draw_rounded_rect(draw, (80, 440, 370, 590), 8, (239, 246, 255), (59, 130, 246), 2)
    draw.text((95, 455), "U.S. SEC Press Releases", font=f_card_hdr, fill=(30, 64, 175))
    draw.text((95, 485), "● REAL PUBLIC REGULATORY NEWS", font=f_badge, fill=(37, 99, 235))
    draw.text((95, 515), "• Endpoint: /news/pressreleases.rss\n• Regulatory enforcement filings\n• Live HTTP fetching + timeout guards", font=f_card_txt, fill=(71, 85, 105))

    # Box 1.3: European Central Bank (Real)
    draw_rounded_rect(draw, (80, 620, 370, 770), 8, (239, 246, 255), (59, 130, 246), 2)
    draw.text((95, 635), "European Central Bank", font=f_card_hdr, fill=(30, 64, 175))
    draw.text((95, 665), "● REAL PUBLIC REGULATORY NEWS", font=f_badge, fill=(37, 99, 235))
    draw.text((95, 695), "• Endpoint: /rss/press.html\n• Eurozone monetary decisions\n• 17 articles active in public store", font=f_card_txt, fill=(71, 85, 105))

    # Box 1.4: Synthetic Social Stream (CSV)
    draw_rounded_rect(draw, (80, 810, 370, 970), 8, (254, 252, 232), (245, 158, 11), 2)
    draw.text((95, 825), "Financial Social Media", font=f_card_hdr, fill=(180, 83, 9))
    draw.text((95, 855), "▲ SYNTHETIC TEST FIXTURE", font=f_badge, fill=(217, 119, 6))
    draw.text((95, 885), "• Source: sample_social_posts.csv\n• Prefix: [SYNTHETIC POST]\n• Local CSV parser (zero live scraping)\n• Tested multi-source pipeline flow", font=f_card_txt, fill=(120, 53, 15))

    # Box 1.5: Synthetic News RSS XML
    draw_rounded_rect(draw, (80, 1010, 370, 1170), 8, (254, 252, 232), (245, 158, 11), 2)
    draw.text((95, 1025), "Sample News RSS XML", font=f_card_hdr, fill=(180, 83, 9))
    draw.text((95, 1055), "▲ SYNTHETIC TEST FIXTURE", font=f_badge, fill=(217, 119, 6))
    draw.text((95, 1085), "• Source: sample_news_rss.xml\n• Deterministic test cases\n• Offline baseline validation", font=f_card_txt, fill=(120, 53, 15))

    # -------------------------------------------------------------
    # COLUMN 2: Normalization & Isolation
    # -------------------------------------------------------------
    draw_rounded_rect(draw, (460, 260, 740, 480), 8, (248, 250, 252), (148, 163, 184), 2)
    draw.text((475, 275), "Normalizer & Deduplicator", font=f_card_hdr, fill=(15, 23, 42))
    draw.text((475, 305), "■ finrisk/ingestion/normalizer.py", font=f_badge, fill=(100, 116, 139))
    draw.text((475, 335), "• SHA-256 fingerprinting (record_id)\n• Strict 7-field standard schema:\n  [id, text, source, type, published, company, url]\n• Safe date parsing (ISO-8601)\n• Duplicate suppression by content hash\n• Zero fabrication of missing fields", font=f_card_txt, fill=(51, 65, 85))

    # Dual File Storage Isolation
    draw_rounded_rect(draw, (460, 530, 740, 730), 8, (239, 246, 255), (59, 130, 246), 2)
    draw.text((475, 545), "Public News Data Store", font=f_card_hdr, fill=(30, 64, 175))
    draw.text((475, 575), "● REAL PUBLIC STORE (ISOLATED)", font=f_badge, fill=(37, 99, 235))
    draw.text((475, 605), "• Path: data/processed/public_news_records.csv\n• Contains 63 verified regulatory records\n• Provenance: 'real_public_news'\n• Never mixed with synthetic fixtures", font=f_card_txt, fill=(71, 85, 105))

    draw_rounded_rect(draw, (460, 780, 740, 980), 8, (254, 252, 232), (245, 158, 11), 2)
    draw.text((475, 795), "Synthetic Data Store", font=f_card_hdr, fill=(180, 83, 9))
    draw.text((475, 825), "▲ SYNTHETIC STORE (ISOLATED)", font=f_badge, fill=(217, 119, 6))
    draw.text((475, 855), "• Path: data/processed/ingested_records.csv\n• Controlled test cases\n• Provenance: 'synthetic'\n• Offline test reproducibility", font=f_card_txt, fill=(120, 53, 15))

    draw_rounded_rect(draw, (460, 1030, 740, 1260), 8, (248, 250, 252), (148, 163, 184), 2)
    draw.text((475, 1045), "Provenance Enforcer", font=f_card_hdr, fill=(15, 23, 42))
    draw.text((475, 1075), "■ Multi-Stream Integrity Guard", font=f_badge, fill=(100, 116, 139))
    draw.text((475, 1105), "• Guarantees synthetic data is never\n  substituted into public news mode\n• Empty public store yields informational\n  notice rather than silent mock insertion", font=f_card_txt, fill=(51, 65, 85))

    # -------------------------------------------------------------
    # COLUMN 3: NLP & Risk Engine
    # -------------------------------------------------------------
    draw_rounded_rect(draw, (830, 260, 1160, 540), 8, (248, 250, 252), (79, 70, 229), 2)
    draw.text((845, 275), "Financial Sentiment Analysis", font=f_card_hdr, fill=(67, 56, 202))
    draw.text((845, 305), "■ FinBERTSentimentAnalyzer", font=f_badge, fill=(99, 102, 241))
    draw.text((845, 335), "• Primary: ProsusAI/finbert neural inference\n• Continuous score: P(pos) - P(neg)\n• Bounded strictly to [-1.0, +1.0]\n• Output classes: Positive, Neutral, Negative\n• Graceful Offline Fallback:\n  Deterministic financial lexicon baseline\n• Metadata tagging:\n  'pretrained_transformers' vs\n  'offline_fallback_baseline'", font=f_card_txt, fill=(51, 65, 85))

    draw_rounded_rect(draw, (830, 580, 1160, 840), 8, (248, 250, 252), (148, 163, 184), 2)
    draw.text((845, 595), "Event Classification", font=f_card_hdr, fill=(15, 23, 42))
    draw.text((845, 625), "■ RuleBasedEventClassifier", font=f_badge, fill=(100, 116, 139))
    draw.text((845, 655), "• 6 Financial & Macro Categories:\n  1. Geopolitical (conflict, tariffs, war)\n  2. Macroeconomic (inflation, rates, GDP)\n  3. Credit Event (default, downgrade)\n  4. Regulatory Action (sanctions, fines)\n  5. Corporate Transaction (M&A)\n  6. Product / Operational (launches, tech)\n• Explainable keyword regex matching", font=f_card_txt, fill=(51, 65, 85))

    draw_rounded_rect(draw, (830, 880, 1160, 1160), 8, (248, 250, 252), (148, 163, 184), 2)
    draw.text((845, 895), "Explainable Impact Scorer", font=f_card_hdr, fill=(15, 23, 42))
    draw.text((845, 925), "■ finrisk/risk_engine/impact_scorer.py", font=f_badge, fill=(100, 116, 139))
    draw.text((845, 955), "• Integer score: 1 to 10 scale\n• Additive heuristic formula:\n  Base Category Severity (2-5)\n  + Sentiment Penalty / Boost (up to +3)\n  + Company Entity Presence (+1)\n  + Quantitative Figure Presence (+1)\n• Bounded strictly: min=1, max=10\n• Full human-readable audit rationale", font=f_card_txt, fill=(51, 65, 85))

    # Risk signals output box
    draw_rounded_rect(draw, (830, 1200, 1160, 1340), 8, (239, 246, 255), (59, 130, 246), 2)
    draw.text((845, 1215), "Structured Risk Signals Store", font=f_card_hdr, fill=(30, 64, 175))
    draw.text((845, 1245), "• data/processed/public_risk_signals.csv\n• Output schema with scores, rationales & provenance", font=f_card_txt, fill=(71, 85, 105))

    # -------------------------------------------------------------
    # COLUMN 4: Alerting & Recommendation
    # -------------------------------------------------------------
    draw_rounded_rect(draw, (1250, 260, 1560, 520), 8, (254, 242, 242), (239, 68, 68), 2)
    draw.text((1265, 275), "High-Impact Event Alerts", font=f_card_hdr, fill=(185, 28, 28))
    draw.text((1265, 305), "■ finrisk/risk_engine/alerting.py", font=f_badge, fill=(220, 38, 38))
    draw.text((1265, 335), "• Configurable Alert Rule:\n  impact_score > threshold (default > 7)\n• Strictly honors boundary (score 7 ≠ alert)\n• Deduplicates recurring news stories\n• Extracts severity, category & context\n• Stages alert for manager review", font=f_card_txt, fill=(127, 29, 29))

    draw_rounded_rect(draw, (1250, 560, 1560, 840), 8, (248, 250, 252), (148, 163, 184), 2)
    draw.text((1265, 575), "Decision Support Mapper", font=f_card_hdr, fill=(15, 23, 42))
    draw.text((1265, 605), "■ finrisk/stress_testing/scenario_mapper.py", font=f_badge, fill=(100, 116, 139))
    draw.text((1265, 635), "• Recommends Stylized Macro Scenario:\n  - Geopolitical → Geopolitical Escalation\n  - Rates / Stagflation → Macro Stagflation\n  - Credit Event → Severe Credit Crunch\n  - Rate Cuts → Mild Recession\n• Microeconomic / M&A filtered out\n  (No spurious macro triggers)", font=f_card_txt, fill=(51, 65, 85))

    # Human-in-the-loop Gate
    draw_rounded_rect(draw, (1250, 890, 1560, 1140), 8, (254, 226, 226), (220, 38, 38), 3)
    draw.text((1265, 910), "Human-in-the-Loop Gate", font=f_card_hdr, fill=(153, 27, 27))
    draw.text((1265, 940), "◆ MANDATORY GOVERNANCE SAFEGUARD", font=f_badge, fill=(185, 28, 28))
    draw.text((1265, 975), "• ZERO AUTOMATIC EXECUTION\n• News events provide qualitative context\n• Portfolio manager explicitly reviews\n  contributing evidence and stages scenario\n• Explicit button execution required\n  to run valuation stress tests", font=f_card_txt, fill=(127, 29, 29))

    # -------------------------------------------------------------
    # COLUMN 5: Portfolio Stress Testing (Module B)
    # -------------------------------------------------------------
    draw_rounded_rect(draw, (1650, 260, 1960, 510), 8, (248, 250, 252), (148, 163, 184), 2)
    draw.text((1665, 275), "Wholesale Banking Portfolio", font=f_card_hdr, fill=(15, 23, 42))
    draw.text((1665, 305), "■ 10 Holdings | $17,000,000 Baseline", font=f_badge, fill=(100, 116, 139))
    draw.text((1665, 335), "• 3 Equities ($5.5M total)\n• 2 Corporate Bonds ($3.0M, D=4.8, 3.2)\n• 2 Sovereign Bonds ($5.0M, D=8.2, 4.5)\n• 2 Commercial Loans ($2.5M, PD/LGD)\n• 1 Pay-Fixed Interest Rate Swap\n  ($1.0M base value, $10M notional,\n   DV01 = $4,200/bp, D=4.2)", font=f_card_txt, fill=(51, 65, 85))

    draw_rounded_rect(draw, (1650, 550, 1960, 930), 8, (248, 250, 252), (13, 148, 136), 2)
    draw.text((1665, 565), "ScenarioEngine (Valuation)", font=f_card_hdr, fill=(15, 118, 110))
    draw.text((1665, 595), "■ Deterministic Factor Pricing", font=f_badge, fill=(20, 184, 166))
    draw.text((1665, 625), "• Equities: Direct price shock\n  V_stressed = V_base * (1 + shock_eq)\n\n• Sovereign Bonds: Duration rate shock\n  V_stressed = V_base * (1 - D * dy)\n\n• Corp Bonds: Rates + spread widening\n  V_stressed = V_base * (1 - D * (dy + ds))\n\n• Loans: Credit loss multiplier\n  V_stressed = V_base * (1 - PD*LGD*(1+s))\n\n• Pay-Fixed Swap: Linear DV01 pricing\n  V_stressed = V_base + (DV01_$ * dy_bps)\n  (Rate hike → Gain / Rate cut → Loss)", font=f_card_txt, fill=(19, 78, 74))

    draw_rounded_rect(draw, (1650, 970, 1960, 1260), 8, (248, 250, 252), (148, 163, 184), 2)
    draw.text((1665, 985), "Portfolio Consolidation", font=f_card_hdr, fill=(15, 23, 42))
    draw.text((1665, 1015), "■ Exact Loss Attribution", font=f_badge, fill=(100, 116, 139))
    draw.text((1665, 1045), "• Total Stressed Portfolio Value\n• Total Absolute Loss ($)\n• Total Percentage Loss (%)\n• Asset-Level Loss Contributions (%)\n  Sum of contributions ≡ 100.0%\n• Verified hand calculations match\n  engine output to 2 decimal places", font=f_card_txt, fill=(51, 65, 85))

    # -------------------------------------------------------------
    # COLUMN 6: Delivery Layer
    # -------------------------------------------------------------
    draw_rounded_rect(draw, (2050, 260, 2320, 750), 8, (248, 250, 252), (16, 185, 129), 2)
    draw.text((2065, 275), "FastAPI Microservice", font=f_card_hdr, fill=(4, 120, 87))
    draw.text((2065, 305), "■ http://127.0.0.1:8000", font=f_badge, fill=(5, 150, 105))
    draw.text((2065, 335), "• High-performance REST API\n• Interactive OpenAPI Docs (/docs)\n• Endpoints:\n  - GET /health\n  - GET /portfolio\n  - GET /scenarios\n  - POST /stress-test\n  - GET /risk-signals\n  - GET /scenarios/suggested\n  - POST /public-news/fetch\n• Strict Pydantic contracts", font=f_card_txt, fill=(6, 95, 70))

    draw_rounded_rect(draw, (2050, 800, 2320, 1340), 8, (248, 250, 252), (244, 63, 94), 2)
    draw.text((2065, 815), "Streamlit Visual UI", font=f_card_hdr, fill=(190, 18, 60))
    draw.text((2065, 845), "■ http://localhost:8501", font=f_badge, fill=(225, 29, 72))
    draw.text((2065, 875), "• Tab 1: NLP Risk Signals\n  - Data stream mode toggle\n    (Synthetic / Public / All)\n  - Sentiment class breakdown\n  - Event category charts\n  - High-impact alert banner\n    with 'Stage for Test' button\n\n• Tab 2: Stress Testing\n  - News scenario recommendation\n  - Macro shock selector\n  - Stressed valuation metrics\n  - P&L in Red (Loss) / Green (Gain)\n  - Asset loss contribution chart\n\n• Tab 3: Methodology & Limits\n  - Honest disclosure caveats", font=f_card_txt, fill=(136, 19, 55))

    # -------------------------------------------------------------
    # FLOW ARROWS CONNECTING COLUMNS
    # -------------------------------------------------------------
    # Col 1 -> Col 2
    draw_arrow(draw, (370, 335), (460, 335))
    draw_arrow(draw, (370, 515), (460, 360))
    draw_arrow(draw, (370, 695), (460, 385))
    draw_arrow(draw, (370, 890), (460, 880))
    draw_arrow(draw, (370, 1090), (460, 910))

    # Col 2 -> Col 3
    draw_arrow(draw, (740, 630), (830, 400))
    draw_arrow(draw, (740, 880), (830, 430))
    draw_arrow(draw, (740, 650), (830, 710))
    draw_arrow(draw, (740, 670), (830, 1020))

    # Col 3 -> Col 4
    draw_arrow(draw, (1160, 1020), (1250, 390))
    draw_arrow(draw, (1160, 710), (1250, 700))
    draw_arrow(draw, (1405, 520), (1405, 560))
    draw_arrow(draw, (1405, 840), (1405, 890))

    # Col 4 -> Col 5 (Human Confirmation Gate to ScenarioEngine)
    draw_arrow(draw, (1560, 1015), (1650, 740), color=(220, 38, 38), width=4)

    # Col 5 (Portfolio -> ScenarioEngine)
    draw_arrow(draw, (1805, 510), (1805, 550))
    draw_arrow(draw, (1805, 930), (1805, 970))

    # Col 5 -> Col 6 (Delivery)
    draw_arrow(draw, (1960, 740), (2050, 500))
    draw_arrow(draw, (1960, 1115), (2050, 1070))

    # Save PNG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG", quality=95)
    print(f"Architecture diagram generated successfully at: {output_path}")

if __name__ == "__main__":
    generate_architecture_diagram("c:\\FinRisk-Intelligence\\docs\\architecture.png")
