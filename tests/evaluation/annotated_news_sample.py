"""Reproducible human-annotated evaluation sample of 30 real public financial news articles.

Sourced directly from official regulatory feeds:
- U.S. Federal Reserve Press Releases
- U.S. Securities & Exchange Commission (SEC) Press Releases
- European Central Bank (ECB) Press Releases

ANNOTATION GUIDELINES:
1. Positive ('positive'):
   - Explicit regulatory approvals for corporate expansion, bank mergers, or new capital formation.
   - Significant reduction of regulatory burden or expansion of market access/retailization.
   - Resilient economic growth, positive IPO market proceeds, or successful remediation.
2. Negative ('negative'):
   - Enforcement actions, fraud charges, civil monetary penalties, or censures.
   - Failures in anti-money laundering (AML), cherry-picking schemes, or Ponzi schemes.
   - Macroeconomic stagflation, overlapping adverse shocks, or systemic risk alerts.
3. Neutral ('neutral'):
   - Routine administrative notices, public comment period extensions, meeting agendas, and advisory committee events.
   - Survey releases (e.g. Survey of Consumer Finances, SESFOD survey) presenting factual statistics without editorial bias.
   - Speeches, interviews, and academic panel discussions examining general central banking concepts (digital euro, AI in finance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class AnnotatedArticle:
    record_id: str
    source: str
    headline: str
    text: str
    gold_sentiment: str
    rationale: str
    is_held_out: bool = False


ANNOTATED_NEWS_SAMPLES: List[AnnotatedArticle] = [
    # --- Federal Reserve (10 Articles) ---
    AnnotatedArticle(
        record_id="ace881c871481cb4",
        source="U.S. Federal Reserve Press Releases",
        headline="Federal Reserve Board releases results of the 2025 Survey of Consumer Finances",
        text="Federal Reserve Board releases results of the 2025 Survey of Consumer Finances, which provides the public and policymakers with detailed insights into the economic condition of American families.",
        gold_sentiment="neutral",
        rationale="Routine release of factual economic survey data without policy shock or directional distress.",
    ),
    AnnotatedArticle(
        record_id="4fd0d0c529d7605e",
        source="U.S. Federal Reserve Press Releases",
        headline="Federal Reserve Board announces enforcement action against American Express Company",
        text="Federal Reserve Board announces enforcement action against American Express Company to address, among other things, the firm’s failure to sufficiently detect and report certain suspicious activity related to money laundering.",
        gold_sentiment="negative",
        rationale="Formal regulatory enforcement action for anti-money laundering compliance failure.",
    ),
    AnnotatedArticle(
        record_id="f7f5881477a775a8",
        source="U.S. Federal Reserve Press Releases",
        headline="Minutes of the Federal Open Market Committee, September 15-16, 2026",
        text="Minutes of the Federal Open Market Committee, September 15-16, 2026.",
        gold_sentiment="neutral",
        rationale="Standard scheduled publication of FOMC monetary policy meeting minutes.",
    ),
    AnnotatedArticle(
        record_id="12ec8c84be1ffb9a",
        source="U.S. Federal Reserve Press Releases",
        headline="Federal Reserve Board announces approval of application by Isabella Bank Corporation",
        text="Federal Reserve Board announces approval of application by Isabella Bank Corporation.",
        gold_sentiment="positive",
        rationale="Regulatory approval granted for corporate banking application.",
    ),
    AnnotatedArticle(
        record_id="263dfa34442e5e9d",
        source="U.S. Federal Reserve Press Releases",
        headline="Federal Reserve Board announces approval of application by Fleur Capital Corporation",
        text="Federal Reserve Board announces approval of application by Fleur Capital Corporation.",
        gold_sentiment="positive",
        rationale="Regulatory approval granted for financial holding company transaction.",
    ),
    AnnotatedArticle(
        record_id="6e7f3b3e775836cf",
        source="U.S. Federal Reserve Press Releases",
        headline="Federal Reserve Board announces it will extend comment period on Regulation O",
        text="Federal Reserve Board announces it will extend, until November 4, the comment period on its proposal to modernize Regulation O.",
        gold_sentiment="neutral",
        rationale="Administrative calendar extension for public regulatory comment.",
    ),
    AnnotatedArticle(
        record_id="7a9893c532ed55d6",
        source="U.S. Federal Reserve Press Releases",
        headline="Federal Reserve Board issues enforcement action with Ontario Bancorporation, Inc.",
        text="Federal Reserve Board issues enforcement action with Ontario Bancorporation, Inc..",
        gold_sentiment="negative",
        rationale="Regulatory enforcement action against banking institution.",
    ),
    AnnotatedArticle(
        record_id="d0b9b80d30783d7f",
        source="U.S. Federal Reserve Press Releases",
        headline="Federal Reserve Board finalizes changes to enhance transparency of stress test",
        text="Federal Reserve Board finalizes changes to enhance the transparency and public accountability of its stress test and reduce volatility in its stress test-related capital requirements.",
        gold_sentiment="positive",
        rationale="Policy reform reducing regulatory volatility and improving capital transparency.",
    ),
    AnnotatedArticle(
        record_id="7305130b011550db",
        source="U.S. Federal Reserve Press Releases",
        headline="Federal Reserve Board issues enforcement action with former employee of Sandy Spring Bank",
        text="Federal Reserve Board issues enforcement action with former employee of Sandy Spring Bank.",
        gold_sentiment="negative",
        rationale="Enforcement sanction against individual employee for regulatory violations.",
    ),
    AnnotatedArticle(
        record_id="17fcf757da8d5f8d",
        source="U.S. Federal Reserve Press Releases",
        headline="Agencies reduce regulatory burden for community banks, increase exam cycle eligibility",
        text="Agencies reduce regulatory burden for community banks, increase eligibility for 18-month exam cycle.",
        gold_sentiment="positive",
        rationale="Explicit reduction of regulatory burden and operational cost relief for banks.",
    ),

    # --- U.S. Securities & Exchange Commission (12 Articles) ---
    AnnotatedArticle(
        record_id="781e3bfdee9ce7ab",
        source="U.S. SEC Press Releases",
        headline="SEC Proposes Expanding Securities Eligible for Cross Trading by Registered Funds",
        text="The Securities and Exchange Commission today proposed amendments to the Investment Company Act cross-trading rule, which permits transactions in securities between a registered fund and its affiliates under certain conditions.",
        gold_sentiment="positive",
        rationale="Regulatory proposal facilitating market flexibility and portfolio trading efficiency.",
    ),
    AnnotatedArticle(
        record_id="8df65cbcb621d1fa",
        source="U.S. SEC Press Releases",
        headline="SEC Seeks Final Judgment Against Former Western Asset Co-CIO Ken Leech in Cherry Picking Case",
        text="The Securities and Exchange Commission today moved for entry of a final judgment by consent against Stephen Kenneth Leech II, the former co-chief investment officer of registered investment adviser Western Asset Management Company LLC, whom the SEC charged with fraudulent cherry-picking.",
        gold_sentiment="negative",
        rationale="Final judgment in major investment fraud and fiduciary breach action.",
    ),
    AnnotatedArticle(
        record_id="43eb6e4ec420aaa9",
        source="U.S. SEC Press Releases",
        headline="SEC to Host Virtual National Compliance Outreach Seminar",
        text="The Securities and Exchange Commission’s Compliance Outreach Program announced today that it will host a virtual national seminar for investment companies and investment advisers on November 19, 2026.",
        gold_sentiment="neutral",
        rationale="Routine informational and educational seminar announcement.",
    ),
    AnnotatedArticle(
        record_id="fa651a694de8fed4",
        source="U.S. SEC Press Releases",
        headline="SEC Coordinates with Global Financial Regulators to Raise Fraud Awareness During World Investor Week",
        text="The Securities and Exchange Commission, in coordination with other U.S. and global financial regulators, today issued a joint investor bulletin as part of World Investor Week (WIW).",
        gold_sentiment="neutral",
        rationale="Educational investor protection bulletin without specific corporate infractions.",
    ),
    AnnotatedArticle(
        record_id="fc02dcd5edef5a63",
        source="U.S. SEC Press Releases",
        headline="SEC’s Division of Examinations Announces New Exam Handbook",
        text="The Securities and Exchange Commission’s Division of Examinations today published its new handbook, The SEC Exam Handbook: A Practical Guide on Process and Engagement.",
        gold_sentiment="neutral",
        rationale="Publication of practical procedural guidance for market participants.",
    ),
    AnnotatedArticle(
        record_id="19f7f41bcb229126",
        source="U.S. SEC Press Releases",
        headline="SEC Charges Meyer Global Management and Its CEO With Defrauding Retail Investors",
        text="The Securities and Exchange Commission today charged private fund adviser Meyer Global Management LLC (MGM) and its CEO, Owen E.H. Meyer, with defrauding investors and MGM-managed funds in connection with investments in SpaceX and other pre-IPO securities.",
        gold_sentiment="negative",
        rationale="Severe securities fraud charges involving misappropriation and retail investor deception.",
    ),
    AnnotatedArticle(
        record_id="772caffdf119af12",
        source="U.S. SEC Press Releases",
        headline="SEC Charges Two Individuals With Orchestrating Fraud Scheme That Targeted Veterans",
        text="The Securities and Exchange Commission today announced charges against Christopher Kenji Dinelli and Jacob David Frankel for allegedly orchestrating a fraud scheme that raised more than $8.7 million from 35 investors.",
        gold_sentiment="negative",
        rationale="Criminal-style multi-million dollar affinity fraud scheme charges.",
    ),
    AnnotatedArticle(
        record_id="0bbde02cadcfd919",
        source="U.S. SEC Press Releases",
        headline="SEC Proposes Amendments to Expand Responsible Retailization of Private Markets",
        text="The Securities and Exchange Commission today voted to propose rule amendments that would facilitate capital formation in the public and private markets by expanding retail investor choice and promoting innovation in regulated fund structures.",
        gold_sentiment="positive",
        rationale="Facilitation of capital formation and broader investor choice.",
    ),
    AnnotatedArticle(
        record_id="8d0f2868e7fbaddc",
        source="U.S. SEC Press Releases",
        headline="SEC Charges Multiple Entities in Fraud Schemes Totaling at Least $15 Million",
        text="The Securities and Exchange Commission today charged multiple entities that are likely operated by individuals located overseas for defrauding hundreds of retail investors through so-called investment confidence scams.",
        gold_sentiment="negative",
        rationale="Large-scale $15M international fraudulent scam enforcement.",
    ),
    AnnotatedArticle(
        record_id="09f4643c754363d0",
        source="U.S. SEC Press Releases",
        headline="SEC Charges Registered Investment Adviser Zoe Financial for Failure to Disclose Conflict of Interest",
        text="The Securities and Exchange Commission today announced settled charges against New York-based investment adviser Zoe Financial Inc. for failing to fully and fairly disclose material facts concerning conflicts of interest to its clients.",
        gold_sentiment="negative",
        rationale="Enforcement action and fine for undisclosed fiduciary conflicts of interest.",
    ),
    AnnotatedArticle(
        record_id="8cd0eca35538c395",
        source="U.S. SEC Press Releases",
        headline="SEC Publishes Updated Market Statistics, Highlighting Increase in IPOs and Proceeds Raised",
        text="The Securities and Exchange Commission’s Division of Economic and Risk Analysis (DERA) published updated statistics covering key segments of the U.S. capital markets, including a notable increase in the number of initial public offerings and proceeds raised.",
        gold_sentiment="positive",
        rationale="Favorable market indicators showing capital market growth and IPO expansion.",
    ),
    AnnotatedArticle(
        record_id="7a1a9166226a9597",
        source="U.S. SEC Press Releases",
        headline="SEC Censures OTC Link LLC for Repeated Compliance Failures Related to Regulation SCI",
        text="The Securities and Exchange Commission today censured New York-based broker dealer OTC Link LLC and ordered it to pay a $575,000 civil penalty for longstanding violations of Regulation Systems Compliance and Integrity.",
        gold_sentiment="negative",
        rationale="Regulatory censure and $575,000 financial penalty for systems compliance failures.",
    ),

    # --- European Central Bank (8 Articles) ---
    AnnotatedArticle(
        record_id="1bb8a99fb7925352",
        source="European Central Bank Press Releases",
        headline="Meeting of 9-10 September 2026",
        text="Meeting of 9-10 September 2026: Summary of economic deliberations and policy discussions.",
        gold_sentiment="neutral",
        rationale="Factual account of scheduled central bank monetary policy meeting.",
        is_held_out=True,
    ),
    AnnotatedArticle(
        record_id="15d2f2ca58000b5e",
        source="European Central Bank Press Releases",
        headline="Piero Cipollone: Interview with Corriere della Sera",
        text="Piero Cipollone discusses central bank strategy, payment systems, and European financial integration in interview with Corriere della Sera.",
        gold_sentiment="neutral",
        rationale="Executive interview discussing general economic policy perspectives.",
        is_held_out=True,
    ),
    AnnotatedArticle(
        record_id="79e2622b9328c00e",
        source="European Central Bank Press Releases",
        headline="Results of the September 2026 survey on credit terms and conditions (SESFOD)",
        text="Results of the September 2026 survey on credit terms and conditions in euro-denominated securities financing and OTC derivatives markets (SESFOD).",
        gold_sentiment="neutral",
        rationale="Statistical survey report on OTC market credit terms.",
        is_held_out=True,
    ),
    AnnotatedArticle(
        record_id="49f7e3abd974b48e",
        source="European Central Bank Press Releases",
        headline="Frank Elderson: Effective supervision through timely remediation",
        text="Frank Elderson remarks on banking supervision, governance standards, and proactive compliance remediation.",
        gold_sentiment="neutral",
        rationale="Keynote speech on supervisory principles and institutional governance.",
        is_held_out=True,
    ),
    AnnotatedArticle(
        record_id="97a6f766baba6e7a",
        source="European Central Bank Press Releases",
        headline="Piero Cipollone: Money in the digital age: digital euro, tokenisation and the role of central banks",
        text="Speech on digital currency architecture, settlement systems, tokenisation, and commercial bank intermediation.",
        gold_sentiment="neutral",
        rationale="Conceptual policy speech on payment system innovation.",
        is_held_out=True,
    ),
    AnnotatedArticle(
        record_id="c04ee1265a1edfea",
        source="European Central Bank Press Releases",
        headline="Philip R. Lane: Diagnostic Challenges for ECB Monetary Policy",
        text="Chief Economist Philip R. Lane outlines analytical challenges in gauging macroeconomic transmission and policy lag effects.",
        gold_sentiment="neutral",
        rationale="Academic diagnostic speech on econometric modeling of policy transmission.",
        is_held_out=True,
    ),
    AnnotatedArticle(
        record_id="c39946a60a2bbe60",
        source="European Central Bank Press Releases",
        headline="Boris Vujčić: Resilience, integration and competitiveness: building the future of European banking",
        text="Governing council member highlights the balance-sheet resilience, integration, and capital competitiveness of European banking institutions.",
        gold_sentiment="positive",
        rationale="Positive economic assessment highlighting banking resilience and competitiveness.",
        is_held_out=True,
    ),
    AnnotatedArticle(
        record_id="a0eccf863318d60e",
        source="European Central Bank Press Releases",
        headline="Isabel Schnabel: Monetary policy in a world of overlapping shocks",
        text="Isabel Schnabel analyzes structural inflation risks, supply disruptions, and monetary headwinds from overlapping geopolitical and climate shocks.",
        gold_sentiment="negative",
        rationale="Economic risk analysis warning of persistent structural inflation headwinds and overlapping shocks.",
        is_held_out=True,
    ),
]
