"""Curated labeled evaluation dataset for financial event classification.

Includes ground-truth test cases across all six canonical categories:
- Geopolitical
- Macroeconomic
- Credit Event
- Merger/Acquisition
- Product Launch
- Other
Plus curated ambiguous boundary cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class EvalSample:
    text: str
    gold_category: str
    rationale: str
    is_ambiguous: bool = False


EVAL_SAMPLES: List[EvalSample] = [
    # --- 1. Geopolitical ---
    EvalSample(
        text="US Trade Representative enacted strict import tariffs on foreign aluminum and steel shipments.",
        gold_category="Geopolitical",
        rationale="Tariffs enacted on foreign trade goods.",
    ),
    EvalSample(
        text="European Union allies announced coordinated sanctions against sovereign export banks.",
        gold_category="Geopolitical",
        rationale="Coordinated international sanctions against sovereign institutions.",
    ),
    EvalSample(
        text="Escalating military conflict in the Middle East threatened crude oil shipping lanes through the strait.",
        gold_category="Geopolitical",
        rationale="Military conflict threatening international shipping.",
    ),
    EvalSample(
        text="Department of Commerce placed export controls on cutting-edge lithography equipment.",
        gold_category="Geopolitical",
        rationale="Export controls and bans on technological hardware.",
    ),
    EvalSample(
        text="Border disputes disrupted bilateral freight transport and international customs clearance.",
        gold_category="Geopolitical",
        rationale="Border conflict impacting bilateral international transit.",
    ),
    EvalSample(
        text="Regulators opened a regulatory inquiry and imposed environmental fines on multinational mining conglomerate.",
        gold_category="Geopolitical",
        rationale="Regulatory enforcement inquiry and sovereign environmental fines.",
    ),

    # --- 2. Macroeconomic ---
    EvalSample(
        text="Federal Reserve officials raised benchmark interest rates by 50 basis points to cool inflation.",
        gold_category="Macroeconomic",
        rationale="Central bank rate hike to combat CPI inflation.",
    ),
    EvalSample(
        text="Consumer price index inflation accelerated to 6.2% annualized, surprising financial markets.",
        gold_category="Macroeconomic",
        rationale="CPI inflation rate metrics.",
    ),
    EvalSample(
        text="Treasury yields spiked across the yield curve following the FOMC monetary policy announcement.",
        gold_category="Macroeconomic",
        rationale="Sovereign treasury yields and yield curve shift from monetary policy.",
    ),
    EvalSample(
        text="National GDP contracted by 0.8% in Q2, intensifying concerns over an impending recession.",
        gold_category="Macroeconomic",
        rationale="GDP contraction and macroeconomic recession indicators.",
    ),
    EvalSample(
        text="Unemployment rate climbed to 4.9% as labor market cooling broadened across industrial sectors.",
        gold_category="Macroeconomic",
        rationale="Unemployment rate and labor market cooling.",
    ),
    EvalSample(
        text="Global central banks signaled quantitative tightening to counter persistent stagflation.",
        gold_category="Macroeconomic",
        rationale="Quantitative tightening and stagflation monetary environment.",
    ),

    # --- 3. Credit Event ---
    EvalSample(
        text="S&P downgraded commercial property developer debt to junk status following bond coupon default.",
        gold_category="Credit Event",
        rationale="Rating downgrade and bond coupon default.",
    ),
    EvalSample(
        text="Holding corporation filed for Chapter 11 bankruptcy protection after debt restructuring negotiations collapsed.",
        gold_category="Credit Event",
        rationale="Chapter 11 bankruptcy filing and debt restructuring failure.",
    ),
    EvalSample(
        text="Commercial bank increased loan loss provisions to $450M amid rising non-performing loans.",
        gold_category="Credit Event",
        rationale="Loan loss provisions and non-performing loans surge.",
    ),
    EvalSample(
        text="Credit spread widening reached 200 basis points across speculative high-yield bonds.",
        gold_category="Credit Event",
        rationale="Credit spread widening on high yield debt instruments.",
    ),
    EvalSample(
        text="Telecom operator defaulted on credit default swaps following a severe debt covenant breach.",
        gold_category="Credit Event",
        rationale="Default and covenant breach on credit obligations.",
    ),
    EvalSample(
        text="Insolvent lender entered judicial liquidation after failing to meet minimum capital adequacy requirements.",
        gold_category="Credit Event",
        rationale="Insolvency and judicial liquidation of financial institution.",
    ),

    # --- 4. Merger/Acquisition ---
    EvalSample(
        text="Global pharmaceuticals giant entered a definitive merger agreement to acquire biotech developer for $8.5B.",
        gold_category="Merger/Acquisition",
        rationale="Definitive merger agreement and acquisition transaction.",
    ),
    EvalSample(
        text="Private equity firm submitted an all-cash buyout bid of $55 per share to take the retailer private.",
        gold_category="Merger/Acquisition",
        rationale="Private equity buyout bid to acquire company.",
    ),
    EvalSample(
        text="Defense contractor announced a strategic corporate spinoff of its commercial satellite division.",
        gold_category="Merger/Acquisition",
        rationale="Corporate spinoff of operating business division.",
    ),
    EvalSample(
        text="Board of directors rejected an unsolicited hostile takeover bid from international rival.",
        gold_category="Merger/Acquisition",
        rationale="Hostile takeover bid attempt.",
    ),
    EvalSample(
        text="Tech conglomerate completed the divestiture of its legacy semiconductor fabrication unit.",
        gold_category="Merger/Acquisition",
        rationale="Corporate asset divestiture.",
    ),
    EvalSample(
        text="Investment consortium agreed on terms for a multi-billion dollar strategic stake purchase.",
        gold_category="Merger/Acquisition",
        rationale="Strategic stake purchase equity transaction.",
    ),

    # --- 5. Product Launch ---
    EvalSample(
        text="Enterprise software vendor unveiled next-generation generative AI analytics platform.",
        gold_category="Product Launch",
        rationale="Unveiling new commercial platform.",
    ),
    EvalSample(
        text="Automaker rolls out flagship electric delivery truck line for commercial fleet operators.",
        gold_category="Product Launch",
        rationale="Commercial rollout of new vehicle models.",
    ),
    EvalSample(
        text="Biotech firm secured FDA approval for its breakthrough oncology therapeutics medication.",
        gold_category="Product Launch",
        rationale="Regulatory FDA approval for clinical pharmaceutical product.",
    ),
    EvalSample(
        text="Consumer electronics manufacturer introduces new flagship mobile devices with neural processors.",
        gold_category="Product Launch",
        rationale="Product introduction of new hardware devices.",
    ),
    EvalSample(
        text="Medical device maker announced promising Phase 3 clinical trials and commercial availability.",
        gold_category="Product Launch",
        rationale="Phase 3 clinical trials and commercial product availability.",
    ),
    EvalSample(
        text="Semiconductor manufacturer was granted key patents for next-generation 2nm architecture.",
        gold_category="Product Launch",
        rationale="Patents granted for proprietary hardware technology.",
    ),

    # --- 6. Other ---
    EvalSample(
        text="Corporation held its 45th annual general shareholder meeting at the convention center.",
        gold_category="Other",
        rationale="Routine corporate governance meeting without risk catalyst.",
    ),
    EvalSample(
        text="Cloud services provider announced planned datacenter cooling infrastructure maintenance.",
        gold_category="Other",
        rationale="Routine facilities maintenance.",
    ),
    EvalSample(
        text="The board appointed a new Chief Human Resources Officer to lead corporate talent strategy.",
        gold_category="Other",
        rationale="Internal executive personnel appointment.",
    ),
    EvalSample(
        text="Logistics firm signed a 10-year commercial lease extension on its regional fulfillment warehouse.",
        gold_category="Other",
        rationale="Routine commercial lease extension.",
    ),
    EvalSample(
        text="Executive leadership participated in a panel discussion at an industry trade conference.",
        gold_category="Other",
        rationale="Industry marketing and speaking event.",
    ),
    EvalSample(
        text="Enterprise software company published its annual corporate sustainability and community report.",
        gold_category="Other",
        rationale="Routine ESG and sustainability disclosure.",
    ),

    # --- 7. Ambiguous Boundary Cases ---
    EvalSample(
        text="Heavy import tariffs triggered consumer price inflation across retail supply chains.",
        gold_category="Geopolitical",
        rationale="Primary catalyst is import tariffs (competing: Macroeconomic inflation).",
        is_ambiguous=True,
    ),
    EvalSample(
        text="Federal Reserve rate hikes drove wave of commercial loan defaults and corporate bankruptcies.",
        gold_category="Credit Event",
        rationale="Direct financial failure consequence is defaults and bankruptcies (competing: Macroeconomic).",
        is_ambiguous=True,
    ),
    EvalSample(
        text="Distressed takeover agreement reached to salvage insolvent regional transport provider.",
        gold_category="Merger/Acquisition",
        rationale="Transaction is a takeover agreement (competing: Credit Event insolvency).",
        is_ambiguous=True,
    ),
    EvalSample(
        text="OPEC oil export quotas drove global inflation and elevated borrowing costs.",
        gold_category="Geopolitical",
        rationale="Sovereign cartel export restrictions catalyst (competing: Macroeconomic).",
        is_ambiguous=True,
    ),
    EvalSample(
        text="Company published annual sustainability report highlighting unveiling of zero-emission platform.",
        gold_category="Product Launch",
        rationale="Commercial platform unveiling feature (competing: Other sustainability report).",
        is_ambiguous=True,
    ),
    EvalSample(
        text="Widening credit spreads and sovereign bond yield curve inversion signaled impending recession.",
        gold_category="Macroeconomic",
        rationale="Yield curve inversion and recession signal (competing: Credit Event spreads).",
        is_ambiguous=True,
    ),
]
