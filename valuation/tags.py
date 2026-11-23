"""XBRL tag catalog.

Each metric lists candidate tags per taxonomy in priority order. For every
period the first candidate with a reported value is used; any candidate other
than the first is a *fallback* and is written to the data-quality log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

US = "us-gaap"
IFRS = "ifrs-full"
DEI = "dei"


@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str
    kind: str  # "duration" or "instant"
    unit: str  # "money", "per_share" or "shares"
    additive: bool  # quarters may be derived by YTD differencing
    candidates: dict[str, tuple[str, ...]]
    notes: dict[str, str] = field(default_factory=dict)


def _m(key, label, kind, unit, additive, us=(), ifrs=(), notes=None) -> MetricDef:
    return MetricDef(key, label, kind, unit, additive, {US: tuple(us), IFRS: tuple(ifrs)}, notes or {})


METRICS: dict[str, MetricDef] = {
    m.key: m
    for m in [
        # ----- income statement / cash flow (durations) -----
        _m(
            "revenue", "Revenue", "duration", "money", True,
            us=(
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
                "SalesRevenueGoodsNet",
                "SalesRevenueServicesNet",
                "RevenuesNetOfInterestExpense",
                "RegulatedAndUnregulatedOperatingRevenue",
                "OperatingLeaseLeaseIncome",
            ),
            ifrs=("Revenue", "RevenueFromContractsWithCustomers", "RevenueFromSaleOfGoods"),
            notes={"OperatingLeaseLeaseIncome": "lease income only (lessor with no total revenue tag)"},
        ),
        _m(
            "lease_income", "Operating lease income", "duration", "money", True,
            us=("OperatingLeaseLeaseIncome", "OperatingLeasesIncomeStatementLeaseRevenue"),
        ),
        _m(
            "cost_of_revenue", "Cost of revenue", "duration", "money", True,
            us=(
                "CostOfRevenue",
                "CostOfGoodsAndServicesSold",
                "CostOfGoodsSold",
                "CostOfServices",
                # Further cost-of-sales lines; each is screened per company as a possible component (see COMPONENT_SCREENED_COST_TAGS).
                "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
                "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization",
                "CostOfServicesExcludingDepreciationDepletionAndAmortization",
                "CostOfPurchasedOilAndGas",
            ),
            ifrs=("CostOfSales",),
            notes={
                "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization": "cost of sales excluding depreciation, depletion and amortization; gross profit is before D&A",
                "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization": "cost of goods sold excluding depreciation, depletion and amortization; gross profit is before D&A",
                "CostOfServicesExcludingDepreciationDepletionAndAmortization": "cost of services excluding depreciation, depletion and amortization; gross profit is before D&A",
                "CostOfPurchasedOilAndGas": "purchased crude oil, gas and products (the cost-of-sales line of energy filers)",
            },
        ),
        _m("gross_profit", "Gross profit", "duration", "money", True, us=("GrossProfit",), ifrs=("GrossProfit",)),
        _m(
            "operating_income", "Operating income", "duration", "money", True,
            us=("OperatingIncomeLoss",),
            ifrs=("ProfitLossFromOperatingActivities",),
        ),
        _m(
            "net_income", "Net income attributable to parent", "duration", "money", True,
            us=("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"),
            ifrs=("ProfitLossAttributableToOwnersOfParent", "ProfitLoss"),
            notes={
                "NetIncomeLossAvailableToCommonStockholdersBasic": "after preferred dividends; not identical to NI attributable to parent",
                "ProfitLoss": "includes noncontrolling interests; NCI income subtracted where tagged, otherwise used as reported",
            },
        ),
        _m(
            "nci_income", "Net income attributable to noncontrolling interest", "duration", "money", True,
            us=("NetIncomeLossAttributableToNoncontrollingInterest",),
            ifrs=("ProfitLossAttributableToNoncontrollingInterests",),
        ),
        _m(
            "eps_diluted", "Diluted EPS", "duration", "per_share", False,
            us=("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
            ifrs=("DilutedEarningsLossPerShare", "BasicAndDilutedEarningsLossPerShare"),
        ),
        _m(
            "diluted_shares", "Weighted diluted shares", "duration", "shares", False,
            us=(
                "WeightedAverageNumberOfDilutedSharesOutstanding",
                "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
                "WeightedAverageNumberOfSharesOutstandingBasic",
            ),
            ifrs=("AdjustedWeightedAverageShares", "WeightedAverageShares"),
            notes={
                "WeightedAverageNumberOfSharesOutstandingBasic": "basic count used because no diluted count is tagged",
                "WeightedAverageShares": "basic count used because no diluted count is tagged",
            },
        ),
        _m(
            "basic_shares", "Weighted basic shares", "duration", "shares", False,
            us=("WeightedAverageNumberOfSharesOutstandingBasic", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"),
            ifrs=("WeightedAverageShares",),
        ),
        _m(
            "ocf", "Operating cash flow", "duration", "money", True,
            us=("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
            ifrs=("CashFlowsFromUsedInOperatingActivities",),
            notes={"NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": "continuing operations only"},
        ),
        _m(
            "capex", "Capital expenditures", "duration", "money", True,
            us=(
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "PaymentsToAcquireProductiveAssets",
                "PaymentsForCapitalImprovements",
                "PaymentsToAcquireOtherProductiveAssets",
                "PaymentsToAcquireOilAndGasPropertyAndEquipment",
                "PaymentsToExploreAndDevelopOilAndGasProperties",
                "PaymentsToAcquireOtherPropertyPlantAndEquipment",
                "PaymentsForConstructionInProcess",
                "PaymentsForProceedsFromProductiveAssets",
                "PaymentsToAcquireFurnitureAndFixtures",
            ),
            ifrs=("PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "PurchaseOfPropertyPlantAndEquipment"),
            notes={
                "PaymentsToAcquireProductiveAssets": "may include intangibles and other productive assets",
                "PaymentsToAcquireOtherProductiveAssets": "'other productive assets' line used as capex (no PP&E capex tag for the period)",
                "PaymentsToAcquireOilAndGasPropertyAndEquipment": "oil and gas property additions; other PP&E lines of the same period are added",
                "PaymentsToAcquireOtherPropertyPlantAndEquipment": "'other PP&E' line; other PP&E lines of the same period are added",
                "PaymentsToExploreAndDevelopOilAndGasProperties": "exploration and development additions; other PP&E lines of the same period are added",
                "PaymentsForConstructionInProcess": "construction expenditures (utility/REIT presentation)",
                "PaymentsForProceedsFromProductiveAssets": "net of disposal proceeds (the only capex line tagged)",
                "PaymentsToAcquireFurnitureAndFixtures": "furniture and fixtures purchases are the only PP&E purchase line tagged",
            },
        ),
        _m(
            "da", "Depreciation & amortization (cash flow)", "duration", "money", True,
            us=(
                "DepreciationDepletionAndAmortization",
                "DepreciationAmortizationAndAccretionNet",
                "DepreciationAndAmortization",
                "Depreciation",
            ),
            ifrs=(
                "AdjustmentsForDepreciationAndAmortisationExpense",
                "DepreciationAndAmortisationExpense",
            ),
            notes={
                "Depreciation": "depreciation only; amortization not included",
                "DepreciationAmortizationAndAccretionNet": "includes accretion",
                "DepreciationAndAmortisationExpense": "income-statement D&A, not the cash-flow add-back",
            },
        ),
        _m(
            "dividends_paid", "Dividends paid (cash flow)", "duration", "money", True,
            us=("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
            ifrs=("DividendsPaidClassifiedAsFinancingActivities", "DividendsPaid"),
            notes={"PaymentsOfDividends": "may include preferred and NCI dividends"},
        ),
        _m(
            "dps_declared", "Dividends declared per share (reference only)", "duration", "per_share", False,
            us=("CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"),
            ifrs=("DividendsRecognisedAsDistributionsToOwnersPerShare",),
        ),
        # one-time items used only for the red-flag screen
        _m(
            "impairment", "Impairment charges", "duration", "money", True,
            us=(
                "AssetImpairmentCharges",
                "GoodwillImpairmentLoss",
                "ImpairmentOfLongLivedAssetsHeldForUse",
                "ImpairmentOfIntangibleAssetsExcludingGoodwill",
                "GoodwillAndIntangibleAssetImpairment",
            ),
            ifrs=("ImpairmentLossRecognisedInProfitOrLoss",),
        ),
        _m(
            "disc_ops", "Income from discontinued operations", "duration", "money", True,
            us=(
                "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
                "IncomeLossFromDiscontinuedOperationsNetOfTax",
            ),
            ifrs=("ProfitLossFromDiscontinuedOperations",),
        ),
        _m(
            "disposal_gains", "Gain (loss) on disposals", "duration", "money", True,
            us=(
                "GainLossOnDispositionOfAssets",
                "GainLossOnSaleOfPropertyPlantEquipment",
                "DisposalGroupNotDiscontinuedOperationGainLossOnDisposal",
                "GainLossOnSaleOfBusiness",
            ),
            ifrs=("GainsLossesOnDisposalsOfPropertyPlantAndEquipment",),
        ),
        # ----- balance sheet (instants) -----
        _m(
            "cash", "Cash and cash equivalents", "instant", "money", False,
            us=(
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                "Cash",
            ),
            ifrs=("CashAndCashEquivalents",),
            notes={"CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "includes restricted cash"},
        ),
        _m("total_assets", "Total assets", "instant", "money", False, us=("Assets",), ifrs=("Assets",)),
        _m("total_liabilities", "Total liabilities", "instant", "money", False, us=("Liabilities",), ifrs=("Liabilities",)),
        _m(
            "equity", "Parent equity", "instant", "money", False,
            us=("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
            ifrs=("EquityAttributableToOwnersOfParent", "Equity"),
            notes={
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "total equity; noncontrolling interest subtracted where tagged at the same date",
                "Equity": "total equity; noncontrolling interest subtracted where tagged at the same date",
            },
        ),
        _m(
            "long_term_debt", "Long-term debt", "instant", "money", False,
            us=("LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"),
            ifrs=("NoncurrentPortionOfNoncurrentBorrowings", "LongtermBorrowings"),
            notes={
                "LongTermDebt": "includes current maturities",
                "LongTermDebtAndCapitalLeaseObligations": "includes finance-lease obligations",
            },
        ),
        _m("goodwill", "Goodwill", "instant", "money", False, us=("Goodwill",), ifrs=("Goodwill",)),
        _m(
            "intangibles", "Intangible assets excl. goodwill", "instant", "money", False,
            us=("IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"),
            ifrs=("IntangibleAssetsOtherThanGoodwill",),
            notes={"FiniteLivedIntangibleAssetsNet": "finite-lived only"},
        ),
        _m("nci", "Noncontrolling interest", "instant", "money", False, us=("MinorityInterest",), ifrs=("NoncontrollingInterests",)),
        _m("preferred", "Preferred stock", "instant", "money", False, us=("PreferredStockValue", "PreferredStockValueOutstanding"), ifrs=()),
        _m(
            "bs_shares", "Shares outstanding (balance sheet)", "instant", "shares", False,
            us=("CommonStockSharesOutstanding",),
            ifrs=("NumberOfSharesOutstanding",),
        ),
    ]
}

# Borrowing components for EV. Each slot is resolved separately so the
# workbook can show what was tagged. "LongTermDebt" (which includes current
# maturities) is only used when no noncurrent tag exists.
DEBT_TAGS = {
    US: {
        "debt_current_total": ("DebtCurrent",),
        "ltd_current": ("LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
        "short_term": ("ShortTermBorrowings", "CommercialPaper"),
        "ltd_noncurrent": ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"),
        "ltd_total": ("LongTermDebt",),
    },
    IFRS: {
        "debt_current_total": ("CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings",),
        "ltd_current": ("CurrentPortionOfNoncurrentBorrowings", "CurrentPortionOfLongtermBorrowings"),
        "short_term": ("ShorttermBorrowings",),
        "ltd_noncurrent": ("NoncurrentPortionOfNoncurrentBorrowings", "LongtermBorrowings"),
        "ltd_total": ("Borrowings",),
    },
}

# Cost-of-revenue fallbacks that some filers use for a single component of cost of sales (e.g. equipment
# cost only). Per company, such a tag is used only if it never disagrees with a higher-priority cost tag,
# or with revenue minus tagged GrossProfit, in a period both report (values.screen_component_cost_tags).
COMPONENT_SCREENED_COST_TAGS = frozenset(
    {
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
        "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization",
        "CostOfServicesExcludingDepreciationDepletionAndAmortization",
        "CostOfPurchasedOilAndGas",
    }
)

SPLIT_RATIO_TAGS = ("StockholdersEquityNoteStockSplitConversionRatio1",)
COVER_SHARES_TAG = "EntityCommonStockSharesOutstanding"

# Tags whose periods define the fiscal calendar and filing report periods.
CALENDAR_METRICS = ("revenue", "net_income", "operating_income", "ocf", "eps_diluted", "diluted_shares", "gross_profit")
BALANCE_ANCHOR_METRICS = ("total_assets", "equity", "cash")

PERIODIC_FORMS = {"10-K", "10-Q", "20-F", "40-F", "10-KT", "10-QT", "10-K405"}

# Split-like factors (shares multiply by k). Reverse splits are 1/k.
SPLIT_FACTORS = (1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100)


def all_tags_for(metric: MetricDef) -> list[tuple[str, str, int]]:
    """(taxonomy, tag, priority) for every candidate of a metric."""
    out = []
    for taxonomy, tags in metric.candidates.items():
        for priority, tag in enumerate(tags):
            out.append((taxonomy, tag, priority))
    return out

# Capex presentations that split PP&E purchases over several cash-flow lines. When the line chosen
# for a period is one of the COMPONENT tags, the other component lines of the same period are added
# (they are separate rows of the same statement, e.g. oil and gas additions + other PP&E, or capital
# improvements + construction in process). A TOTAL tag is used alone, unless construction
# expenditures for the same period exceed it (then the "total" tag holds a smaller separate line,
# e.g. a utility's asset acquisitions, and construction expenditures are the capex line).
CAPEX_TOTAL_TAGS = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
CAPEX_COMPONENT_TAGS = (
    "PaymentsToAcquireOilAndGasPropertyAndEquipment",
    "PaymentsToExploreAndDevelopOilAndGasProperties",
    "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    "PaymentsForCapitalImprovements",
    "PaymentsForConstructionInProcess",
)
