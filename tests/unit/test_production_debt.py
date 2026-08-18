import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../guidance/library/production_debt.py",
)
spec = importlib.util.spec_from_file_location("guidance_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["guidance_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtGuidanceGate = production_debt_mod.ProductionDebtGuidanceGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtGuidanceGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtGuidanceGate(
            never_equate_intent_to_approval=True,
            max_acceptable_gdi=12.0,
        )

    def test_clean_generation_passes_readiness(self) -> None:
        report = self.gate.evaluate_generation(
            grammar_id="pydantic_schema_invoice_parser",
            expected_schema_tokens=150,
            actual_generated_tokens=160,
            generation_latency_seconds=0.45,
            grammar_retry_count=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.gdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_generation_fails_debt(self) -> None:
        report = self.gate.evaluate_generation(
            grammar_id="unconstrained_recursive_ast",
            expected_schema_tokens=100,
            actual_generated_tokens=350,  # High token inflation (3.5x)
            generation_latency_seconds=4.5,  # High generation latency
            grammar_retry_count=3,  # 3 grammar retry cycles
            un_gated_mutations=2,  # 2 un-gated AST mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.gdi_score, 50.0)
        self.assertIn("HIGH_TOKEN_INFLATION_3.50X", report.critical_smells)
        self.assertIn("HIGH_GENERATION_LATENCY_4.50S", report.critical_smells)
        self.assertIn("DETECTED_3_GRAMMAR_RETRIES", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_AST_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_generation("grammar-1")
        self.gate.evaluate_generation("grammar-2")
        self.gate.evaluate_generation("grammar-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
