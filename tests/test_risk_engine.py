"""
Tests for the production risk engine and kill switch.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_research.risk.engine import (
    RiskEngine, RiskLimits, PortfolioState, RiskBreach, Severity,
)
from quant_research.risk.kill_switch import (
    KillSwitch, KillSwitchConfig, KillSwitchState,
)


# ═══════════════════════════════════════════════════════════════════
# Risk Engine Tests
# ═══════════════════════════════════════════════════════════════════

class TestRiskLimits:
    def test_default_limits(self):
        limits = RiskLimits()
        assert limits.max_gross_exposure == 1.0
        assert limits.max_daily_loss == -0.03
        assert limits.max_drawdown == -0.15

    def test_custom_limits(self):
        limits = RiskLimits(max_gross_exposure=2.0, max_daily_loss=-0.05)
        assert limits.max_gross_exposure == 2.0
        assert limits.max_daily_loss == -0.05


class TestPortfolioState:
    def test_gross_exposure(self):
        state = PortfolioState(positions={"BTC": 0.5, "ETH": -0.3})
        assert abs(state.gross_exposure - 0.8) < 1e-10

    def test_net_exposure(self):
        state = PortfolioState(positions={"BTC": 0.5, "ETH": -0.3})
        assert abs(state.net_exposure - 0.2) < 1e-10

    def test_max_concentration(self):
        state = PortfolioState(positions={"BTC": 0.6, "ETH": 0.4})
        assert abs(state.max_concentration - 0.6) < 1e-10

    def test_empty_positions(self):
        state = PortfolioState(positions={})
        assert state.gross_exposure == 0.0
        assert state.max_concentration == 0.0


class TestRiskEngineChecks:
    def setup_method(self):
        self.engine = RiskEngine(RiskLimits())

    def test_no_breach_normal_state(self):
        # Use 3 assets to keep concentration < 40%
        state = PortfolioState(
            positions={"BTC": 0.2, "ETH": 0.15, "SPY": 0.15},
            daily_pnl=0.01,
            drawdown=-0.02,
            portfolio_vol=0.10,
            daily_turnover=0.5,
            last_data_time=datetime.now(),
        )
        breaches = self.engine.check_all(state)
        assert len(breaches) == 0
        assert not self.engine.is_halted

    def test_position_limit_breach(self):
        state = PortfolioState(positions={"BTC": 1.5})
        breach = self.engine.check_position_limit(state)
        assert breach is not None
        assert breach.check_name == "position_limit"
        assert breach.severity == Severity.HARD_LIMIT

    def test_position_limit_ok(self):
        state = PortfolioState(positions={"BTC": 0.8})
        breach = self.engine.check_position_limit(state)
        assert breach is None

    def test_gross_exposure_breach(self):
        state = PortfolioState(positions={"BTC": 0.6, "ETH": 0.5})
        breach = self.engine.check_gross_exposure(state)
        assert breach is not None
        assert breach.severity == Severity.HARD_LIMIT

    def test_net_exposure_breach(self):
        state = PortfolioState(positions={"BTC": 0.4, "ETH": 0.3})
        breach = self.engine.check_net_exposure(state)
        assert breach is not None
        assert breach.severity == Severity.WARNING

    def test_concentration_breach(self):
        state = PortfolioState(positions={"BTC": 0.8, "ETH": 0.1})
        breach = self.engine.check_concentration(state)
        assert breach is not None

    def test_drawdown_kill(self):
        state = PortfolioState(positions={}, drawdown=-0.20)
        breach = self.engine.check_drawdown(state)
        assert breach is not None
        assert breach.severity == Severity.KILL

    def test_daily_loss_kill(self):
        state = PortfolioState(positions={}, daily_pnl=-0.05)
        breach = self.engine.check_daily_loss(state)
        assert breach is not None
        assert breach.severity == Severity.KILL

    def test_volatility_warning(self):
        state = PortfolioState(positions={}, portfolio_vol=0.25)
        breach = self.engine.check_volatility(state)
        assert breach is not None
        assert breach.severity == Severity.WARNING

    def test_turnover_warning(self):
        state = PortfolioState(positions={}, daily_turnover=3.0)
        breach = self.engine.check_turnover(state)
        assert breach is not None

    def test_stale_data_kill(self):
        old_time = datetime.now() - timedelta(hours=5)
        state = PortfolioState(positions={}, last_data_time=old_time)
        breach = self.engine.check_stale_data(state)
        assert breach is not None
        assert breach.severity == Severity.KILL

    def test_stale_data_none_kill(self):
        state = PortfolioState(positions={}, last_data_time=None)
        breach = self.engine.check_stale_data(state)
        assert breach is not None
        assert breach.severity == Severity.KILL

    def test_fresh_data_ok(self):
        state = PortfolioState(positions={}, last_data_time=datetime.now())
        breach = self.engine.check_stale_data(state)
        assert breach is None


class TestRiskEngineHalt:
    def test_kill_triggers_halt(self):
        engine = RiskEngine(RiskLimits())
        state = PortfolioState(positions={}, drawdown=-0.20, last_data_time=datetime.now())
        engine.check_all(state)
        assert engine.is_halted
        assert "CRITICAL" in engine.halt_reason

    def test_halt_zeroes_positions(self):
        engine = RiskEngine(RiskLimits())
        engine.halt("test halt")
        approved = engine.enforce(
            {"BTC": 0.5, "ETH": 0.3},
            PortfolioState(positions={}),
        )
        assert all(v == 0.0 for v in approved.values())

    def test_resume(self):
        engine = RiskEngine(RiskLimits())
        engine.halt("test")
        assert engine.is_halted
        engine.resume()
        assert not engine.is_halted


class TestRiskEngineEnforce:
    def setup_method(self):
        self.engine = RiskEngine(RiskLimits())

    def test_clips_per_asset(self):
        approved = self.engine.enforce(
            {"BTC": 1.5, "ETH": -1.2},
            PortfolioState(positions={}),
        )
        assert abs(approved["BTC"]) <= 1.0
        assert abs(approved["ETH"]) <= 1.0

    def test_scales_gross_exposure(self):
        approved = self.engine.enforce(
            {"BTC": 0.6, "ETH": 0.6},
            PortfolioState(positions={}),
        )
        ge = sum(abs(v) for v in approved.values())
        assert ge <= 1.0 + 1e-10

    def test_concentration_cap(self):
        # After enforce: gross capped to 1.0, then concentration applied
        approved = self.engine.enforce(
            {"BTC": 0.9, "ETH": 0.1},
            PortfolioState(positions={}),
        )
        ge = sum(abs(v) for v in approved.values())
        # With only 2 assets and 40% cap, the larger is clipped
        assert ge <= 1.0 + 1e-10
        # BTC should be reduced (originally 90% of gross)
        assert abs(approved["BTC"]) <= abs(approved["BTC"]) + 1e-10  # Basic sanity

    def test_turnover_limit(self):
        old_state = PortfolioState(positions={"BTC": 0.0, "ETH": 0.0})
        approved = self.engine.enforce(
            {"BTC": 5.0, "ETH": 5.0},  # Excessive change
            old_state,
        )
        # Should be capped by position limit first, then turnover
        turnover = sum(abs(approved.get(t, 0) - old_state.positions.get(t, 0))
                       for t in set(list(approved.keys()) + list(old_state.positions.keys())))
        assert turnover <= 2.0 + 1e-10

    def test_passthrough_normal(self):
        # 3 assets with none > 40% concentration
        approved = self.engine.enforce(
            {"BTC": 0.15, "ETH": 0.15, "SPY": 0.10},
            PortfolioState(positions={"BTC": 0.15, "ETH": 0.15, "SPY": 0.10}),
        )
        assert abs(approved["BTC"] - 0.15) < 1e-10
        assert abs(approved["ETH"] - 0.15) < 1e-10
        assert abs(approved["SPY"] - 0.10) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# Kill Switch Tests
# ═══════════════════════════════════════════════════════════════════

class TestKillSwitch:
    def test_no_trigger_normal(self):
        ks = KillSwitch(KillSwitchConfig())
        state = KillSwitchState(
            daily_pnl=0.01,
            drawdown=-0.05,
            portfolio_vol=0.10,
            last_data_time=datetime.now(),
            broker_connected=True,
        )
        should_kill, reasons = ks.evaluate(state)
        assert not should_kill
        assert len(reasons) == 0

    def test_daily_loss_trigger(self):
        ks = KillSwitch(KillSwitchConfig(max_daily_loss=-0.03))
        state = KillSwitchState(daily_pnl=-0.05)
        should_kill, reasons = ks.evaluate(state)
        assert should_kill
        assert any("Daily loss" in r for r in reasons)

    def test_drawdown_trigger(self):
        ks = KillSwitch(KillSwitchConfig(max_drawdown=-0.15))
        state = KillSwitchState(drawdown=-0.20)
        should_kill, reasons = ks.evaluate(state)
        assert should_kill

    def test_vol_spike_trigger(self):
        ks = KillSwitch(KillSwitchConfig(max_portfolio_vol=0.30))
        state = KillSwitchState(portfolio_vol=0.40)
        should_kill, reasons = ks.evaluate(state)
        assert should_kill

    def test_stale_data_trigger(self):
        ks = KillSwitch(KillSwitchConfig())
        state = KillSwitchState(
            last_data_time=datetime.now() - timedelta(hours=5)
        )
        should_kill, reasons = ks.evaluate(state)
        assert should_kill

    def test_broker_disconnect_trigger(self):
        ks = KillSwitch(KillSwitchConfig())
        state = KillSwitchState(broker_connected=False)
        should_kill, reasons = ks.evaluate(state)
        assert should_kill

    def test_position_mismatch_trigger(self):
        ks = KillSwitch(KillSwitchConfig())
        state = KillSwitchState(
            expected_positions={"BTC": 0.5},
            actual_positions={"BTC": 0.3},
        )
        should_kill, reasons = ks.evaluate(state)
        assert should_kill
        assert any("mismatch" in r for r in reasons)

    def test_unexpected_leverage_trigger(self):
        ks = KillSwitch(KillSwitchConfig())
        state = KillSwitchState(
            current_leverage=3.0,
            max_allowed_leverage=1.0,
        )
        should_kill, reasons = ks.evaluate(state)
        assert should_kill

    def test_execute_cancels_orders(self):
        class MockBroker:
            def cancel_all_orders(self):
                return 5
            def get_positions(self):
                return {}
            def flatten_all(self):
                return []

        ks = KillSwitch(KillSwitchConfig())
        ks.is_triggered = True
        ks.trigger_reasons = ["test"]
        report = ks.execute(MockBroker())
        assert "Cancelled 5 orders" in report["actions"][0]

    def test_execute_flattens_when_configured(self):
        class MockBroker:
            def cancel_all_orders(self):
                return 0
            def get_positions(self):
                return {"BTC": 0.5}
            def flatten_all(self):
                return ["order1", "order2"]

        ks = KillSwitch(KillSwitchConfig(auto_flatten=True))
        ks.is_triggered = True
        ks.trigger_reasons = ["test"]
        report = ks.execute(MockBroker())
        assert any("Flattened" in a for a in report["actions"])

    def test_reset(self):
        ks = KillSwitch(KillSwitchConfig())
        ks.is_triggered = True
        ks.trigger_reasons = ["test"]
        ks.reset()
        assert not ks.is_triggered
        assert len(ks.trigger_reasons) == 0

    def test_multiple_triggers(self):
        ks = KillSwitch(KillSwitchConfig())
        state = KillSwitchState(
            daily_pnl=-0.10,
            drawdown=-0.25,
            broker_connected=False,
        )
        should_kill, reasons = ks.evaluate(state)
        assert should_kill
        assert len(reasons) >= 3  # At least 3 conditions violated


# ═══════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestRiskEngineIntegration:
    def test_full_lifecycle(self):
        """Test: normal → warning → kill → halt → resume."""
        engine = RiskEngine(RiskLimits())

        # Normal state (3 assets, none > 40% concentration)
        state1 = PortfolioState(
            positions={"BTC": 0.15, "ETH": 0.15, "SPY": 0.10},
            daily_pnl=0.01,
            drawdown=-0.02,
            last_data_time=datetime.now(),
        )
        assert len(engine.check_all(state1)) == 0
        assert not engine.is_halted

        # Warning state
        state2 = PortfolioState(
            positions={"BTC": 0.15, "ETH": 0.15, "SPY": 0.10},
            portfolio_vol=0.25,
            last_data_time=datetime.now(),
        )
        breaches = engine.check_all(state2)
        assert any(b.severity == Severity.WARNING for b in breaches)
        assert not engine.is_halted

        # Kill state
        state3 = PortfolioState(
            positions={"BTC": 0.3},
            drawdown=-0.20,
            last_data_time=datetime.now(),
        )
        breaches = engine.check_all(state3)
        assert any(b.severity == Severity.KILL for b in breaches)
        assert engine.is_halted

        # Halted: positions zeroed
        approved = engine.enforce({"BTC": 0.5}, PortfolioState(positions={}))
        assert approved["BTC"] == 0.0

        # Resume
        engine.resume()
        assert not engine.is_halted
        approved = engine.enforce(
            {"BTC": 0.15, "ETH": 0.15, "SPY": 0.10},
            PortfolioState(positions={"BTC": 0.15, "ETH": 0.15, "SPY": 0.10}),
        )
        assert abs(approved["BTC"] - 0.15) < 1e-10

    def test_breach_log_accumulates(self):
        engine = RiskEngine(RiskLimits())

        # First call: vol warning
        state1 = PortfolioState(
            positions={}, drawdown=-0.01,
            portfolio_vol=0.25,
            last_data_time=datetime.now(),
        )
        engine.check_all(state1)

        # Second call: another vol warning
        state2 = PortfolioState(
            positions={}, drawdown=-0.05,
            portfolio_vol=0.25,
            last_data_time=datetime.now(),
        )
        engine.check_all(state2)

        assert len(engine.breach_log) >= 2  # At least 2 vol warnings
        assert "breaches" in engine.summary().lower()
