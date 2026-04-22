"""
Smoke-tests for Python code blocks in the MkDocs guide pages.

Each test is a self-contained function that reproduces the key logic from
the corresponding doc block. Tests that require live network access are
skipped by default (mark them explicitly to run).
"""
import fundcloud  # noqa: F401 - registers .fc accessor
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_bars():
    """Four-asset synthetic OHLCV bar panel used across many tests."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2022-01-03", periods=504)

    def _asset(p0, vol):
        c = p0 + np.cumsum(rng.normal(0, vol, len(idx)))
        return {"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1e6}

    bars = pd.concat(
        {
            "AAPL": pd.DataFrame(_asset(180.0, 1.2), index=idx),
            "MSFT": pd.DataFrame(_asset(400.0, 2.0), index=idx),
            "BND":  pd.DataFrame(_asset(75.0,  0.3), index=idx),
            "GLD":  pd.DataFrame(_asset(180.0, 1.2), index=idx),
        },
        axis=1,
    )
    bars.columns = bars.columns.swaplevel(0, 1)
    return bars.sort_index(axis=1)


@pytest.fixture(scope="module")
def returns_series(synth_bars):
    return synth_bars.xs("close", level=0, axis=1)["AAPL"].pct_change().dropna()


@pytest.fixture(scope="module")
def returns_df(synth_bars):
    return synth_bars.xs("close", level=0, axis=1).pct_change().dropna()


@pytest.fixture(scope="module")
def spy_returns(synth_bars):
    return synth_bars.xs("close", level=0, axis=1)["MSFT"].pct_change().dropna()


# ---------------------------------------------------------------------------
# quickstart.md
# ---------------------------------------------------------------------------

class TestQuickstart:
    def test_block1_accessor_import(self):
        returns = pd.Series([0.012, -0.005, 0.008, -0.010, 0.015])
        assert hasattr(returns, "fc")
        returns.fc.sharpe()

    def test_block2_synthetic_data(self):
        rng = np.random.default_rng(0)
        idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=252, freq="B").values)

        def asset(price0, vol):
            close = price0 + np.cumsum(rng.normal(0, vol, len(idx)))
            return {"open": close, "high": close + 0.5, "low": close - 0.5,
                    "close": close, "volume": 1_000_000.0}

        bars = pd.concat(
            {"AAPL": pd.DataFrame(asset(180.0, 1.2), index=idx),
             "MSFT": pd.DataFrame(asset(400.0, 2.0), index=idx)},
            axis=1,
        )
        bars.columns = bars.columns.swaplevel(0, 1)
        bars = bars.sort_index(axis=1)
        assert bars.shape[0] == 252

    def test_block3_simulator_dca(self, synth_bars):
        from fundcloud.sim import Simulator
        from fundcloud.strategies import DCA

        result = Simulator(synth_bars, cash=100_000).run_strategy(
            DCA(amount=1_000, horizon="weekly",
                weights={"AAPL": 0.5, "MSFT": 0.5})
        )
        result.portfolio.sharpe()
        result.portfolio.max_drawdown()
        result.equity_curve.tail()

    def test_block5_plots_summary(self, synth_bars):
        import fundcloud as fc
        from fundcloud.sim import Simulator
        from fundcloud.strategies import DCA

        result = Simulator(synth_bars, cash=100_000).run_strategy(
            DCA(amount=1_000, horizon="weekly", weights={"AAPL": 0.5, "MSFT": 0.5})
        )
        fc.set_theme("dark")
        fig = fc.plots.summary(result.portfolio.returns)
        assert fig is not None

    def test_block6_sklearn_pipeline(self, returns_df):
        from fundcloud.features import FeaturePipeline
        from fundcloud.features.indicators import RSI, SMA
        from fundcloud.optimize import MeanRisk, RiskMeasure
        from fundcloud.validate import PurgedKFold
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline

        # verify the pipeline objects can be constructed correctly
        pipe = Pipeline([
            ("features", FeaturePipeline([("rsi", RSI(timeperiod=14)), ("sma", SMA(timeperiod=20))])),
            ("optim",    MeanRisk(risk_measure=RiskMeasure.CVAR)),
        ])
        grid = GridSearchCV(
            pipe,
            param_grid={"optim__min_weights": [0.0, 0.02]},
            cv=PurgedKFold(n_splits=5, purge=3),
        )
        # verify params are accessible
        assert "optim__min_weights" in grid.param_grid


# ---------------------------------------------------------------------------
# guides/portfolio/returns-analysis.md
# ---------------------------------------------------------------------------

class TestReturnsAnalysis:
    def test_metrics_series(self, returns_series):
        result = returns_series.fc.metrics()
        assert "sharpe" in result.index

    def test_summary_dataframe(self, returns_df):
        result = returns_df.fc.summary()
        assert result.shape[1] == returns_df.shape[1]

    def test_portfolio_worst_drawdowns(self, returns_series):
        from fundcloud.portfolio import Portfolio
        pf = Portfolio(returns=returns_series, name="test")
        pf.worst_drawdowns(top=5)
        pf.drawdown_details()

    def test_drawdown_series(self, returns_series):
        dd = returns_series.fc.drawdown_series()
        assert (dd <= 0).all()
        dd.idxmin()

    def test_period_returns(self, returns_series, spy_returns):
        from fundcloud.portfolio import Portfolio
        pf = Portfolio(returns=returns_series, name="test")
        pf.period_returns(benchmark=spy_returns)
        pf.yearly_returns(benchmark=spy_returns)

    def test_rolling_metrics(self, returns_series, spy_returns):
        window = 63
        returns_series.fc.rolling_sharpe(window=window)
        returns_series.fc.rolling_volatility(window=21)
        returns_series.fc.rolling_drawdown()
        returns_series.fc.rolling_beta(spy_returns, window=window)


# ---------------------------------------------------------------------------
# guides/portfolio/metrics.md
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_core_metrics(self, returns_series):
        r = returns_series
        r.fc.sharpe()
        r.fc.drawdown_series()
        r.fc.metrics()

    def test_sharpe_variants(self, returns_series):
        r = returns_series
        r.fc.smart_sharpe()
        r.fc.smart_sortino()
        r.fc.probabilistic_sharpe(target_sharpe=0.5)   # fixed name
        r.fc.adjusted_sortino()
        r.fc.kelly_criterion()
        r.fc.risk_of_ruin(ruin_level=0.5)              # fixed kwarg

    def test_tail_pain_metrics(self, returns_series):
        r = returns_series
        r.fc.tail_ratio()
        r.fc.gain_to_pain_ratio()
        r.fc.pain_index()
        r.fc.pain_ratio()
        r.fc.common_sense_ratio()
        r.fc.ulcer_performance_index()

    def test_trade_quality_metrics(self, returns_series):
        r = returns_series
        r.fc.win_rate()
        r.fc.avg_win()
        r.fc.avg_loss()
        r.fc.payoff_ratio()
        r.fc.profit_factor()
        r.fc.consecutive_wins()
        r.fc.consecutive_losses()
        r.fc.exposure()

    def test_rolling_risk_adjusted(self, returns_series, spy_returns):
        window = 63
        returns_series.fc.rolling_beta(spy_returns, window=window)
        returns_series.fc.rolling_sharpe(window=window)
        returns_series.fc.rolling_sortino(window=window)


# ---------------------------------------------------------------------------
# guides/strategies/dca.md
# ---------------------------------------------------------------------------

class TestStrategies:
    def test_hold_strategy(self, synth_bars):
        from fundcloud.sim import Simulator
        from fundcloud.strategies import Hold

        s = Hold(weights={"AAPL": 0.6, "MSFT": 0.4})
        result = Simulator(synth_bars, cash=100_000).run_strategy(s)
        assert result.portfolio is not None

    def test_dca_strategy(self, synth_bars):
        from fundcloud.sim import Simulator
        from fundcloud.strategies import DCA

        strategy = DCA(
            amount=2_000, horizon="weekly",
            weights={"AAPL": 0.4, "MSFT": 0.3, "BND": 0.2, "GLD": 0.1},
        )
        result = Simulator(synth_bars, cash=200_000).run_strategy(strategy)
        result.portfolio.summary()

    def test_dca_vs_hold_comparison(self, synth_bars):
        from fundcloud.sim import Simulator
        from fundcloud.strategies import DCA, Hold

        WEIGHTS = {"AAPL": 0.6, "MSFT": 0.4}
        dca  = Simulator(synth_bars, cash=120_000).run_strategy(
            DCA(amount=1_000, horizon="weekly", weights=WEIGHTS)
        )
        hold = Simulator(synth_bars, cash=120_000).run_strategy(
            Hold(weights=WEIGHTS)
        )
        comparison = pd.concat(
            {"DCA": dca.portfolio.summary(), "Hold": hold.portfolio.summary()},
            axis=1,
        )
        assert comparison.shape[1] == 2

    def test_trades_inspection(self, synth_bars):
        from fundcloud.sim import Simulator
        from fundcloud.strategies import DCA

        result = Simulator(synth_bars, cash=100_000).run_strategy(
            DCA(amount=500, horizon="weekly", weights={"AAPL": 0.5, "MSFT": 0.5})
        )
        buys = result.trades[result.trades["qty"] > 0]
        assert len(buys) > 0


# ---------------------------------------------------------------------------
# guides/sim/simulator.md
# ---------------------------------------------------------------------------

class TestSimulator:
    def test_simulator_with_costs(self, synth_bars):
        from fundcloud.sim import FixedBps, Simulator
        from fundcloud.strategies import DCA

        result = Simulator(synth_bars, cash=100_000, costs=FixedBps(10)).run_strategy(
            DCA(amount=500, horizon="weekly", weights={"AAPL": 0.5, "MSFT": 0.5})
        )
        result.portfolio.sharpe()
        result.trades[["ts", "asset", "qty", "price", "fee"]].head()


# ---------------------------------------------------------------------------
# guides/portfolio/optimization.md
# ---------------------------------------------------------------------------

class TestOptimization:
    def test_hrp(self, returns_df):
        from fundcloud.optimize import HierarchicalRiskParity
        hrp = HierarchicalRiskParity()
        hrp.fit(returns_df)
        opt_pf = hrp.predict(returns_df)
        # weights is a (1, n_assets) DataFrame
        assert abs(opt_pf.weights.values.sum() - 1.0) < 1e-6

    def test_mean_risk_variants(self, returns_df):
        from fundcloud.optimize import MeanRisk, RiskMeasure
        for rm in [RiskMeasure.VARIANCE, RiskMeasure.CVAR]:
            opt = MeanRisk(risk_measure=rm)
            opt.fit(returns_df)
            w = opt.predict(returns_df).weights
            # weights is a (1, n_assets) DataFrame
            assert abs(float(w.values.sum()) - 1.0) < 1e-6

    def test_equal_weighted_invvol(self, returns_df):
        from fundcloud.optimize import EqualWeighted, InverseVolatility
        assert EqualWeighted().fit(returns_df).predict(returns_df).weights is not None
        assert InverseVolatility().fit(returns_df).predict(returns_df).weights is not None

    def test_cross_validated_optimization(self, returns_df):
        from fundcloud.optimize import MeanRisk, RiskMeasure
        from fundcloud.portfolio import Portfolio
        from fundcloud.validate import PurgedKFold

        cv = PurgedKFold(n_splits=3, purge=21)
        oos_sharpes = []
        for _, (train_idx, test_idx) in enumerate(cv.split(returns_df)):
            train = returns_df.iloc[train_idx]
            test  = returns_df.iloc[test_idx]
            opt = MeanRisk(risk_measure=RiskMeasure.CVAR, min_weights=0.02)
            opt.fit(train)
            w = opt.predict(train).weights.to_numpy().squeeze()  # (n_assets,)
            oos_rets = test.to_numpy() @ w
            pf = Portfolio(returns=pd.Series(oos_rets, index=test.index))
            oos_sharpes.append(pf.sharpe())
        assert len(oos_sharpes) == 3

    def test_grid_search_optimization(self, returns_df):
        from fundcloud.optimize import MeanRisk, RiskMeasure
        from fundcloud.validate import PurgedKFold
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([("optim", MeanRisk(risk_measure=RiskMeasure.CVAR))])
        search = GridSearchCV(
            pipe,
            param_grid={"optim__min_weights": [0.0, 0.02]},
            cv=PurgedKFold(n_splits=3, purge=21),
        )
        search.fit(returns_df)
        assert search.best_params_ is not None


# ---------------------------------------------------------------------------
# guides/data/backends-and-catalog.md
# ---------------------------------------------------------------------------

class TestDataBackends:
    def test_catalog(self, tmp_path):
        import numpy as np
        import pandas as pd
        from fundcloud.data import DuckDB

        idx = pd.bdate_range("2024-01-01", periods=30)
        df = pd.DataFrame({
            ("close",  "SPY"): np.random.default_rng(0).normal(400, 2, 30),
            ("volume", "SPY"): np.ones(30) * 1e6,
        }, index=idx)

        store = DuckDB(str(tmp_path / "warehouse.duckdb"))
        store.write("us_eq", df)
        assert store.exists("us_eq")
        assert "us_eq" in store
        loaded = store.read("us_eq")
        assert loaded.shape == df.shape


# ---------------------------------------------------------------------------
# guides/plots/summary.md + themes.md
# ---------------------------------------------------------------------------

class TestPlots:
    def test_plots_summary(self, returns_series, spy_returns):
        from fundcloud import plots
        fig = plots.summary(returns_series)
        assert fig is not None
        fig2 = plots.summary(returns_series, benchmark=spy_returns)
        assert fig2 is not None

    def test_theme_switching(self, returns_series):
        import fundcloud as fc
        for theme in ("dark", "white", "ggplot2", "seaborn"):
            fc.set_theme(theme)
            fc.plots.cumulative(returns_series)
        fc.set_theme("white")  # reset

    def test_custom_plotly_template(self, returns_series):
        import fundcloud as fc
        import plotly.graph_objects as go
        import plotly.io as pio

        pio.templates["test-brand"] = go.layout.Template(
            layout={"colorway": ["#003a70", "#e6a817"]}
        )
        fc.set_theme("test-brand")
        fc.plots.summary(returns_series)
        fc.set_theme("white")  # reset


# ---------------------------------------------------------------------------
# guides/accelerators/rust-kernels.md
# ---------------------------------------------------------------------------

class TestKernels:
    def test_has_rust_flag(self):
        from fundcloud import kernels
        assert isinstance(kernels.HAS_RUST, bool)
        v = kernels.kernel_version()
        assert isinstance(v, str)


# ---------------------------------------------------------------------------
# guides/reports/tearsheets.md
# ---------------------------------------------------------------------------

class TestTearsheets:
    def test_render_html(self, returns_series, tmp_path):
        from fundcloud.portfolio import Portfolio
        from fundcloud.reports import Tearsheet

        pf = Portfolio(returns=returns_series, name="test")
        out = tmp_path / "test.html"
        Tearsheet(pf, title="Test").render_html(str(out))
        assert out.exists()

    def test_render_excel(self, returns_series, tmp_path):
        from fundcloud.portfolio import Portfolio
        from fundcloud.reports import Tearsheet

        pf = Portfolio(returns=returns_series, name="test")
        out = tmp_path / "test.xlsx"
        Tearsheet(pf, title="Test").render_excel(str(out))
        assert out.exists()
