import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover


class MyStrategy(Strategy):
    sig_class = None
    pos_class = None
    config = None

    def init(self):
        super().init()
        self.signal_detector = self.sig_class(self)
        self.position_manager = self.pos_class(self)

    def next(self):
        super().next()
        signal = self.signal_detector.next()
        self.position_manager.next(signal)


class StrategyWrapper:
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy
        self._config = strategy.config
        self._data = strategy.data

    def add_indicator(self, name, series, **kwargs):
        indicator = self._strategy.I(lambda: series, name=name, **kwargs)
        setattr(self._strategy, name, indicator)

    def get_indicator(self, name):
        return getattr(self._strategy, name)

    @property
    def position(self):
        return self._strategy.position

    @property
    def trades(self):
        return self._strategy._broker.trades

    def buy(self, size=Strategy._FULL_EQUITY, sl=None, tp=None):
        return self._strategy.buy(size=size, sl=sl, tp=tp)

    def sell(self, size=Strategy._FULL_EQUITY, sl=None, tp=None):
        return self._strategy.sell(size=size, sl=sl, tp=tp)

    def calculate_atrs(self, period):
        """
        Set the lookback period for computing ATR. The default value
        of 100 ensures a _stable_ ATR.
        """
        hi, lo, c_prev = (
            self._data.High,
            self._data.Low,
            pd.Series(self._data.Close).shift(1),
        )
        tr = np.max([hi - lo, (c_prev - hi).abs(), (c_prev - lo).abs()], axis=0)
        return pd.Series(tr).rolling(period).mean().bfill().values

    def close_prev_trades(self, close):
        left_size = 0
        for trade in self.trades:
            if close == "long" and trade.size > 0:
                trade.close()
            elif close == "short" and trade.size < 0:
                trade.close()
            else:
                left_size += trade.size
        return left_size


def get_or_setdefault(d, k, default):
    v = d.get(k)
    if v is not None:
        return v

    d[k] = default
    return default


class ModelSignal(StrategyWrapper):
    def __init__(self, strategy: Strategy) -> None:
        super().__init__(strategy)

        next_positions = []
        signal_strengths = []
        for proba in self._data["pred_up_proba"]:
            position = 0
            strength = 0

            if proba >= self._config["up_threshold"]:
                position = 1
                strength = proba / self._config["up_threshold"]
            elif proba <= self._config["down_threshold"]:
                position = -1
                strength = self._config["down_threshold"] / proba

            next_positions.append(position)
            signal_strengths.append(strength)

        # when predicted probability is just at threshold, the strength should be 'signal_strength_scale'
        signal_strengths = np.asarray(signal_strengths) * self._config.get("signal_strength_scale", 1)

        self.add_indicator(name="pred_proba", series=self._data["pred_up_proba"], plot=True)
        self.add_indicator(name="next_position", series=next_positions, plot=True)
        self.add_indicator(name="signal_strength", series=signal_strengths, plot=True)

    def next(self):
        return self.get_indicator("next_position")[-1]


class SmaSignal(StrategyWrapper):
    def __init__(self, strategy: Strategy) -> None:
        super().__init__(strategy)
        close_prices = pd.Series(self._data.Close)

        short_win = self._config.get("short_win", 10)
        self.add_indicator(
            name="short_sma", series=close_prices.rolling(short_win).mean(), plot=True, overlay=True
        )

        long_win = self._config.get("long_win", 20)
        self.add_indicator(
            name="long_sma", series=close_prices.rolling(long_win).mean(), plot=True, overlay=True
        )

    def next(self):
        short_sma = self.get_indicator("short_sma")
        long_sma = self.get_indicator("long_sma")

        if crossover(short_sma, long_sma):
            return 1

        elif crossover(long_sma, short_sma):
            return -1

        return 0


class SimplePosition(StrategyWrapper):
    def next(self, signal):
        if signal > 0 and self.position.size <= 0:
            self.position.close()
            self.buy()

        elif signal < 0 and self.position.size >= 0:
            self.position.close()
            self.sell()


class DummySignal(StrategyWrapper):
    def next(self):
        """No real signal, work with SellBuyEveryday"""
        return 0


class SellBuyEveryday(StrategyWrapper):
    def next(self, signal):
        self.position.close()
        self.buy()


class AtrPosition(StrategyWrapper):
    def __init__(self, strategy: Strategy) -> None:
        super().__init__(strategy)

        self._atrs = self.calculate_atrs(period=self._config.get("atr_period", 20))
        self._init_cash = self._strategy.equity  # 0-th bar has no trade yet

    def reset_sl_tp(self):
        bar_index = len(self._data) - 1
        atr = self._atrs[bar_index]

        for trade in self.trades:
            lb = trade.entry_price - atr * self._config["n_atr"]
            ub = trade.entry_price + atr * self._config["n_atr"]

            if trade.is_long:
                if self._config["stop_loss"]:
                    sl = lb
                    # stop loss for long trade, need to set price's lower boundary, the higher the tighter, use max
                    trade.sl = max(trade.sl or -np.inf, sl)

                if self._config["take_profit"]:
                    tp = ub
                    # take profit for long trade, need to set price's upper boundary, the lower the tighter, use min
                    trade.tp = min(trade.tp or np.inf, tp)
            else:
                if self._config["stop_loss"]:
                    sl = ub
                    # stop loss for short trade, need to set price's upper boundary, the lower the tighter, use min
                    trade.sl = min(trade.sl or np.inf, sl)

                if self._config["take_profit"]:
                    tp = lb
                    # take profit for short trade, need to set price's lower boundary, the higher the tigher, use max
                    trade.tp = max(trade.tp or -np.inf, tp)

    def next(self, signal):
        atr = self._atrs[len(self._data) - 1]

        if self._config.get("adjust_size_by_proba", False):
            signal_strength = abs(self.get_indicator("signal_strength")[-1])
        else:
            signal_strength = 1

        trade_size = int(
            signal_strength * self._init_cash * self._config["risk_ratio"] / (self._config["n_atr"] * atr)
        )
        if trade_size == 0:  # less than one share
            return

        if signal > 0:
            left_size = self.close_prev_trades("short")
            assert left_size >= 0

            if left_size == 0 or (left_size > 0 and self._config["allow_scale_in"]):
                self.buy(size=trade_size)

        elif signal < 0:
            left_size = self.close_prev_trades("long")
            assert left_size <= 0

            if left_size == 0 or (left_size < 0 and self._config["allow_scale_in"]):
                self.sell(size=trade_size)

        if self._config["stop_loss"] or self._config["take_profit"]:
            self.reset_sl_tp()
