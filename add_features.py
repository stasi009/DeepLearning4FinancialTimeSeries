import numpy as np
import talib
from task_manager import Task, STAGE
import argparse


class JobAddFeatures:
    def __init__(self, task: Task) -> None:
        self.task = task
        self.df = Task.load_ohlcv(task.config["symbol"])

    def __add_label(self):
        # target to fit, tomorrow minus today
        labels = (self.df["Close"].shift(-1) > 1.0025 * self.df["Close"]).astype(int)
        # self.df["Close"].shift(-1) has its last value as Nan, then '>' return False
        # which is incorrect as last example's label
        # set its value to NAN, then this example will be dropped later
        labels.iloc[-1] = np.nan
        self.df["forward_move"] = labels

        self.df["forward_change"] = self.df["Close"].shift(-1) - self.df["Close"]

    def __add_price_range(self):
        self.df["c-o"] = self.df["Close"] - self.df["Open"]
        self.df["h-l"] = self.df["High"] - self.df["Low"]

    def __add_rolling_return(self, windows):
        # today minus yesterday
        self.df["return_b1d"] = np.log(self.df["Close"]).diff()

        for window in windows:
            assert window != 1
            self.df[f"return_b{window}d"] = self.df["return_b1d"].rolling(window).sum()
            self.df[f"return_b{window}d_std"] = self.df["return_b1d"].rolling(window).std()

    def __add_rolling_volume(self, windows):
        volumes = self.df["Volume"]
        for window in windows:
            assert window != 1
            self.df[f"vlm_{window}d_avg"] = volumes.rolling(window).mean()
            self.df[f"vlm_{window}d_std"] = volumes.rolling(window).std()

    def __add_rolling_price(self, windows):
        for window in windows:
            self.df[f"price_{window}d_sma"] = self.df["Close"].rolling(window).mean()
            self.df[f"price_{window}d_std"] = self.df["Close"].rolling(window).std()
            self.df[f"price_{window}d_ewma"] = self.df["Close"].ewm(span=window, min_periods=window).mean()

    def __add_ma_price_cross(self, short_win, long_win):
        short_ma, long_ma = self.df[f"price_{short_win}d_sma"], self.df[f"price_{long_win}d_sma"]

        cross = short_ma > long_ma
        prev_cross = cross.shift(1).fillna(False)

        self.df["is_pricema_gold"] = ((~prev_cross) & cross).astype(int)
        self.df["is_pricema_dead"] = (prev_cross & (~cross)).astype(int)

    def __add_rsi(self, windows):
        for window in windows:
            rsi = talib.RSI(self.df["Close"], window)
            self.df[f"rsi_{window}d"] = rsi

            over_bought = rsi >= 80
            over_sold = rsi <= 20

            # yesterday is over_sold, today is not, buy signal
            self.df[f"is_rsi{window}_gold"] = (over_sold.shift(1) & (~over_sold)).astype(int)
            # yesterday is over_bought, today is not, sell signal
            self.df[f"is_rsi{window}_dead"] = (over_bought.shift(1) & (~over_bought)).astype(int)

    def __add_macd(self):
        # use default parameters, fastperiod=12, slowperiod=26, signalperiod=9
        macd, macdsignal, macdhist = talib.MACD(self.df["Close"])
        self.df["macd"] = macd
        self.df["macdsignal"] = macdsignal
        # macdhist = macd - macdsignal, which makes 'macd' redundant
        self.df["macdhist"] = macdhist

        # yesterday's hist become today's prev_hist
        prev_hist = self.df["macdhist"].shift(1)
        self.df["is_macd_gold"] = ((prev_hist <= 0) & (self.df["macdhist"] > 0)).astype(int)
        self.df["is_macd_dead"] = ((prev_hist >= 0) & (self.df["macdhist"] < 0)).astype(int)

    def __add_bbands(self, windows):
        for window in windows:
            boll_up, boll_middle, boll_low = talib.BBANDS(self.df["Close"], timeperiod=window)

            up_cross = self.df["Close"] > boll_up
            prev_up_cross = up_cross.shift(1).fillna(False)

            low_cross = self.df["Close"] < boll_low
            prev_low_cross = low_cross.shift(1).fillna(False)

            self.df[f"is_boll{window}_gold"] = ((~prev_up_cross) & up_cross).astype(int)
            self.df[f"is_boll{window}_dead"] = ((~prev_low_cross) & low_cross).astype(int)

    def __add_atr(self, windows):
        for window in windows:
            atr = talib.ATR(
                high=self.df["High"],
                low=self.df["Low"],
                close=self.df["Close"],
                timeperiod=window,
            )
            self.df[f"atr_{window}d"] = atr

    def __add_chaikin(self):
        """Chaikin A/D Line"""
        chaikin = talib.AD(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"],
            volume=self.df["Volume"],
        )
        self.df["chaikin"] = chaikin

    def __add_obv(self):
        self.df["obv"] = talib.OBV(real=self.df["Close"], volume=self.df["Volume"])

    def main(self):
        self.__add_label()
        self.__add_price_range()
        self.__add_macd()
        self.__add_chaikin()
        self.__add_obv()

        # if window size=N, [:N] will be NaN, but aligned with datetime index
        windows = [5, 10, 20]
        self.__add_rolling_return(windows)
        self.__add_rolling_volume(windows)
        self.__add_rolling_price(windows)
        self.__add_ma_price_cross(short_win=10, long_win=20)
        self.__add_rsi(windows)
        self.__add_atr(windows)
        self.__add_bbands(windows)

        self.df.dropna(inplace=True)
        self.task.save_raw_features(self.df)

        self.task.description = f"add new features for {self.task.config['symbol']}, shape={self.df.shape}"
        self.task.outputs["features"] = list(self.df.columns)  # make it JSON compatible
        return self.task.done()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--symbol")
    args = parser.parse_args()

    task = Task(
        stage=STAGE.ADD_FEATURE,
        config={"symbol": args.symbol},
    )
    JobAddFeatures(task).main()
