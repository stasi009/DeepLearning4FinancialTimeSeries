import yfinance as yf
import os
import argparse

PROXY = "http://127.0.0.1:4780"
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPs_PROXY"] = PROXY


def download_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, period="1d", proxy=PROXY)
    if df.shape[0] == 0:
        print(f"Failed to download symbol=[{symbol}] from {start_date}~{end_date}")
        return

    real_startdate = df.index[0]
    real_enddate = df.index[-1]

    filename = symbol[1:] if symbol[0] == "^" else symbol
    filename = f"datas/{filename}.csv"
    df.to_csv(filename)

    print(f"{real_startdate}~{real_enddate} {df.shape} Data saved to '{filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--symbol")
    parser.add_argument("-b", "--begin", default="2000-01-03")
    parser.add_argument("-e", "--end", default="2023-07-30")
    args = parser.parse_args()
    download_data(symbol=args.symbol.upper(), start_date=args.begin, end_date=args.end)
