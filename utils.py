from datetime import datetime, timedelta
from urllib import request
import json, numpy
import pickle
import pandas as pd

def offset_value(test_start_date, test, predictions_df):
    temp_date = test_start_date
    average_last_5_days_test = 0
    average_upcoming_5_days_predicted = 0
    total_days = 10
    for i in range(total_days):
        average_last_5_days_test += test.loc[temp_date, 'prices']
        temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
        difference = temp_date + timedelta(days=1)
        temp_date = difference.strftime('%Y-%m-%d')
    average_last_5_days_test = average_last_5_days_test / total_days

    temp_date = test_start_date
    for i in range(total_days):
        average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
        temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
        difference = temp_date + timedelta(days=1)
        temp_date = difference.strftime('%Y-%m-%d')
    average_upcoming_5_days_predicted = average_upcoming_5_days_predicted / total_days
    difference_test_predicted_prices = average_last_5_days_test - average_upcoming_5_days_predicted
    return difference_test_predicted_prices

def createtable(db):
    db.query("""CREATE TABLE etherium (
        tid bigint NOT NULL,
        timestampms bigint,
        price double precision,
        amount double precision,
        type character varying(255),
        day integer NOT NULL,
        month integer NOT NULL,
        year integer NOT NULL,
        datetime date NOT NULL
    );"""
    )

def nextBatch(url, db):
    response = request.urlopen(url)
    print("response")
    batch_of_trades = json.loads(response.read())
    # print batch_of_trades, len(batch_of_trades)
    lastTradeTimesec = -1;
    for trade in batch_of_trades:
        lastTradeTimesec = trade['timestamp']
        # print lastTradeTimesec
        date = datetime.fromtimestamp(lastTradeTimesec)
        trade['day']        = date.day
        trade['month']      = date.month
        trade['year']       = date.year
        trade['datetime']   = date.date()
        db.insert('etherium',trade)

    print(lastTradeTimesec)
    return lastTradeTimesec

def dumpDailyAverageToPickle(db):
    allRows = db.query("select * from etherium").getresult()
    dailyTrades = {}
    for row in allRows:
        if (str(row[8]) not in dailyTrades):
            dailyTrades[str(row[8])] = [row[2]]
        else:
            dailyTrades[str(row[8])].append(row[2])

    dates = []
    highs = []
    lows = []
    avgs = []

    for date, trades in sorted(dailyTrades.items(), key=lambda kv: (kv[0],kv[1])):
        standardDeviation = numpy.std(trades)
        mean = numpy.mean(trades)
        # indexValues = [date, mean + standardDeviation, mean - standardDeviation, mean]
        dates.append(date)
        highs.append(mean + standardDeviation)
        lows.append(mean - standardDeviation)
        avgs.append(mean)
    d = {"Date" : dates, "High" : highs, "Low" : lows, "avg" : avgs}
    df = pd.DataFrame(data=d)
    df.to_csv("data/etherium_indices_data.csv", sep=',', index=False)
    return df
