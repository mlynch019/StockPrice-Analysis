CREATE TABLE calculated_static_quantities (
    average_close FLOAT,
    year_max FLOAT,
    year_min FLOAT,
    largest_daily_increase DATE,
    largest_daily_decrease DATE,
    average_daily_change FLOAT

);

-- Cast columns to FLOAT, from CSV of type string
UPDATE all_data
SET Close = CAST(Close AS FLOAT);
UPDATE all_data
SET Open= CAST(Open AS FLOAT);
UPDATE all_data
SET High = CAST(High AS FLOAT);
UPDATE all_data
SET Volume = CAST(Volume AS FLOAT);

-- add column for daily change between open and close prices. Update said column below.
-- ALTER TABLE all_data
-- ADD COLUMN daily_percent_diff FLOAT;

UPDATE all_data
SET daily_percent_diff = ((Close - Open) / Open) * 100;

-- cleared table for queries
DELETE FROM calculated_static_quantities;

-- insert basic metrics. Yearly average, maxiumum, minimum. Highest, lowest, and average daily percent gain/loss.
INSERT INTO calculated_static_quantities (average_close, year_max, year_min, largest_daily_increase, largest_daily_decrease, average_daily_change)
SELECT 
       AVG(CAST(close AS FLOAT)),
       MAX(CAST(close AS FLOAT)),
       MIN(CAST(close AS FLOAT)),
       MAX(daily_percent_diff),
       MIN(daily_percent_diff),
       AVG(daily_percent_diff)
FROM all_data
GROUP BY strftime('%Y', Date); -- strftime (sort by each year: 2021, 2022, 2023)


-- Moving average table to store 5, 20, 50, 150 day moving averages.
CREATE TABLE moving_avg (
    Date DATE,
    five_day_moving_average FLOAT,
    twenty_day_moving_average FLOAT,
    fifty_day_moving_average FLOAT,
    one_fifty_day_moving_average FLOAT
);

DELETE FROM moving_avg;

-- Calculate and insert moving averages with appropriate previous row usage
INSERT INTO moving_avg(Date, five_day_moving_average, twenty_day_moving_average, fifty_day_moving_average, one_fifty_day_moving_average)
SELECT 
    Date,
    AVG(close) OVER (ORDER BY Date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
    AVG(close) OVER (ORDER BY Date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
    AVG(close) OVER (ORDER BY Date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),
    AVG(close) OVER (ORDER BY Date ROWS BETWEEN 149 PRECEDING AND CURRENT ROW)
FROM all_data;



--RSI for each year
-- prepare to calculate RSI by adding column for daily gain/loss
ALTER TABLE all_data
ADD COLUMN abs_difference FLOAT;

UPDATE all_data
SET abs_difference = (Close - Open);

ALTER TABLE all_data
ADD COLUMN gain;

ALTER TABLE all_data
ADD COLUMN loss;

-- Determine if each day was gain or loss. Separate to average when calculating Relative Strength.
UPDATE all_data
SET gain = 
    CASE
        WHEN abs_difference >= 0 THEN abs_difference
        ELSE NULL
    END;

UPDATE all_data
SET loss = 
    CASE
        WHEN abs_difference <= 0 THEN abs_difference
        ELSE NULL
    END;

ALTER TABLE all_data
ADD COLUMN avg_gain;

ALTER TABLE all_data
ADD COLUMN avg_loss;


-- Calculate average gain/loss from previous 14 days
UPDATE all_data
SET avg_gain = (
    SELECT AVG(sub.gain)
    FROM all_data AS sub
    WHERE sub.Date BETWEEN all_data.Date - 13 AND all_data.Date AND sub.gain IS NOT NULL
);

-- Update avg_loss column
UPDATE all_data
SET avg_loss = (
    SELECT -1 * AVG(sub.loss)
    FROM all_data AS sub
    WHERE sub.Date BETWEEN all_data.Date - 13 AND all_data.Date AND sub.loss IS NOT NULL
);

-- create table for RSI. Insert values based on formula
CREATE TABLE rsi (
    Date DATE,
    rsi_by_year FLOAT
);

DELETE FROM rsi;

INSERT INTO rsi (Date, rsi_by_year)
SELECT 
    Date,
    (100 - (100 / (1 + avg_gain / avg_loss)))
FROM all_data;


DELETE FROM rsi
WHERE rsi_by_year IS NULL;


-- Create Table for Moving Average Convergence/Divergence Value
CREATE TABLE macd (
    date DATE,
    close FLOAT,
    ema_12 FLOAT, -- 12 period exponential moving average
    ema_26 FLOAT, -- 26 period exponential moving average
    macd_line FLOAT, -- 12 period minus 26 period
    signal_line FLOAT, -- 9 period exponential moving average
    macd_histogram FLOAT -- MACD histogram values
);

-- Must allocate ema_12 and ema_26 first before using to calculate macd_line, signal_line, and macd_hist
-- Then insert all values into MACD table based on formula
-- Referenced: https://www.investopedia.com/terms/e/ema.asp 
WITH macd_data_inter AS (
    SELECT
        date,
        close,
        (2.0 / 13) * (close - LAG(close, 1) OVER (ORDER BY date)) + LAG(close, 1) OVER (ORDER BY date) AS ema_12,
        (2.0 / 27) * (close - LAG(close, 1) OVER (ORDER BY date)) + LAG(close, 1) OVER (ORDER BY date) AS ema_26
    FROM
        all_data
)
INSERT INTO macd (date, close, ema_12, ema_26, macd_line, signal_line, macd_histogram)
SELECT
    date,
    close,
    ema_12,
    ema_26,
    ema_12 - ema_26,
    (2.0 / 10) * ((ema_12 - ema_26) - LAG((ema_12 - ema_26), 1) OVER (ORDER BY date)) + LAG((ema_12 - ema_26), 1) OVER (ORDER BY date),
    (ema_12 - ema_26) - ((2.0 / 10) * ((ema_12 - ema_26) - LAG((ema_12 - ema_26), 1) OVER (ORDER BY date)) + LAG((ema_12 - ema_26), 1) OVER (ORDER BY date))
FROM
    macd_data_inter;
