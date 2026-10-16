# Chart 4: Review Volume Forecast
library(tidyverse)
library(forecast)

# read data
monthly <- read_csv("data/clean/reviews_monthly.csv")

# transform Time series
ts_reviews <- ts(monthly$review_count, 
                 start = c(2019, 1), 
                 frequency = 12)

# run auto.arima
fit <- auto.arima(ts_reviews)

# the next 6 months
fc <- forecast(fit, h = 6)

# forecast result
forecast_df <- data.frame(
  year_month = seq(as.Date("2025-10-01"), by = "month", length.out = 6),
  point_forecast = as.numeric(fc$mean),
  lo_80 = as.numeric(fc$lower[,1]),
  hi_80 = as.numeric(fc$upper[,1]),
  lo_95 = as.numeric(fc$lower[,2]),
  hi_95 = as.numeric(fc$upper[,2]),
  type = "Forecast"
)

# history data
history_df <- monthly %>%
  mutate(
    point_forecast = review_count,
    lo_80 = NA, hi_80 = NA, lo_95 = NA, hi_95 = NA,
    type = "Actual"
  ) %>%
  rename(year_month = year_month) %>%
  select(year_month, point_forecast, lo_80, hi_80, lo_95, hi_95, type)

last_actual <- history_df %>% 
  slice_tail(n = 1) %>%
  mutate(type = "Forecast") 

chart4_df <- bind_rows(history_df, last_actual, forecast_df)

write_csv(chart4_df, "data/clean/chart4_forecast.csv")
print(fc)

