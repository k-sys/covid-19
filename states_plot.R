#adopted from https://github.com/hafen/geofacet/issues/43
library(tidyverse)
library(geofacet)
library(ggthemes)
options(scipen = 99)


us_state_grid4 <- data.frame(
  row = c(1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8),
  col = c(11, 1, 11, 10, 6, 2, 6, 10, 7, 5, 3, 9, 4, 1, 10, 6, 5, 2, 9, 7, 1, 8, 11, 4, 3, 1, 3, 10, 6, 9, 5, 4, 2, 8, 7, 2, 5, 4, 3, 7, 8, 6, 9, 7, 8, 5, 6, 4, 9, 1, 4),
  code = c("ME", "AK", "NH", "VT", "WI", "ID", "IL", "MA", "MI", "MN", "MT", "NY", "ND", "WA", "CT", "IN", "IA", "NV", "NJ", "OH", "OR", "PA", "RI", "SD", "WY", "CA", "CO", "DE", "KY", "MD", "MO", "NE", "UT", "VA", "WV", "AZ", "AR", "KS", "NM", "NC", "SC", "TN", "DC", "AL", "GA", "LA", "MS", "OK", "FL", "HI", "TX"),
  name = c("Maine", "Alaska", "New Hampshire", "Vermont", "Wisconsin", "Idaho", "Illinois", "Massachusetts", "Michigan", "Minnesota", "Montana", "New York", "North Dakota", "Washington", "Connecticut", "Indiana", "Iowa", "Nevada", "New Jersey", "Ohio", "Oregon", "Pennsylvania", "Rhode Island", "South Dakota", "Wyoming", "California", "Colorado", "Delaware", "Kentucky", "Maryland", "Missouri", "Nebraska", "Utah", "Virginia", "West Virginia", "Arizona", "Arkansas", "Kansas", "New Mexico", "North Carolina", "South Carolina", "Tennessee", "District of Columbia", "Alabama", "Georgia", "Louisiana", "Mississippi", "Oklahoma", "Florida", "Hawaii", "Texas"),
  stringsAsFactors = FALSE
)

rt <- read.csv("data/rt.csv")
rt$date = as.Date(rt$date)
rt %>% ggplot(aes(date, ML)) +geom_line(aes(group=1),size = 1)  +geom_hline(yintercept = 1,color = "red",size = 1,linetype="dashed")+
  facet_geo(~state, grid = us_state_grid4) +
  scale_fill_identity() +scale_x_date(breaks = as.Date(c("2020-02-19","2020-04-01")))+
  theme_fivethirtyeight() + 
  theme(legend.position = "top",
        legend.margin = margin(b = -1, unit = "cm")) + 
  labs(
    x = " ",
    y = "Day",
    title = "Temporal Trend for COVID-19 Effective Reproduction Number"
  )

ggsave("rt.png", height = 10, width = 17)
