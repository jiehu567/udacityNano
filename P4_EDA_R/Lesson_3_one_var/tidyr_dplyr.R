# tidyr and dplyr

data("iris")
library(dplyr)
library(tidyr)

# convert data to tbl class

dplyr::tbl_df(iris)
head(iris)


# information dense summary of tbl data
dplyr::glimpse(iris)
View(iris)

# dplyr: %>% passes object on left side as argument of function on right side
# e.g.:
x %>% f(y) # same as f(x, y)

y %<% f(x,.,z) # same as f(x,y,z)

# pipelines:

iris %>%
  group_by(Species) %>%
  summarise(avg = mean(Sepal.Width)) %>%
  arrange(avg) # sort

# ? gather columns into rows, 2 to 4 columns into key - value pairs
tidyr::gather(iris, "t3", "t4", 2:4)

stocks <- data_frame(
  time = as.Date('2009-01-01') + 0:9,
  X = rnorm(10, 0, 1),
  Y = rnorm(10, 0, 2),
  Z = rnorm(10, 0, 4)
)

gather(stocks, stock, price, -time)
stocks %>% gather(stock, price, -time)


# seperate and unite
sep_stocks <- separate(stocks, time, c('y', 'm', 'd'))
sep_stocks
unite(sep_stocks, time2, y,m,d, sep = ',')

# spread rows into columns
spread(stocks, time, X)

# combine vectors into df
data_frame(a = 1:3, b=4:6)

# order by values of a column
arrange(stocks, X)

# rename column name
rename(stocks, X_t = X)
