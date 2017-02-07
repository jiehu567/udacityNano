library(ggplot2)
data(diamonds)

summary(diamonds)

dim(diamonds)
str(diamonds)
levels(diamonds$color)

# 2. Create a histogram of the price of all diamonds

ggplot(aes(x= price), data = diamonds)+
  geom_histogram()

# 3. 
summary(diamonds$price)


# 4. Diamonds cost

## less than 500

dim(subset(diamonds, diamonds$price < 500))
dim(subset(diamonds, diamonds$price < 250))
dim(subset(diamonds, diamonds$price >= 15000))

# You can save images by using the ggsave() command.
# ggsave() will save the last plot created.
# For example...
#                  qplot(x = price, data = diamonds)
#                  ggsave('priceHistogram.png')

# ggsave currently recognises the extensions eps/ps, tex (pictex),
# pdf, jpeg, tiff, png, bmp, svg and wmf (windows only).

ggplot(aes(x= log10(price+1)), data = diamonds)+
  geom_histogram(bins = 30, color = 'black', fill = '#01cab4')



# Break out the histogram of diamond prices by cut.

# You should have five histograms in separate
# panels on your resulting plot

library(gridExtra)
g <- ggplot(aes(x= log10(price+1)), data = diamonds)+
  geom_histogram(bins = 30, color = 'black', fill = '#01cab4')

g1 <- g + coord_cartesian(xlim = c(2.5, 3))
g2 <- g + coord_cartesian(xlim = c(3, 3.5))
g3 <- g + coord_cartesian(xlim = c(3.5, 4))
g4 <- g + coord_cartesian(xlim = c(4, 4.5))
g5 <- g + coord_cartesian(xlim = c(4.5, 5))

grid.arrange(g1, g2, g3, g4, g5, ncol=5)


by(diamonds$price, diamonds$cut, min)


by(diamonds$price, diamonds$cut, median)


# Free scale

qplot(x = price, data = diamonds) + facet_wrap(~cut , scales="free")


# Create a histogram of price per carat
# and facet it by cut. You can make adjustments
# to the code from the previous exercise to get
# started.

# Adjust the bin width and transform the scale
# of the x-axis using log10.

ggplot(aes(x = price/carat), data = diamonds) + 
  geom_histogram() +
  scale_x_log10() +
  facet_grid(cut~.) 


cols = c('#ff3f49',
  '#01cab4',
  '#01ffe7',
  '#ffb914',
  '#ffcb4b')

ggplot(aes(y = price, x = cut), data = diamonds) +
  geom_boxplot(color = 'black', fill = cols) + coord_cartesian(ylim = c(0, 8000))


# middle prices
by(diamonds$price, diamonds$color, summary)


# price per caret of different color

cols = c('#ff3f49',
         '#01cab4',
         '#01ffe7',
         '#ffb914',
         '#ffcb4b', 
         '#006775',
         '#f63d47')

ggplot(aes(y = price/carat, x = color), data = diamonds) +
  geom_boxplot(color = 'black', fill = cols) + coord_cartesian(ylim = c(0,8000))



# freq polygon

ggplot(aes(x = carat), data = diamonds) +
  geom_freqpoly(binwidth=.1) + scale_x_continuous(breaks = seq(0,5,0.1))





# beginning work
employRate <- read.csv('employmentRate.csv', header = T, row.names = 1, check.names = F)
head(employRate)

employRate <- employRate[,1:ncol(employRate)-1]  # remove last NA col

# change index by numbers instead of row names
employRate$country <- rownames(employRate)
rownames(employRate) <- 1:nrow(employRate)
head(employRate)

# switch column order
library(dplyr)
employRate <- employRate %>% select(country, everything())
head(employRate)

library(tidyr)

employRate_gathered <- gather(employRate, "year", "employ_Rate", 2:ncol(employRate))

employRate_gathered$country <- as.factor(employRate_gathered$country)
employRate_gathered$year <- as.integer(employRate_gathered$year)

employ_2001 <- subset(employRate_gathered, year=2001)

ggplot(aes(x = employ_Rate), data = employ_2001) +
  geom_histogram(binwidth = 5, color = 'black', fill = 'red')

ggplot(aes(y = employ_Rate, x = country), data = subset(employ_2001, country == c("Afghanistan", "Albania"))) +
  geom_boxplot(color = "black", fill = c("red","green")) +
  ylab("Employ Rate") +
  ggtitle("Employ Rate of Afghanistan and Albania in 2001")


ggplot(employRate_gathered, aes(year, employ_Rate)) +
  geom_point(alpha = 0.1, color = 'orange') +
  scale_x_continuous(breaks = seq(1980, 2010, 5))

