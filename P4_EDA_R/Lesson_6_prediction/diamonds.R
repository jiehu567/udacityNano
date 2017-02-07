# Let's start by examining two variables in the data set.
# The scatterplot is a powerful tool to help you understand
# the relationship between two continuous variables.

# We can quickly see if the relationship is linear or not.
# In this case, we can use a variety of diamond
# characteristics to help us figure out whether
# the price advertised for any given diamond is 
# reasonable or a rip-off.

# Let's consider the price of a diamond and it's carat weight.
# Create a scatterplot of price (y) vs carat weight (x).

# Limit the x-axis and y-axis to omit the top 1% of values.

library(ggplot2)
data("diamonds")
quantile(diamonds$price, 0.99)
quantile(diamonds$carat, 0.99)

ggplot(diamonds, aes(carat, price)) + 
  geom_point() +
  xlim(0,2.18) +
  ylim(0, 17378)


# packages

library(GGally)  # scatter matrix
library(scales)
library(memisc)  # summarize regression
library(lattice)
library(MASS)
library(car)     # recode variable
library(reshape2) # wrangle data
library(dplyr)    # create summary and transformation

# sample 10,000 diamonds
set.seed(20022012)
diamond_samp <- diamonds[sample(1:length(diamonds$price),10000),]
ggpairs(diamond_samp, 
        lower = list(continuous = wrap("points", shape = I('.'))), 
        upper = list(combo = wrap("box", outlier.shape = I('.'))))


# Price associates with: carat, cut, color, clarity, table, x, y, z
# Unassociated with: depth, 

# Create two histograms of the price variable
# and place them side by side on one output image.

# We’ve put some code below to get you started.

# The first plot should be a histogram of price
# and the second plot should transform
# the price variable using log10.

# Set appropriate bin widths for each plot.
# ggtitle() will add a title to each histogram.

# You can self-assess your work with the plots
# in the solution video.

# ALTER THE CODE BELOW THIS LINE
# ==============================================

library(gridExtra)

plot1 <- ggplot(diamonds, aes(price)) +
  geom_histogram(binwidth = 100) +
  scale_x_continuous(breaks = seq(0,18823,2000)) +
  ggtitle('Price')

plot2 <- ggplot(diamonds, aes(log10(price))) +
  geom_histogram(binwidth = 0.01) +
  scale_x_continuous(breaks = seq(0,4.28,0.5))+
  ggtitle('Price (log10)')

grid.arrange(plot1, plot2)



# rescale
cuberoot_trans = function() trans_new('cuberoot',
                                      transform = function(x) x^(1/3),
                                      inverse = function(x) x^3)

ggplot(aes(carat, price), data = diamonds) +
  geom_point() +
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2,3),
                      breaks = c(0.2,0.5,1,2,3)) +
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),
                     breaks = c(350,1000,5000,10000,15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat')

# decision to make jitter to avoid over plotting
head(sort(table(diamonds$carat), decreasing = T))

head(sort(table(diamonds$price), decreasing = T))

# the count is too many, very likely to overplot
# so should add transparency and jitter

ggplot(aes(carat, price), data = diamonds) +
  geom_point(alpha = 0.5, size = 0.75,position = position_jitter(h=0)) +
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2,3),
                     breaks = c(0.2,0.5,1,2,3)) +
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),
                     breaks = c(350,1000,5000,10000,15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat') +
  geom_smooth()



# price by clarity

ggplot(aes(carat, price), data = diamonds) +
  geom_point(alpha = 0.5, size = 0.75,position = position_jitter(h=0)) +
  scale_color_brewer(type = 'div',
                     guide = guide_legend(title = 'Clarity', reverse = T)) +
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2,3),
                     breaks = c(0.2,0.5,1,2,3)) +
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),
                     breaks = c(350,1000,5000,10000,15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat') +
  geom_smooth()



ggplot(aes(x = carat, y = price), data = diamonds) + 
  geom_point(alpha = 0.5, size = 1, position = 'jitter', aes(color = clarity)) +
  scale_color_brewer(type = 'div',
                     guide = guide_legend(title = 'Clarity', reverse = T,
                                          override.aes = list(alpha = 1, size = 2))) +  
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2, 3),
                     breaks = c(0.2, 0.5, 1, 2, 3)) + 
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),
                     breaks = c(350, 1000, 5000, 10000, 15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat and Clarity')


# color by cut
ggplot(aes(x = carat, y = price, color = cut), data = diamonds) + 
  geom_point(alpha = 0.5, size = 1, position = 'jitter') +
  scale_color_brewer(type = 'div',
                     guide = guide_legend(title = 'Clarity', reverse = T,
                                          override.aes = list(alpha = 1, size = 2))) +  
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2, 3),
                     breaks = c(0.2, 0.5, 1, 2, 3)) + 
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),
                     breaks = c(350, 1000, 5000, 10000, 15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat and Cut')

# color by color
ggplot(aes(x = carat, y = price, color = color), data = diamonds) + 
  geom_point(alpha = 0.5, size = 1, position = 'jitter') +
  scale_color_brewer(type = 'div',
                     guide = guide_legend(title = 'Color',
                                          override.aes = list(alpha = 1, size = 2))) +  
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2, 3),
                     breaks = c(0.2, 0.5, 1, 2, 3)) + 
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),
                     breaks = c(350, 1000, 5000, 10000, 15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat and Color')



# model
m1 <- lm(I(log(price)) ~ I(carat^(1/3)), data = diamonds) 
# I: transform before regression
m2 <- update(m1, ~.+carat)
m3 <- update(m2, ~.+ cut)
m4 <- update(m3, ~.+color)
m5 <- update(m4, ~.+clarity)
mtable(m1, m2, m3, m4, m5, sdigits = 3)


# big diamond dataset quiz
load("BigDiamonds.rda")
set.seed(123)
diamondsbig_sample <- diamondsbig[sample(1:length(diamondsbig$carat),10000),]
diamondsbig_sample <- diamondsbig_sample[,-8]
ggpairs(diamondsbig_sample, 
        lower = list(continuous = wrap("points", shape = I('.'))), 
        upper = list(combo = wrap("box", outlier.shape = I('.'))))

m1 <- lm(I(log(price)) ~ I(carat^(1/3)), data = diamondsbig_sample) 
m2 <- update(m1, ~.+carat)
m3 <- update(m2, ~.+ cut)
m4 <- update(m3, ~.+color)
m5 <- update(m4, ~.+clarity)
mtable(m1, m2, m3, m4, m5, sdigits = 3)

# predict

thisDiamond <- data.frame(carat = 1.00, cut = "V.Good", color = "I", clarity = "VS1")
modelEstimate <- predict(m5, newdata = thisDiamond, interval = "prediction", level = .95)
exp(modelEstimate)


dat = data.frame(m4$model, m4$residuals) 

with(dat, sd(m4.residuals)) 

with(subset(dat, carat > .9 & carat < 1.1), sd(m4.residuals)) 

dat$resid <- as.numeric(dat$m4.residuals)
ggplot(aes(y = resid, x = round(carat, 2)), data = dat) + 
  geom_line(stat = "summary", fun.y = sd) 




