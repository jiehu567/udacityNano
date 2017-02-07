library(ggplot2)

pf <- read.csv('pseudo_facebook.tsv', sep = '\t')
names(pf)

ggplot(aes(x = dob_day), data = pf) + 
  geom_histogram(binwidth = 1) + 
  scale_x_continuous(breaks = 1:31)


# friends count

ggplot(aes(x=friend_count), data = pf) +
  geom_histogram(binwidth = 50) +
  scale_x_continuous(limits = c(0,1000))

# this one doesn't work
ggplot(aes(x = friend_count), data = pf, xlim = c(0,1000)) +
  geom_histogram(binwidth = 50)


# change scale of x by breaks

ggplot(aes(x=friend_count), data = pf) +
  geom_histogram(binwidth = 25) +
  scale_x_continuous(limits = c(0,1000), breaks = seq(0,1000,50))


# 2 cate, male and female, subset by none NA values

ggplot(aes(x=friend_count), data = subset(pf, !is.na(pf$gender))) +
  geom_histogram(binwidth = 25) +
  scale_x_continuous(limits = c(0,1000), breaks = seq(0,1000,50)) +
  facet_grid(.~gender)

# count by gender
by(pf$friend_count, pf$gender, summary)


# measure change
ggplot(aes(x = tenure/365), data = pf) + 
  geom_histogram(binwidth = .25, color = 'black', fill = '#F79420')


# add xlab and ylab
ggplot(aes(x = tenure/365), data = pf) + 
  geom_histogram(binwidth = .5, color = 'black', fill = '#F79420') +
  scale_x_continuous(breaks = seq(1,7,1)) +
  xlab("Number of years using Facebook") +
  ylab("Number of users in sample")


# age hist
ggplot(aes(x = age), data = pf) +
  geom_histogram(binwidth = 10, color = 'black', fill = '#01cab4') +
  scale_x_continuous(breaks = seq(0, 120, 10)) +
  xlab("Age") + ylab("Number of people in sample")


# transform and multiple plot
install.packages("gridExtra")
library(gridExtra)

g = ggplot(aes(x=friend_count), data = pf) + 
      geom_histogram(color = 'black', fill = '#01cab4')

g1 = g + scale_x_continuous() +
  xlab("Count of friends") + ylab("Number of people in sample")

g2 = g +
  scale_x_log10() +
  xlab("Log") + ylab("Number of people in sample")

g3 = g +
  scale_x_sqrt() +
  xlab("Sqrt") + ylab("Number of people in sample")

grid.arrange(g1, g2, g3, ncol=1)


# difference btw log and manual log scale: only scale is different

plot_log <- ggplot(aes(x=log10(friend_count + 1)), data = pf) + 
  geom_histogram(color = 'black', fill = '#01cab4')

plot_log_scale <- g2

grid.arrange(plot_log, plot_log_scale, ncol = 1)

# freq polygon by gender
ggplot(aes(x = friend_count), data = subset(pf, !is.na(pf$gender))) +
  geom_freqpoly(aes(color = gender)) +
  scale_x_log10()


# use proportion:
ggplot(aes(x = friend_count, y = ..count../sum(..count..)), data = subset(pf, !is.na(gender))) + 
  geom_freqpoly(aes(color = gender), binwidth=10) + 
  scale_x_continuous(limits = c(0, 1000), breaks = seq(0, 1000, 50)) + 
  xlab('Friend Count') + 
  ylab('Percentage of users with that friend count')

#  To plot percentages within each group, you can try y = ..density...
ggplot(aes(x = friend_count, y = ..density..), data = subset(pf, !is.na(gender))) + 
  geom_freqpoly(aes(color = gender), binwidth=10) + 
  scale_x_continuous(limits = c(0, 1000), breaks = seq(0, 1000, 50)) + 
  xlab('Friend Count') + 
  ylab('Percentage of users with that friend count')
  
# www_likes by gender

ggplot(aes(x = www_likes), data = subset(pf, !is.na(pf$gender))) +
  geom_freqpoly(aes(color = gender)) +
  scale_x_log10()

by(pf$www_likes, pf$gender, sum)

# boxplot
ggplot(aes(x = gender, y = friend_count), data = subset(pf, !is.na(pf$gender))) +
  geom_boxplot(aes(color = gender)) + ylim(c(0, 1000))

ggplot(aes(x = gender, y = friend_count), data = subset(pf, !is.na(pf$gender))) +
  geom_boxplot(aes(color = gender)) + 
  coord_cartesian(ylim = c(0, 250))

by(pf$friend_count, pf$gender, mean)

# Getting logical

mobile_check_in <- NA
pf$mobile_check_in <- ifelse(pf$mobile_likes >0, 0, 1)
pf$mobile_check_in <- factor(pf$mobile_check_in)
table(pf$mobile_check_in)




