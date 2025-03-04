#Clear the workspace
rm(list=ls())
dev.off()
#specification.r
require(dynlm); require(forecast);require(car);require(urca);require(readxl)

#data("USMacroSW")
import <- read_excel("cpi_dk.xlsx")
data <- import$Forbrugerprisindeks

s <- 12
startyear <- 1980
startperiod <- 1
datastart <- c(startyear,1)
fstart <- c(2025,2)
T <- length(data)
years <- T/s


data <- ts(data, frequency=s, start=datastart)
plot(data)

y <- data
plot(y, main="Consumer Price Index",ylab="Index")
monthplot(y)

par(mfrow=c(2,1), mar=c(3,5,3,3))
acf(y)
pacf(y)
dev.off()

#==============================Test for Unit Root==============================#
plot(diff(y),main="CPI, First difference",ylab="First difference")
#First differences look thoroughly stationary and close to white noise.

plot(diff(y,lag=s))


#Let's inspect ACF and PACF of the series and its differences:
par(mfrow=c(2,1), mar=c(3,5,3,3))
acf(y, lag.max = 3*s,main="IP:MVP"); pacf(y, lag.max = 3*s,main="")
acf(diff(y), lag.max = 3*s,main="IP:MVP, first difference"); pacf(diff(y), lag.max = 3*s,main="")
dev.off()

#Formal test on Unit roots: Dickey-Fuller Test
test <- ur.df(y,type=c("trend"),lags=10,selectlags="AIC")
summary(test)


test <- ur.df(diff(data),type=c("drift"),lags=10,selectlags="AIC")
summary(test)


model.sarima <- auto.arima(y, d=1, D=1, seasonal = TRUE, ic = "aic", 
                           stepwise = FALSE, approximation = FALSE, 
                           trace = TRUE)

par(mfrow=c(2,1), mar=c(3,5,3,3))
acf(model.sarima$residuals,main="ARIMA(3,0,1)(0,1,1)[12]"); pacf(model.sarima$residuals,main="")


fit_auto <- auto.arima(y, seasonal = TRUE)
summary(fit_auto)
checkresiduals(fit_auto)

fit_auto2 <- auto.arima(y, d=1, seasonal = TRUE, ic = "aic", 
                        stepwise = FALSE, approximation = FALSE, 
                        trace = TRUE)
checkresiduals(fit_auto2)



fit_alt <- Arima(y, 
                 order = c(3, 1, 0), 
                 seasonal = list(order = c(2, 1, 0), period = s), 
                 include.drift = TRUE)

summary(fit_alt)
checkresiduals(fit_alt)




fit <- Arima(y, order = c(1, 1, 0), seasonal = list(order = c(0, 1, 1), period = s))
summary(fit)





model.sarima <- auto.arima(y, d=1, D=1, seasonal = TRUE, ic = "aic", 
                           stepwise = FALSE, approximation = FALSE, 
                           trace = TRUE)

fit_model=Arima(y,order=c(3,1,0),seasonal = list(order=c(0,1,2),period=s))
summary(fit_model)
checkresiduals(fit_model)

par(mfrow=c(2,1), mar=c(3,5,3,3))
acf(fit_model$residuals,main="ARIMA(3,0,1)(0,1,1)[12]"); pacf(fit_model$residuals,main="")
dev.off()


#unit root test -- on CPI with trend
summary(ur.df(cpi,type=c("trend"),lags=8))

infl <- diff(log(cpi))
plot(infl, main="Inflation Rate")
#unit root test -- on Inflation without trend
summary(ur.df(infl,type=c("drift"),lags=10,selectlags=c("AIC")))

par(mfrow=c(2,1), mar=c(3,5,3,3))
acf(cpi)
pacf(cpi)

acf(infl)
pacf(infl)


#--------ARIMA model--------#
model.arima <- auto.arima(cpi, d=1, seasonal = FALSE, ic = "aic", 
                          stepwise = FALSE, approximation = FALSE, 
                          trace = TRUE)
model.arima
#Best model: ARIMA(0,1,2) 
#Alt. ARIMA(2,1,0)

e.arima <- model.arima$residuals
FIT.arima <- y-e.arima


h <- s
forecast.arima<-forecast(model.arima, h=h)
forecast.arima

plot(forecast.arima,ylab="Consumer Goods",main="MA(2) in first differences of Consumer Goods")
lines(FIT.arima, col="green")






