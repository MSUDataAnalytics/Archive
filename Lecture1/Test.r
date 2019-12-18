library(IRdisplay)

data <- list(list(x=c(1999, 2000, 2001, 2002), y=c(10, 15, 13, 17), type='scatter'))
figure <- list(data=data)

mimebundle <- list('application/vnd.plotly.v1+json'=figure)
IRdisplay::publish_mimebundle(mimebundle)
