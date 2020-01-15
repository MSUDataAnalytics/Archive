generate_data = function(int = 1,
                         slope = 2,
                         sigma = 5,
                         n_obs = 15,
                         x_min = 0,
                         x_max = 10) {
  x = seq(x_min, x_max, length.out = n_obs)
  y = int + slope * x + rnorm(n_obs, 0, sigma)
  fit = lm(y ~ x)
  y_hat = fitted(fit)
  y_bar = rep(mean(y), n_obs)
  data.frame(x, y, y_hat, y_bar)
}

plot_total_dev = function(reg_data) {
  plot(reg_data$x, reg_data$y, main = "SST (Sum of Squares Total)",
       xlab = "x", ylab = "y", pch = 20, cex = 2, col = "grey")
  arrows(reg_data$x, reg_data$y_bar,
         reg_data$x, reg_data$y,
         col = 'grey', lwd = 1, lty = 3, length = 0.2, angle = 20)
  abline(h = mean(reg_data$y), lwd = 2,col = "grey")
  # abline(lm(y ~ x, data = reg_data), lwd = 2, col = "grey")
}

plot_total_dev_prop = function(reg_data) {
  plot(reg_data$x, reg_data$y, main = "SST (Sum of Squares Total)",
       xlab = "x", ylab = "y", pch = 20, cex = 2, col = "grey")
  arrows(reg_data$x, reg_data$y_bar,
         reg_data$x, reg_data$y_hat,
         col = 'darkorange', lwd = 1, length = 0.2, angle = 20)
  arrows(reg_data$x, reg_data$y_hat,
         reg_data$x, reg_data$y,
         col = 'dodgerblue', lwd = 1, lty = 2, length = 0.2, angle = 20)
  abline(h = mean(reg_data$y), lwd = 2,col = "grey")
  abline(lm(y ~ x, data = reg_data), lwd = 2, col = "grey")
}

plot_unexp_dev = function(reg_data) {
  plot(reg_data$x, reg_data$y, main = "SSE (Sum of Squares Error)",
       xlab = "x", ylab = "y", pch = 20, cex = 2, col = "grey")
  arrows(reg_data$x, reg_data$y_hat,
         reg_data$x, reg_data$y,
         col = 'dodgerblue', lwd = 1, lty = 2, length = 0.2, angle = 20)
  abline(lm(y ~ x, data = reg_data), lwd = 2, col = "grey")
}

plot_exp_dev = function(reg_data) {
  plot(reg_data$x, reg_data$y, main = "SSReg (Sum of Squares Regression)",
  xlab = "x", ylab = "y", pch = 20, cex = 2, col = "grey")
  arrows(reg_data$x, reg_data$y_bar,
         reg_data$x, reg_data$y_hat,
         col = 'darkorange', lwd = 1, length = 0.2, angle = 20)
  abline(lm(y ~ x, data = reg_data), lwd = 2, col = "grey")
  abline(h = mean(reg_data$y), col = "grey")
}

par(mfrow = c(2, 2))
plot_exp_dev(plot_data)
plot_unexp_dev(plot_data)
plot_total_dev(plot_data)
plot_total_dev_prop(plot_data)

set.seed(1)
plot_data = generate_data(slope = -1.5, sigma = 10)
