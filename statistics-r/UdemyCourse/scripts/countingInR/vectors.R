prices <- c(100, 110, 200, 250)
discounts <- c(3, 10, 90, 15)

current_prices <- prices - discounts
current_prices

prices - prices * (discounts / 100)

are_jobs_easy <- c(T, T, F, F)
are_jobs_well_paid <- c(T, F, T, F)

are_jobs_easy & are_jobs_well_paid
are_jobs_easy | are_jobs_well_paid

# not working
are_jobs_easy && are_jobs_well_paid
are_jobs_easy || are_jobs_well_paid
