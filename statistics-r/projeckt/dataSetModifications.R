library(tidyverse)

vehicles = read.csv("vehicles.csv")
fuel_sample = vehicles[sample(nrow(vehicles), 2000),]
fuel_sample

colnames(fuel_sample)[colnames(fuel_sample) == "city08"] <- "city_mpg"
colnames(fuel_sample)[colnames(fuel_sample) == "highway08"] <- "highway_mpg"
colnames(fuel_sample)[colnames(fuel_sample) == "comb08"] <- "comb_mpg"

write.csv(fuel_sample, "fuel_sample.csv", row.names = FALSE)

shapiro.test(sample(fuel_sample$city_mpg, 20))


fuel_sample$fuel_type <- factor(fuel_sample$fuel_type, 
                              levels = c("Electric", "Hybrid", "Gasoline", "Diesel"), 
                              ordered = TRUE)

colnames(fuel_sample)
table(fuel_sample$fuelType1)
is.factor(fuel_sample$VClass)
unique(fuel_sample$VClass)




fuel_data <- read.csv("fuel_sample.csv", stringsAsFactors = TRUE)
nrow(unique(fuel_data)) == nrow(fuel_data)


is.factor(fuel_data$VClass)
is.ordered(fuel_data$VClass)
fuel_data <- fuel_data %>%
  mutate(across(c(VClass, make, model, fuelType, fuelType1, drive, trany, atvType, guzzler, mpgData), as.factor))
is.ordered(fuel_data$mpgData)
unique(fuel_data$mpgData)
