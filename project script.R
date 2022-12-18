library(dplyr)
library(randomForest)

train_data <- read.csv('train_final.csv')

test_data <- read.csv('test_final.csv')

train_data_encoded <- read.csv('train_encoded2.csv')

test_data_encoded <- read.csv('test_encoded2.csv')

train_data_encoded <- 
  train_data_encoded %>% 
  mutate(male_flag = case_when(sex == 'Male' ~ 1, sex == 'Female' ~ 0)) %>% 
  mutate(female_flag = case_when(sex == 'Male' ~ 0, sex == 'Female' ~ 1))

train_data_encoded <-
  train_data_encoded %>% 
  dplyr::select(-c(X, workclass, education, education.num, marital.status, occupation, relationship, race, sex, native.country))

test_data_encoded <- 
  test_data_encoded %>% 
  mutate(male_flag = case_when(sex == 'Male' ~ 1, sex == 'Female' ~ 0)) %>% 
  mutate(female_flag = case_when(sex == 'Male' ~ 0, sex == 'Female' ~ 1))

test_data_encoded <-
  test_data_encoded %>% 
  dplyr::select(-c(X, workclass, education, education.num, marital.status, occupation, relationship, race, sex, native.country))

logistic_model <- glm(income.50K ~ ., family = binomial(link = 'logit'), data = train_data_encoded)

poisson_model <- glm(income.50K ~., family = poisson(link = 'log'), data = train_data_encoded)

quasibin_model <- glm(income.50K ~., family = quasibinomial(link = 'logit'), data = train_data_encoded)

gaussian_model <- glm(income.50K ~., family = gaussian(link = 'identity'), data = train_data_encoded)

summary(logistic_model)

summary(poisson_model)



significant_model <- glm(income.50K ~ age + fnlwgt + capital.gain + capital.loss + hours.per.week + education_Prof.school + education_Bachelors + education_Doctorate + education_Masters + education_Some.college + education_11th + education_Assoc.acdm + education_9th + education_Assoc.voc + education_7th.8th + education_10th + education_1st.4th + marital.status_Divorced + marital.status_Never.married + marital.status_Married.spouse.absent + marital.status_Widowed + marital.status_Separated + marital.status_Married.AF.spouse + occupation_Priv.house.serv + relationship_Other.relative + relationship_Not.in.family + relationship_Own.child + race_Black + male_flag + female_flag, family = binomial(link = 'logit'), data = train_data_encoded)

summary(significant_model)

reg_predictions <- predict(logistic_model, test_data_encoded, type = 'response')
test_id <- seq(23842)

reg_predictions_df <- data.frame(test_id, reg_predictions)


sig_predictions <- predict(significant_model, test_data_encoded, type = 'response')

sig_predictions_df <- data.frame(test_id, sig_predictions)

poisson_predictions <- predict(poisson_model, test_data_encoded, type = 'response')

quasibin_predictions <- predict(quasibin_model, test_data_encoded, type = 'response')

quasibin_predictions_df <- data.frame(test_id, quasibin_predictions)

poisson_predictions_df <- data.frame(test_id, poisson_predictions)

gaussian_predictions <- predict(gaussian_model, test_data_encoded, type = 'response')


gaussian_predictions_df <- data.frame(test_id, gaussian_predictions)

colnames(sig_predictions_df) = c("ID", "Prediction")
colnames(reg_predictions_df) = c("ID", "Prediction")
colnames(poisson_predictions_df) = c("ID", "Prediction")
colnames(quasibin_predictions_df) = c("ID", "Prediction")
colnames(gaussian_predictions_df) = c("ID", "Prediction")

write.csv(reg_predictions_df, "reg-predictions-10262022-malefemaleflags2.csv")

write.csv(sig_predictions_df, "sig-predictions-10262022-malefemaleflags2.csv")

write.csv(poisson_predictions_df, "poisson-predictions-10262022.csv")

write.csv(quasibin_predictions_df, "quasibin-predictions-10262022.csv")

write.csv(gaussian_predictions_df, "gaussian-predictions-10262022.csv")

train_reg_predictions <- predict(logistic_model, train_data_encoded, type = 'response')

train_id <- seq(25000)

train_pred_vs_actual <- data.frame(train_id, train_data_encoded$income.50K, train_reg_predictions)

colnames(train_pred_vs_actual) = c('ID', 'Actual', 'Predicted')

train_pred_vs_actual <- 
  train_pred_vs_actual %>% 
  mutate(Error = abs(Actual - Predicted)) %>% 
  arrange(desc(Error))

really_fucked <-
  train_pred_vs_actual %>% 
  filter(Error > 0.5)

really_fucked_ids <- really_fucked$ID

really_fucked_data <- 
  train_data_encoded %>% 
  filter(row_number() %in% really_fucked_ids)


rf_model <- randomForest(income.50K ~., train_data_encoded)

rf_predictions <- predict(rf_model, test_data_encoded, type = 'response')
rf_df <- data.frame(test_id, rf_predictions)

write.csv(rf_df, 'rf-predictions-10262022.csv')


bigass_rf_model <- randomForest(income.50K~., train_data_encoded, ntree=1000)


bigass_rf_predictions <- predict(bigass_rf_model, test_data_encoded, type = 'response')

bigass_rf_df <- data.frame(test_id, bigass_rf_predictions)

write.csv(bigass_rf_df, 'big-rf-predictions-10272022.csv')


