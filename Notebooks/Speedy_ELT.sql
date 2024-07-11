-- Databricks notebook source
-- select * from digan.speedy_motors.car_reviews__c

-- COMMAND ----------

-- select * from digan.speedy_motors.cars__c

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Bronze Table - Creating cars raw table**

-- COMMAND ----------

CREATE STREAMING LIVE TABLE cars_raw
COMMENT "The raw car data, ingested from Salesforce."
TBLPROPERTIES ("myCompanyPipeline.quality" = "bronze")
AS
SELECT * FROM stream(digan.speedy_motors.cars__c)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Bronze Table - Creating car reviews raw table**

-- COMMAND ----------

CREATE STREAMING LIVE TABLE car_review_raw
COMMENT "The raw car review data, ingested from Salesforce."
TBLPROPERTIES ("myCompanyPipeline.quality" = "bronze")
AS
SELECT * FROM stream(digan.speedy_motors.car_reviews__c)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Silver Table - Full view of car and review data**

-- COMMAND ----------

CREATE STREAMING LIVE TABLE car_360(
  CONSTRAINT valid_review EXPECT (car_review IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_rating EXPECT (car_rating IS NOT NULL) ON VIOLATION DROP ROW
)
COMMENT "Cleaned car reviews joined with car Salesforce table."
TBLPROPERTIES ("myCompanyPipeline.quality" = "silver")
AS
SELECT 
  reviews.Id as Id,
  cars.Name__c as car_name, 
  cars.MSRP_c__c as car_MSRP,
  cars.Current_Inventory_c__c as car_current_inventory,
  cars.Car_Type_c__c as car_type,
  cars.Primary_Competitor_c__c as car_primary_competitor,
  cars.Color_c__c as car_color,
  cars.Seats_c__c as car_seats,
  cars.Fuel_Type_c__c as car_fuel_type,
  cars.MPG_c__c as car_MPG,
  cars.Drive_Train_Type_c__c as car_drive_train_type,
  cars.Towing_Capacity_c__c as car_towing_capacity,
  reviews.Rating__c as car_rating, 
  reviews.Review__c as car_review
FROM STREAM(live.cars_raw) as cars
JOIN STREAM(live.car_review_raw) as reviews
ON cars.Name__c = reviews.Car_Name__c


-- COMMAND ----------

-- use digan.speedy_motors;
-- SELECT 
--   cars.Name__c as car_name, 
--   cars.MSRP_c__c as car_MSRP,
--   cars.Current_Inventory_c__c as car_current_inventory,
--   cars.Car_Type_c__c as car_type,
--   cars.Primary_Competitor_c__c as car_primary_competitor,
--   cars.Color_c__c as car_color,
--   cars.Seats_c__c as car_seats,
--   cars.Fuel_Type_c__c as car_fuel_type,
--   cars.MPG_c__c as car_MPG,
--   cars.Drive_Train_Type_c__c as car_drive_train_type,
--   cars.Towing_Capacity_c__c as car_towing_capacity,
--   reviews.Rating__c as car_rating, 
--   reviews.Review__c as car_review
-- FROM cars__c as cars
-- JOIN car_reviews__c as reviews
-- ON cars.Name__c = reviews.Car_Name__c;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Gold Table - Joining reviews with car data**

-- COMMAND ----------

CREATE MATERIALIZED VIEW agg_cars
COMMENT "Cleaned car reviews joined with car Salesforce table."
AS
SELECT 
  car_name, 
  round(avg(car_rating),2) as average_review_score, 
  array_join(collect_set(car_review), ', ') as concat_reviews
FROM live.car_360
group by car_name
 


-- COMMAND ----------

-- SELECT 
--   cars.Name__c as car_name, 
--   round(avg(reviews.Rating__c),2) as average_review_score, 
--   array_join(collect_set(reviews.Review__c), ', ') as concat_reviews
-- FROM cars__c as cars
-- JOIN car_reviews__c as reviews
-- ON cars.Name__c = reviews.Car_Name__c
-- GROUP BY cars.Name__c
