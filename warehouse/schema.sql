-- ==============================================================================
-- Adult Income Analysis & Prediction: Data Warehouse Schema
-- Multidimensional Star Schema Definition (SQLite Compatible)
-- ==============================================================================

DROP TABLE IF EXISTS FactIncome;
DROP TABLE IF EXISTS DimDemographics;
DROP TABLE IF EXISTS DimEducation;
DROP TABLE IF EXISTS DimOccupation;
DROP TABLE IF EXISTS DimGeography;

-- 1. DimDemographics: Demographic profile of individuals
CREATE TABLE DimDemographics (
    demo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    age_band TEXT NOT NULL,           -- '<30', '30-45', '46-60', '>60'
    sex TEXT NOT NULL,                -- 'Male', 'Female'
    race TEXT NOT NULL,               -- 'White', 'Black', 'Asian-Pac-Islander', etc.
    marital_status TEXT NOT NULL      -- 'Married-civ-spouse', 'Never-married', etc.
);

-- 2. DimEducation: Education levels and hierarchies
CREATE TABLE DimEducation (
    edu_id INTEGER PRIMARY KEY AUTOINCREMENT,
    education_level TEXT NOT NULL,    -- 'Bachelors', 'HS-grad', 'Masters', etc.
    education_num INTEGER NOT NULL    -- Numeric scale 1 to 16
);

-- 3. DimOccupation: Professional profile and employment type
CREATE TABLE DimOccupation (
    occ_id INTEGER PRIMARY KEY AUTOINCREMENT,
    workclass TEXT NOT NULL,          -- 'Private', 'Self-emp-inc', 'Federal-gov', etc.
    occupation TEXT NOT NULL          -- 'Exec-managerial', 'Prof-specialty', etc.
);

-- 4. DimGeography: Origin and regional classification
CREATE TABLE DimGeography (
    geo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    native_country TEXT NOT NULL      -- 'United-States', 'Mexico', 'India', etc.
);

-- 5. FactIncome: Central fact table recording socioeconomic observations
CREATE TABLE FactIncome (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    demo_id INTEGER NOT NULL REFERENCES DimDemographics(demo_id),
    edu_id INTEGER NOT NULL REFERENCES DimEducation(edu_id),
    occ_id INTEGER NOT NULL REFERENCES DimOccupation(occ_id),
    geo_id INTEGER NOT NULL REFERENCES DimGeography(geo_id),
    hours_per_week INTEGER NOT NULL DEFAULT 40,
    capital_gain INTEGER NOT NULL DEFAULT 0,
    capital_loss INTEGER NOT NULL DEFAULT 0,
    income_flag INTEGER NOT NULL      -- 1 if '>50K', 0 if '<=50K'
);

-- Indexes for OLAP query performance
CREATE INDEX idx_fact_demographics ON FactIncome(demo_id);
CREATE INDEX idx_fact_education ON FactIncome(edu_id);
CREATE INDEX idx_fact_occupation ON FactIncome(occ_id);
CREATE INDEX idx_fact_geography ON FactIncome(geo_id);
CREATE INDEX idx_fact_income_flag ON FactIncome(income_flag);
