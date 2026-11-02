-- ==============================================================================
-- DMW Activity 2: Adult Census Income Data Warehouse
-- Star Schema Definition (SQLite compatible)
-- ==============================================================================

DROP TABLE IF EXISTS FactIncome;
DROP TABLE IF EXISTS DimDemographics;
DROP TABLE IF EXISTS DimEducation;
DROP TABLE IF EXISTS DimOccupation;
DROP TABLE IF EXISTS DimGeography;

-- 1. DimDemographics: Demographic profile of individuals
CREATE TABLE DimDemographics (
    demographics_id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER NOT NULL,
    age_group TEXT NOT NULL,          -- '<25', '25-39', '40-54', '55-64', '65+'
    sex TEXT NOT NULL,                -- 'Male', 'Female'
    race TEXT NOT NULL,               -- 'White', 'Black', 'Asian-Pac-Islander', etc.
    relationship TEXT NOT NULL,       -- 'Husband', 'Wife', 'Own-child', etc.
    marital_status TEXT NOT NULL      -- 'Married-civ-spouse', 'Never-married', etc.
);

-- 2. DimEducation: Education levels and hierarchies
CREATE TABLE DimEducation (
    education_id INTEGER PRIMARY KEY AUTOINCREMENT,
    education TEXT NOT NULL,          -- 'Bachelors', 'HS-grad', 'Masters', etc.
    education_num INTEGER NOT NULL,   -- Numeric scale 1 to 16
    education_tier TEXT NOT NULL      -- 'Primary', 'Secondary', 'Undergraduate', 'Postgraduate'
);

-- 3. DimOccupation: Professional profile and employment type
CREATE TABLE DimOccupation (
    occupation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    workclass TEXT NOT NULL,          -- 'Private', 'Self-emp-inc', 'Federal-gov', etc.
    occupation TEXT NOT NULL,         -- 'Exec-managerial', 'Prof-specialty', etc.
    job_sector TEXT NOT NULL          -- 'Private', 'Government', 'Self-Employed', 'Other'
);

-- 4. DimGeography: Origin and regional classification
CREATE TABLE DimGeography (
    geography_id INTEGER PRIMARY KEY AUTOINCREMENT,
    native_country TEXT NOT NULL,     -- 'United-States', 'Mexico', 'India', etc.
    region TEXT NOT NULL              -- 'North-America', 'Central-America', 'Asia', 'Europe', etc.
);

-- 5. FactIncome: Central fact table recording socioeconomic observations
CREATE TABLE FactIncome (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    demographics_id INTEGER NOT NULL REFERENCES DimDemographics(demographics_id),
    education_id INTEGER NOT NULL REFERENCES DimEducation(education_id),
    occupation_id INTEGER NOT NULL REFERENCES DimOccupation(occupation_id),
    geography_id INTEGER NOT NULL REFERENCES DimGeography(geography_id),
    capital_gain INTEGER NOT NULL DEFAULT 0,
    capital_loss INTEGER NOT NULL DEFAULT 0,
    net_capital INTEGER NOT NULL DEFAULT 0,
    hours_per_week INTEGER NOT NULL DEFAULT 40,
    income_bracket TEXT NOT NULL,     -- '<=50K' or '>50K'
    is_high_income INTEGER NOT NULL   -- 1 if '>50K', 0 otherwise
);

-- Indexes for OLAP query performance
CREATE INDEX idx_fact_demographics ON FactIncome(demographics_id);
CREATE INDEX idx_fact_education ON FactIncome(education_id);
CREATE INDEX idx_fact_occupation ON FactIncome(occupation_id);
CREATE INDEX idx_fact_geography ON FactIncome(geography_id);
CREATE INDEX idx_fact_high_income ON FactIncome(is_high_income);

