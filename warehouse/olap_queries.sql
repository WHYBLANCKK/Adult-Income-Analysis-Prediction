-- ==============================================================================
-- DMW Activity 2: Adult Census Income OLAP Operations
-- Four Core OLAP Queries Executed Against the Star Schema
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- OLAP QUERY 1: Roll-up (Hierarchical Aggregation across Education Levels)
-- Evaluates population volume, high-income count, high-income rate (%), and
-- average work hours across formal education tiers and credentials.
-- ------------------------------------------------------------------------------
SELECT 
    e.education_tier,
    e.education,
    e.education_num,
    COUNT(f.fact_id) AS total_individuals,
    SUM(f.is_high_income) AS high_income_count,
    ROUND(100.0 * SUM(f.is_high_income) / COUNT(f.fact_id), 2) AS pct_high_income,
    ROUND(AVG(f.hours_per_week), 1) AS avg_hours_per_week,
    ROUND(AVG(f.capital_gain), 2) AS avg_capital_gain
FROM FactIncome f
JOIN DimEducation e ON f.education_id = e.education_id
GROUP BY e.education_tier, e.education, e.education_num
ORDER BY e.education_num DESC;


-- ------------------------------------------------------------------------------
-- OLAP QUERY 2: Drill-down (Occupation & Gender Breakdown for Higher Degrees)
-- Drills down into individuals holding Undergraduate or Postgraduate degrees
-- to examine how high-earning rates vary by specific occupation and gender.
-- ------------------------------------------------------------------------------
SELECT 
    o.occupation,
    d.sex,
    COUNT(f.fact_id) AS sample_size,
    SUM(f.is_high_income) AS high_income_count,
    ROUND(100.0 * SUM(f.is_high_income) / COUNT(f.fact_id), 2) AS pct_high_income,
    ROUND(AVG(f.capital_gain - f.capital_loss), 2) AS avg_net_capital
FROM FactIncome f
JOIN DimEducation e ON f.education_id = e.education_id
JOIN DimOccupation o ON f.occupation_id = o.occupation_id
JOIN DimDemographics d ON f.demographics_id = d.demographics_id
WHERE e.education_tier IN ('Undergraduate', 'Postgraduate')
  AND o.occupation != '?'
GROUP BY o.occupation, d.sex
HAVING COUNT(f.fact_id) >= 20
ORDER BY pct_high_income DESC, sample_size DESC;


-- ------------------------------------------------------------------------------
-- OLAP QUERY 3: Slice and Dice (Regional & Demographic Subcube)
-- Slice: Filter by native_country = 'United-States'
-- Dice: Analyze across Age Groups and Work Sectors (Government, Private, Self-Employed)
-- Computes high-earning rate, average hours, and net capital gains.
-- ------------------------------------------------------------------------------
SELECT 
    d.age_group,
    o.job_sector,
    COUNT(f.fact_id) AS total_count,
    SUM(f.is_high_income) AS high_earners,
    ROUND(100.0 * SUM(f.is_high_income) / COUNT(f.fact_id), 2) AS high_income_rate_pct,
    ROUND(AVG(f.hours_per_week), 1) AS avg_weekly_hours,
    ROUND(AVG(f.net_capital), 1) AS avg_net_capital
FROM FactIncome f
JOIN DimGeography g ON f.geography_id = g.geography_id
JOIN DimDemographics d ON f.demographics_id = d.demographics_id
JOIN DimOccupation o ON f.occupation_id = o.occupation_id
WHERE g.native_country = 'United-States'
  AND o.job_sector != 'Other'
GROUP BY d.age_group, o.job_sector
ORDER BY d.age_group, high_income_rate_pct DESC;


-- ------------------------------------------------------------------------------
-- OLAP QUERY 4: Pivot / Cross-Tabulation (Workclass vs Marital Status High-Income Matrix)
-- Aggregates income profiles across employment sector and marital stability,
-- providing a 2D view of high-income concentration.
-- ------------------------------------------------------------------------------
SELECT 
    o.workclass,
    COUNT(CASE WHEN d.marital_status LIKE 'Married%' THEN f.fact_id END) AS married_total,
    ROUND(100.0 * SUM(CASE WHEN d.marital_status LIKE 'Married%' AND f.is_high_income = 1 THEN 1 ELSE 0 END) / 
          NULLIF(COUNT(CASE WHEN d.marital_status LIKE 'Married%' THEN f.fact_id END), 0), 2) AS married_high_income_pct,
    COUNT(CASE WHEN d.marital_status NOT LIKE 'Married%' THEN f.fact_id END) AS unmarried_total,
    ROUND(100.0 * SUM(CASE WHEN d.marital_status NOT LIKE 'Married%' AND f.is_high_income = 1 THEN 1 ELSE 0 END) / 
          NULLIF(COUNT(CASE WHEN d.marital_status NOT LIKE 'Married%' THEN f.fact_id END), 0), 2) AS unmarried_high_income_pct,
    ROUND(AVG(f.hours_per_week), 1) AS overall_avg_hours
FROM FactIncome f
JOIN DimOccupation o ON f.occupation_id = o.occupation_id
JOIN DimDemographics d ON f.demographics_id = d.demographics_id
WHERE o.workclass != '?'
GROUP BY o.workclass
ORDER BY married_high_income_pct DESC;

