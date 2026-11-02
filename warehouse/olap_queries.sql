-- ==============================================================================
-- DMW Activity 2: Adult Census Income OLAP Operations
-- Four Core OLAP Queries Executed Against the Star Schema
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- OLAP QUERY 1: Roll-up (Aggregate Income Distribution from Occupation to Workclass)
-- Aggregates workforce count and high-earner rate across individual occupations,
-- rolling up to broader employment sectors (workclass).
-- ------------------------------------------------------------------------------
SELECT 
    o.workclass,
    o.occupation,
    COUNT(f.fact_id) AS total_workers,
    SUM(f.income_flag) AS high_earners_count,
    ROUND(100.0 * SUM(f.income_flag) / COUNT(f.fact_id), 2) AS high_earner_pct,
    ROUND(AVG(f.hours_per_week), 1) AS avg_hours_per_week
FROM FactIncome f
JOIN DimOccupation o ON f.occ_id = o.occ_id
WHERE o.workclass != '?' AND o.occupation != '?'
GROUP BY o.workclass, o.occupation
ORDER BY o.workclass, high_earner_pct DESC;


-- ------------------------------------------------------------------------------
-- OLAP QUERY 2: Drill-down (Analyze High-Earner Rate: Education Levels to Marital Status)
-- Drills down into each education tier to reveal how marital stability impacts
-- the probability of earning >50K across credentials.
-- ------------------------------------------------------------------------------
SELECT 
    e.education_level,
    e.education_num,
    d.marital_status,
    COUNT(f.fact_id) AS sample_size,
    SUM(f.income_flag) AS high_earners_count,
    ROUND(100.0 * SUM(f.income_flag) / COUNT(f.fact_id), 2) AS high_earner_pct,
    ROUND(AVG(f.capital_gain), 2) AS avg_capital_gain
FROM FactIncome f
JOIN DimEducation e ON f.edu_id = e.edu_id
JOIN DimDemographics d ON f.demo_id = d.demo_id
GROUP BY e.education_level, e.education_num, d.marital_status
HAVING COUNT(f.fact_id) >= 20
ORDER BY e.education_num DESC, high_earner_pct DESC;


-- ------------------------------------------------------------------------------
-- OLAP QUERY 3: Slice (Filter: sex = 'Female', Analyze Hours Worked vs. Income)
-- Isolates the female workforce subcube to quantify the relationship between
-- weekly working hours brackets and high-income attainment.
-- ------------------------------------------------------------------------------
SELECT 
    CASE 
        WHEN f.hours_per_week < 35 THEN 'Part-time (<35 hrs)'
        WHEN f.hours_per_week BETWEEN 35 AND 45 THEN 'Full-time (35-45 hrs)'
        ELSE 'Overtime (>45 hrs)'
    END AS work_hours_category,
    COUNT(f.fact_id) AS total_females,
    SUM(f.income_flag) AS high_earners_count,
    ROUND(100.0 * SUM(f.income_flag) / COUNT(f.fact_id), 2) AS female_high_income_pct,
    ROUND(AVG(f.hours_per_week), 1) AS avg_weekly_hours,
    ROUND(AVG(f.capital_gain), 2) AS avg_capital_gain
FROM FactIncome f
JOIN DimDemographics d ON f.demo_id = d.demo_id
WHERE d.sex = 'Female'
GROUP BY work_hours_category
ORDER BY female_high_income_pct DESC;


-- ------------------------------------------------------------------------------
-- OLAP QUERY 4: Dice (Filter: hours_per_week >= 40, age_band = '30-45', workclass = 'Private')
-- Slices the multidimensional cube along three simultaneous constraints:
-- Prime career age (30-45), standard/overtime work (>=40 hrs), and private sector.
-- Analyzes income distribution across education levels and marital status within this subcube.
-- ------------------------------------------------------------------------------
SELECT 
    d.age_band,
    o.workclass,
    e.education_level,
    d.marital_status,
    COUNT(f.fact_id) AS cohort_size,
    SUM(f.income_flag) AS high_earners_count,
    ROUND(100.0 * SUM(f.income_flag) / COUNT(f.fact_id), 2) AS high_income_pct,
    ROUND(AVG(f.hours_per_week), 1) AS avg_weekly_hours,
    ROUND(AVG(f.capital_gain - f.capital_loss), 2) AS avg_net_capital
FROM FactIncome f
JOIN DimDemographics d ON f.demo_id = d.demo_id
JOIN DimOccupation o ON f.occ_id = o.occ_id
JOIN DimEducation e ON f.edu_id = e.edu_id
WHERE f.hours_per_week >= 40
  AND d.age_band = '30-45'
  AND o.workclass = 'Private'
GROUP BY d.age_band, o.workclass, e.education_level, d.marital_status
HAVING COUNT(f.fact_id) >= 20
ORDER BY high_income_pct DESC;
