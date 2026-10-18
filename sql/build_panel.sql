-- Build a state x generic drug x year panel of diabetes drugs from the
-- CMS Part D Prescribers by Geography and Drug files.
-- src/02_build_panel.py loads each year into table src (with a year column)
-- after cleaning column names, then runs this file.

CREATE OR REPLACE TABLE raw AS
SELECT
    year,
    Prscrbr_Geo_Lvl                       AS geo_lvl,
    Prscrbr_Geo_Cd                        AS state_fips,
    Prscrbr_Geo_Desc                      AS state,
    Brnd_Name                             AS brand,
    Gnrc_Name                             AS generic,
    lower(Gnrc_Name)                      AS g,
    TRY_CAST(Tot_Clms AS DOUBLE)          AS claims,
    TRY_CAST(Tot_30day_Fills AS DOUBLE)   AS fills,
    TRY_CAST(Tot_Drug_Cst AS DOUBLE)      AS drug_cost,
    TRY_CAST(Tot_Benes AS DOUBLE)         AS benes,
    GE65_Sprsn_Flag                       AS ge65_flag,
    TRY_CAST(GE65_Tot_30day_Fills AS DOUBLE) AS ge65_fills,
    TRY_CAST(LIS_Bene_Cst_Shr AS DOUBLE)     AS lis_oop,
    TRY_CAST(NonLIS_Bene_Cst_Shr AS DOUBLE)  AS nonlis_oop
FROM src;

-- Keep the 50 states plus DC (FIPS 01 to 56). Drops national rows, territories and unknown.
CREATE OR REPLACE TABLE states AS
SELECT * FROM raw
WHERE geo_lvl = 'State'
  AND TRY_CAST(state_fips AS INTEGER) BETWEEN 1 AND 56;

-- Classify drugs. Insulin is treated. Oral and DPP-4 or SGLT2 drugs are controls.
-- GLP-1s are left out on purpose: obesity driven demand broke their trend after 2021.
-- Insulin and GLP-1 combinations (Soliqua, Xultophy) and supplies are dropped.
CREATE OR REPLACE TABLE classified AS
SELECT *,
    CASE
        WHEN g LIKE '%needle%' OR g LIKE '%syringe%' OR g LIKE '%lancet%'      THEN NULL
        WHEN g LIKE '%device%'                                               THEN NULL
        WHEN g LIKE '%lixisenatide%' OR g LIKE '%liraglutide%'               THEN NULL
        WHEN g LIKE '%insulin%'                                              THEN 'insulin'
        WHEN g LIKE '%gliflozin%'                                            THEN 'sglt2'
        WHEN g LIKE '%gliptin%'                                              THEN 'dpp4'
        WHEN g LIKE '%metformin%'                                            THEN 'metformin'
        WHEN g LIKE '%glipizide%' OR g LIKE '%glimepiride%' OR g LIKE '%glyburide%' THEN 'sulfonylurea'
        WHEN g LIKE '%pioglitazone%' OR g LIKE '%rosiglitazone%'             THEN 'tzd'
        WHEN g LIKE '%acarbose%' OR g LIKE '%miglitol%'
          OR g LIKE '%repaglinide%' OR g LIKE '%nateglinide%'                THEN 'other_oral'
        ELSE NULL
    END AS drug_class
FROM states;

-- Collapse brands to generic. Bene counts summed across brands slightly overcount
-- people who used two brands of the same generic, so fills are the main outcome.
CREATE OR REPLACE TABLE panel_raw AS
SELECT
    state_fips, any_value(state) AS state, generic, drug_class, year,
    (drug_class = 'insulin')::INTEGER                    AS insulin,
    sum(claims)                                          AS claims,
    sum(fills)                                           AS fills,
    sum(drug_cost)                                       AS drug_cost,
    sum(benes)                                           AS benes,
    sum(coalesce(lis_oop, 0) + coalesce(nonlis_oop, 0))  AS oop,
    sum(coalesce(nonlis_oop, 0))                         AS nonlis_oop,
    -- age split only where CMS did not suppress the 65+ cell
    sum(CASE WHEN coalesce(ge65_flag, '') = '' THEN ge65_fills END)          AS ge65_fills,
    sum(CASE WHEN coalesce(ge65_flag, '') = '' THEN fills - ge65_fills END)  AS u65_fills,
    bool_and(coalesce(ge65_flag, '') = '')               AS age_split_ok
FROM classified
WHERE drug_class IS NOT NULL
GROUP BY state_fips, generic, drug_class, year;

-- Balanced panel: a state x drug cell must have positive fills and at least
-- 50 claims in every year, so logs are defined and tiny cells don't add noise.
CREATE OR REPLACE TABLE panel AS
WITH n_years AS (SELECT count(DISTINCT year) AS n FROM panel_raw),
keep AS (
    SELECT state_fips, generic
    FROM panel_raw
    GROUP BY state_fips, generic
    HAVING count(*) = (SELECT n FROM n_years)
       AND min(fills) > 0
       AND min(claims) >= 50
)
SELECT p.*,
       p.state_fips || '|' || p.generic                     AS unit,
       ln(p.fills)                                          AS log_fills,
       ln(p.claims)                                         AS log_claims,
       p.fills / nullif(p.benes, 0)                         AS fills_per_bene,
       p.oop / nullif(p.fills, 0)                           AS oop_per_fill,
       ln(nullif(p.oop, 0) / p.fills)                       AS log_oop_per_fill
FROM panel_raw p
JOIN keep USING (state_fips, generic);
