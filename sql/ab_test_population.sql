-- Eligible population for the home-screen personalized-playlist experiment.
--
-- Unit of randomization: user. Eligibility: non-paying listeners (free tier
-- and in-trial) at the snapshot. The module changes the free home experience
-- they all share, and short-term retention concentrates there (14d churn
-- 60.4% free / 49.5% trial vs ~33% premium). The ideal target — users who
-- signed up in the days right before the experiment start — is empty in this
-- snapshot (minimum observed tenure is 18 days), so tenure_days is exported
-- as a covariate and new-vs-established users become a drill-down segment
-- instead of an eligibility filter. Depends on the wide table built by
-- sql/build_user_feature_table.sql (run that script first).

DROP VIEW IF EXISTS ab_test_population;

-- TEMP, matching user_level_feature_table: session-scoped views on top of the
-- in-memory SQLite database created by src.data_loader.connect().
CREATE TEMP VIEW ab_test_population AS
SELECT
    user_id,
    current_subscription_type,
    CAST(julianday(snapshot_date) - julianday(signup_date) AS INTEGER)
        AS tenure_days,
    active_days_30d,
    listen_min_30d AS listen_minutes_30d,
    skip_rate_30d,
    playlist_adds_30d,
    ad_revenue_30d,
    cancel_count_30d,
    churn_label_14d
FROM user_level_feature_table
WHERE current_subscription_type IN ('free', 'trial');
