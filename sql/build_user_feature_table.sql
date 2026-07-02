-- Builds the user-level feature/label wide table (view: user_level_feature_table).
-- Windows: observation 2026-03-02 .. 2026-04-01 (exclusive), snapshot 2026-04-01,
-- churn label window 14d (to 2026-04-15), conversion label window 30d (to 2026-05-01).
-- Executed against the in-memory SQLite database built by src/data_loader.py.

DROP VIEW IF EXISTS user_level_feature_table;

CREATE TEMP VIEW user_level_feature_table AS
WITH latest_subscription AS (
    SELECT user_id, plan_type AS latest_plan_type
    FROM (
        SELECT user_id, plan_type,
               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_timestamp DESC) AS rn
        FROM subscription_events
        WHERE event_timestamp < '2026-04-01'
    ) WHERE rn = 1
),
base_users AS (
    SELECT
        u.*,
        COALESCE(l.latest_plan_type, 'free') AS current_subscription_type,
        '2026-04-01' AS snapshot_date
    FROM users u
    LEFT JOIN latest_subscription l ON u.user_id = l.user_id
),
listening_features AS (
    SELECT
        user_id,
        COUNT(DISTINCT DATE(event_timestamp)) AS active_days_30d,
        COUNT(*) AS listen_events_30d,
        ROUND(SUM(play_duration_sec) / 60.0, 2) AS listen_min_30d,
        COUNT(DISTINCT session_id) AS sessions_30d,
        ROUND(AVG(play_duration_sec), 2) AS avg_play_duration_sec_30d,
        ROUND(AVG(skipped_flag), 4) AS skip_rate_30d,
        ROUND(AVG(completed_flag), 4) AS completion_rate_30d,
        SUM(liked_flag) AS liked_songs_30d,
        SUM(playlist_add_flag) AS playlist_adds_30d,
        SUM(search_used_flag) AS search_events_30d
    FROM listening_events
    WHERE event_timestamp >= '2026-03-02' AND event_timestamp < '2026-04-01'
    GROUP BY user_id
),
ad_features AS (
    SELECT
        user_id,
        COUNT(*) AS ad_impressions_30d,
        SUM(clicked_flag) AS ad_clicks_30d,
        ROUND(SUM(clicked_flag) * 1.0 / NULLIF(COUNT(*), 0), 4) AS ad_click_rate_30d,
        SUM(completed_flag) AS ad_completions_30d,
        ROUND(SUM(completed_flag) * 1.0 / NULLIF(COUNT(*), 0), 4) AS ad_completion_rate_30d,
        ROUND(SUM(revenue_usd), 4) AS ad_revenue_30d
    FROM ad_events
    WHERE event_timestamp >= '2026-03-02' AND event_timestamp < '2026-04-01'
    GROUP BY user_id
),
subscription_features AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_type = 'trial_exposed' THEN 1 ELSE 0 END) AS trial_exposed_30d,
        MAX(CASE WHEN event_type = 'trial_started' THEN 1 ELSE 0 END) AS trial_started_30d,
        MAX(CASE WHEN event_type = 'paid_started' THEN 1 ELSE 0 END) AS paid_started_in_observation_30d,
        SUM(CASE WHEN event_type = 'renewal_success' THEN 1 ELSE 0 END) AS renewal_success_count_30d,
        SUM(CASE WHEN event_type = 'payment_failed' THEN 1 ELSE 0 END) AS payment_failed_count_30d,
        SUM(CASE WHEN event_type = 'cancel' THEN 1 ELSE 0 END) AS cancel_count_30d,
        ROUND(SUM(price_usd), 2) AS subscription_revenue_observation_30d
    FROM subscription_events
    WHERE event_timestamp >= '2026-03-02' AND event_timestamp < '2026-04-01'
    GROUP BY user_id
),
churn_label AS (
    SELECT
        u.user_id,
        CASE WHEN COUNT(l.user_id) = 0 THEN 1 ELSE 0 END AS churn_label_14d
    FROM users u
    LEFT JOIN listening_events l ON u.user_id = l.user_id
       AND l.event_timestamp >= '2026-04-01' AND l.event_timestamp < '2026-04-15'
    GROUP BY u.user_id
),
conversion_label AS (
    SELECT
        u.user_id,
        CASE WHEN COUNT(s.user_id) > 0 THEN 1 ELSE 0 END AS paid_conversion_30d
    FROM users u
    LEFT JOIN subscription_events s ON u.user_id = s.user_id
       AND s.event_timestamp >= '2026-04-01' AND s.event_timestamp < '2026-05-01' AND s.event_type = 'paid_started'
    GROUP BY u.user_id
)
SELECT
    b.*,
    -- Fill nulls in aggregated features
    COALESCE(lf.active_days_30d, 0) AS active_days_30d,
    COALESCE(lf.listen_events_30d, 0) AS listen_events_30d,
    COALESCE(lf.listen_min_30d, 0.0) AS listen_min_30d,
    COALESCE(lf.sessions_30d, 0) AS sessions_30d,
    COALESCE(lf.avg_play_duration_sec_30d, 0.0) AS avg_play_duration_sec_30d,
    COALESCE(lf.skip_rate_30d, 0.0) AS skip_rate_30d,
    COALESCE(lf.completion_rate_30d, 0.0) AS completion_rate_30d,
    COALESCE(lf.liked_songs_30d, 0) AS liked_songs_30d,
    COALESCE(lf.playlist_adds_30d, 0) AS playlist_adds_30d,
    COALESCE(lf.search_events_30d, 0) AS search_events_30d,

    COALESCE(af.ad_impressions_30d, 0) AS ad_impressions_30d,
    COALESCE(af.ad_clicks_30d, 0) AS ad_clicks_30d,
    COALESCE(af.ad_click_rate_30d, 0.0) AS ad_click_rate_30d,
    COALESCE(af.ad_completions_30d, 0) AS ad_completions_30d,
    COALESCE(af.ad_completion_rate_30d, 0.0) AS ad_completion_rate_30d,
    COALESCE(af.ad_revenue_30d, 0.0) AS ad_revenue_30d,

    COALESCE(sf.trial_exposed_30d, 0) AS trial_exposed_30d,
    COALESCE(sf.trial_started_30d, 0) AS trial_started_30d,
    COALESCE(sf.paid_started_in_observation_30d, 0) AS paid_started_in_observation_30d,
    COALESCE(sf.renewal_success_count_30d, 0) AS renewal_success_count_30d,
    COALESCE(sf.payment_failed_count_30d, 0) AS payment_failed_count_30d,
    COALESCE(sf.cancel_count_30d, 0) AS cancel_count_30d,
    COALESCE(sf.subscription_revenue_observation_30d, 0.0) AS subscription_revenue_observation_30d,

    cl.churn_label_14d,
    pvl.paid_conversion_30d
FROM base_users b
LEFT JOIN listening_features lf ON b.user_id = lf.user_id
LEFT JOIN ad_features af ON b.user_id = af.user_id
LEFT JOIN subscription_features sf ON b.user_id = sf.user_id
LEFT JOIN churn_label cl ON b.user_id = cl.user_id
LEFT JOIN conversion_label pvl ON b.user_id = pvl.user_id;
