-- Per-user paid-conversion modeling table (landmark design).
--
-- The snapshot framing ("free/trial at 2026-04-01, converted after?") is
-- untrainable in this data: subscription events end mid-April, and users
-- who converted inside the observation window are already premium at the
-- snapshot — only 30 positives remain. The landmark design makes the same
-- business question well-posed by shifting the clock back:
--
--   landmark        2026-03-02 (the observation-window start in src/config.py)
--   population      users signed up before the landmark with NO paid_started
--                   event before it (never-paid at landmark)
--   label           paid_conversion_next_30d = any paid_started event in
--                   [2026-03-02, 2026-04-01)
--   feature window  [2026-01-31, 2026-03-02) — strictly before the landmark
--
-- Leakage safety by construction: every behavioral aggregate is bounded by
-- event_timestamp < '2026-03-02' while the label only reads events on/after
-- it, and no payment/plan/revenue field exists in the table at all (the
-- population is never-paid, and trial funnel flags are pre-landmark steps
-- that do not encode the label).

DROP VIEW IF EXISTS conversion_table;

-- TEMP, matching the other project views: session-scoped on the in-memory
-- SQLite database created by src.data_loader.connect().
CREATE TEMP VIEW conversion_table AS
WITH never_paid AS (
    SELECT u.user_id,
           u.primary_device      AS device,
           u.acquisition_channel,
           u.country,
           u.age_group,
           u.music_persona,
           u.student_eligible,
           u.marketing_opt_in,
           CAST(julianday('2026-03-02') - julianday(date(u.signup_date))
                AS INTEGER)      AS tenure_days_at_landmark
    FROM users u
    WHERE date(u.signup_date) < '2026-03-02'
      AND u.user_id NOT IN (
          SELECT user_id FROM subscription_events
          WHERE event_type = 'paid_started'
            AND event_timestamp < '2026-03-02'
      )
),
listening_w AS (  -- behavior in the 30 days before the landmark
    SELECT user_id,
           COUNT(DISTINCT date(event_timestamp)) AS active_days_w,
           COUNT(*)                              AS listen_events_w,
           SUM(play_duration_sec) / 60.0         AS listen_minutes_w,
           COUNT(DISTINCT session_id)            AS sessions_w,
           AVG(skipped_flag)                     AS skip_rate_w,
           AVG(completed_flag)                   AS completion_rate_w,
           SUM(liked_flag)                       AS liked_songs_w,
           SUM(playlist_add_flag)                AS playlist_adds_w,
           SUM(search_used_flag)                 AS search_events_w
    FROM listening_events
    WHERE event_timestamp >= '2026-01-31' AND event_timestamp < '2026-03-02'
    GROUP BY user_id
),
ads_w AS (
    SELECT user_id,
           COUNT(*)         AS ad_impressions_w,
           SUM(clicked_flag) AS ad_clicks_w,
           AVG(clicked_flag) AS ad_click_rate_w,
           SUM(revenue_usd)  AS ad_revenue_w
    FROM ad_events
    WHERE event_timestamp >= '2026-01-31' AND event_timestamp < '2026-03-02'
    GROUP BY user_id
),
trial_pre AS (  -- trial funnel position reached at any point before landmark
    SELECT user_id,
           MAX(CASE WHEN event_type = 'trial_exposed' THEN 1 ELSE 0 END)
               AS trial_exposed_pre,
           MAX(CASE WHEN event_type = 'trial_started' THEN 1 ELSE 0 END)
               AS trial_started_pre,
           MAX(CASE WHEN event_type = 'trial_expired' THEN 1 ELSE 0 END)
               AS trial_expired_pre
    FROM subscription_events
    WHERE event_timestamp < '2026-03-02'
    GROUP BY user_id
),
label AS (
    SELECT DISTINCT user_id, 1 AS converted
    FROM subscription_events
    WHERE event_type = 'paid_started'
      AND event_timestamp >= '2026-03-02' AND event_timestamp < '2026-04-01'
)
SELECT
    np.user_id,
    np.device,
    np.acquisition_channel,
    np.country,
    np.age_group,
    np.music_persona,
    np.student_eligible,
    np.marketing_opt_in,
    np.tenure_days_at_landmark,
    COALESCE(l.active_days_w, 0)      AS active_days_w,
    COALESCE(l.listen_events_w, 0)    AS listen_events_w,
    COALESCE(l.listen_minutes_w, 0)   AS listen_minutes_w,
    COALESCE(l.sessions_w, 0)         AS sessions_w,
    COALESCE(l.skip_rate_w, 0)        AS skip_rate_w,
    COALESCE(l.completion_rate_w, 0)  AS completion_rate_w,
    COALESCE(l.liked_songs_w, 0)      AS liked_songs_w,
    COALESCE(l.playlist_adds_w, 0)    AS playlist_adds_w,
    COALESCE(l.search_events_w, 0)    AS search_events_w,
    COALESCE(a.ad_impressions_w, 0)   AS ad_impressions_w,
    COALESCE(a.ad_clicks_w, 0)        AS ad_clicks_w,
    COALESCE(a.ad_click_rate_w, 0)    AS ad_click_rate_w,
    COALESCE(a.ad_revenue_w, 0)       AS ad_revenue_w,
    COALESCE(t.trial_exposed_pre, 0)  AS trial_exposed_pre,
    COALESCE(t.trial_started_pre, 0)  AS trial_started_pre,
    COALESCE(t.trial_expired_pre, 0)  AS trial_expired_pre,
    COALESCE(lb.converted, 0)         AS paid_conversion_next_30d
FROM never_paid np
LEFT JOIN listening_w l ON l.user_id = np.user_id
LEFT JOIN ads_w a ON a.user_id = np.user_id
LEFT JOIN trial_pre t ON t.user_id = np.user_id
LEFT JOIN label lb ON lb.user_id = np.user_id;
