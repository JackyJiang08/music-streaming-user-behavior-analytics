-- Per-user time-to-churn table for survival analysis.
--
-- Churn (event) = the first sustained inactivity gap: 14 consecutive days
-- with no listening event, consistent with the project's 14-day churn
-- concept. The event is dated when the 14-day silence completes
-- (last active day + 14); users whose observed history ends without such a
-- gap are right-censored at the data horizon (max event date). Signup day
-- counts as an engagement day, so a user who never listens churns at day 14.
-- Reactivations after a qualifying silence are ignored: the first sustained
-- disengagement is the event of interest.
--
-- Cohort: users who signed up on/after the first day of the event log, so
-- every user's full history is observed and durations are measured from
-- signup with no left truncation. Pre-log signups are excluded rather than
-- entered mid-history (their earlier behavior is unobservable).
--
-- Week-1 covariates are landmarked: computed from days 0-6 after signup,
-- which all precede the earliest possible event time (day 14), so they are
-- leakage-safe by construction.

DROP VIEW IF EXISTS survival_table;

-- TEMP, matching the other project views: session-scoped on the in-memory
-- SQLite database created by src.data_loader.connect().
CREATE TEMP VIEW survival_table AS
WITH horizon AS (
    SELECT MIN(date(event_timestamp)) AS obs_start,
           MAX(date(event_timestamp)) AS obs_end
    FROM listening_events
),
cohort AS (
    SELECT u.user_id,
           date(u.signup_date)   AS signup_date,
           u.acquisition_channel,
           u.primary_device      AS device,
           u.age_group,
           u.music_persona
    FROM users u, horizon
    WHERE date(u.signup_date) >= horizon.obs_start
),
activity AS (  -- distinct engagement days; signup itself counts as day 0
    SELECT c.user_id, date(l.event_timestamp) AS day
    FROM cohort c
    JOIN listening_events l ON l.user_id = c.user_id
    UNION
    SELECT user_id, signup_date FROM cohort
),
gaps AS (
    SELECT user_id,
           LAG(day) OVER (PARTITION BY user_id ORDER BY day) AS prev_day,
           julianday(day)
             - julianday(LAG(day) OVER (PARTITION BY user_id ORDER BY day))
             AS gap_days
    FROM activity
),
first_churn AS (  -- earliest 14-day silence between observed engagement days
    SELECT user_id, MIN(julianday(prev_day) + 14) AS churn_julian
    FROM gaps
    WHERE gap_days >= 14
    GROUP BY user_id
),
last_activity AS (
    SELECT user_id, MAX(day) AS last_day FROM activity GROUP BY user_id
),
week1 AS (  -- landmark covariates from days 0-6 after signup
    SELECT c.user_id,
           COUNT(DISTINCT date(l.event_timestamp)) AS active_days_w1,
           COUNT(*)                                AS listen_events_w1,
           AVG(l.skipped_flag)                     AS skip_rate_w1,
           MAX(CASE WHEN l.playlist_add_flag = 1 OR l.liked_flag = 1
                    THEN 1 ELSE 0 END)             AS playlist_or_like_w1
    FROM cohort c
    JOIN listening_events l
      ON l.user_id = c.user_id
     AND julianday(date(l.event_timestamp)) - julianday(c.signup_date) < 7
    GROUP BY c.user_id
)
SELECT
    c.user_id,
    c.signup_date,
    c.acquisition_channel,
    c.device,
    c.age_group,
    c.music_persona,
    COALESCE(w.active_days_w1, 0)      AS active_days_w1,
    COALESCE(w.listen_events_w1, 0)    AS listen_events_w1,
    COALESCE(w.skip_rate_w1, 0)        AS skip_rate_w1,
    COALESCE(w.playlist_or_like_w1, 0) AS playlist_or_like_w1,
    CASE
        WHEN f.churn_julian IS NOT NULL
            THEN f.churn_julian - julianday(c.signup_date)
        WHEN julianday(h.obs_end) - julianday(la.last_day) >= 14
            THEN julianday(la.last_day) + 14 - julianday(c.signup_date)
        ELSE julianday(h.obs_end) - julianday(c.signup_date)
    END AS duration_days,
    CASE
        WHEN f.churn_julian IS NOT NULL THEN 1
        WHEN julianday(h.obs_end) - julianday(la.last_day) >= 14 THEN 1
        ELSE 0
    END AS churn_event
FROM cohort c
CROSS JOIN horizon h
LEFT JOIN first_churn f ON f.user_id = c.user_id
LEFT JOIN last_activity la ON la.user_id = c.user_id
LEFT JOIN week1 w ON w.user_id = c.user_id;
