-- Read-only pre-migration report for ExtraID identity conflicts.
-- Do not automatically merge/delete these rows: game progress lives in the
-- primary database and needs an explicit, audited owner decision.

SELECT
    'user_id' AS conflict_type,
    user_id::text AS normalized_value,
    COUNT(*) AS row_count,
    ARRAY_AGG(id ORDER BY created_at, id) AS account_ids,
    ARRAY_AGG(user_id ORDER BY created_at, id) AS user_ids
FROM extra_accounts
WHERE user_id IS NOT NULL AND deleted_at IS NULL
GROUP BY user_id
HAVING COUNT(*) > 1

UNION ALL

SELECT
    'email' AS conflict_type,
    LOWER(email) AS normalized_value,
    COUNT(*) AS row_count,
    ARRAY_AGG(id ORDER BY created_at, id) AS account_ids,
    ARRAY_AGG(user_id ORDER BY created_at, id) AS user_ids
FROM extra_accounts
WHERE deleted_at IS NULL
GROUP BY LOWER(email)
HAVING COUNT(*) > 1

UNION ALL

SELECT
    'nickname' AS conflict_type,
    LOWER(nickname) AS normalized_value,
    COUNT(*) AS row_count,
    ARRAY_AGG(id ORDER BY created_at, id) AS account_ids,
    ARRAY_AGG(user_id ORDER BY created_at, id) AS user_ids
FROM extra_accounts
WHERE deleted_at IS NULL AND nickname IS NOT NULL
GROUP BY LOWER(nickname)
HAVING COUNT(*) > 1

ORDER BY conflict_type, normalized_value;
