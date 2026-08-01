SELECT
    stage || ' ' || metric AS stage_metric,
    stage,
    metric,
    mode,
    seconds,
    request_count
FROM latency_metrics
ORDER BY sort_order, mode_order;
