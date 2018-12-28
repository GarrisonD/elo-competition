COPY(
  SELECT
    card_id,

    COUNT(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '1 month'  THEN 1 END) as count_transactions_last_1_month,
    COUNT(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '3 months' THEN 1 END) as count_transactions_last_3_months,
    COUNT(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '6 months' THEN 1 END) as count_transactions_last_6_months,
    COUNT(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '1 year'   THEN 1 END) as count_transactions_last_1_year,

    SUM(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '1 month'  THEN ABS(purchase_amount) ELSE 0 END) as sum_purchase_amount_last_1_month,
    SUM(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '3 months' THEN ABS(purchase_amount) ELSE 0 END) as sum_purchase_amount_last_3_months,
    SUM(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '6 months' THEN ABS(purchase_amount) ELSE 0 END) as sum_purchase_amount_last_6_months,
    SUM(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '1 year'   THEN ABS(purchase_amount) ELSE 0 END) as sum_purchase_amount_last_1_year,

    COALESCE(AVG(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '1 month'  THEN purchase_amount END), -999) as avg_purchase_amount_last_1_month,
    COALESCE(AVG(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '3 months' THEN purchase_amount END), -999) as avg_purchase_amount_last_3_months,
    COALESCE(AVG(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '6 months' THEN purchase_amount END), -999) as avg_purchase_amount_last_6_months,
    COALESCE(AVG(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '1 year'   THEN purchase_amount END), -999) as avg_purchase_amount_last_1_year
  FROM historical_transactions
  GROUP BY card_id
) TO STDOUT WITH CSV HEADER
