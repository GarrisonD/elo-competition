COPY(
  SELECT
    card_id,

    SUM(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '1 month'  THEN purchase_amount ELSE 0 END) as purchase_amount_last_1_month,
    SUM(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '3 months' THEN purchase_amount ELSE 0 END) as purchase_amount_last_3_months,
    SUM(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '6 months' THEN purchase_amount ELSE 0 END) as purchase_amount_last_6_months,
    SUM(CASE WHEN purchase_date >= DATE '2018-03-01' - INTERVAL '1 year'   THEN purchase_amount ELSE 0 END) as purchase_amount_last_1_year
  FROM historical_transactions
  GROUP BY card_id
) TO STDOUT WITH CSV HEADER
