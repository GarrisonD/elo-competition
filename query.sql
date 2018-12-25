COPY(
  SELECT
    card_id,
    avg(authorized_flag::int) AS pct_of_authorized_transactions,
    mode() WITHIN GROUP (ORDER BY category_1) AS top_1_category_1,
    mode() WITHIN GROUP (ORDER BY category_2) AS top_1_category_2,
    mode() WITHIN GROUP (ORDER BY category_3) AS top_1_category_3
  FROM historical_transactions
  GROUP BY card_id
) TO STDOUT WITH CSV HEADER
