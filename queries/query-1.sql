COPY(
  SELECT
    card_id,
    
    avg(authorized_flag) AS pct_of_authorized_transactions,
    
    avg(category_1) AS pct_of_category_1_1,
    mode() WITHIN GROUP (ORDER BY category_1) AS top_1_category_1,
    
    avg(CASE WHEN category_2 = -1 THEN 1 ELSE 0 END) AS pct_of_category_2_missing,
    avg(CASE WHEN category_2 =  1 THEN 1 ELSE 0 END) AS pct_of_category_2_1,
    avg(CASE WHEN category_2 =  2 THEN 1 ELSE 0 END) AS pct_of_category_2_2,
    avg(CASE WHEN category_2 =  3 THEN 1 ELSE 0 END) AS pct_of_category_2_3,
    avg(CASE WHEN category_2 =  4 THEN 1 ELSE 0 END) AS pct_of_category_2_4,
    avg(CASE WHEN category_2 =  5 THEN 1 ELSE 0 END) AS pct_of_category_2_5,
    mode() WITHIN GROUP (ORDER BY category_2) AS top_1_category_2,

    avg(CASE WHEN category_3 = -1 THEN 1 ELSE 0 END) AS pct_of_category_3_missing,
    avg(CASE WHEN category_3 =  1 THEN 1 ELSE 0 END) AS pct_of_category_3_1,
    avg(CASE WHEN category_3 =  2 THEN 1 ELSE 0 END) AS pct_of_category_3_2,
    avg(CASE WHEN category_3 =  3 THEN 1 ELSE 0 END) AS pct_of_category_3_3,
    mode() WITHIN GROUP (ORDER BY category_3) AS top_1_category_3,
    
    avg(CASE WHEN installments =  -1 THEN 1 ELSE 0 END) AS pct_of_installments_missing,
    avg(CASE WHEN installments =   0 THEN 1 ELSE 0 END) AS pct_of_installments_0,
    avg(CASE WHEN installments =   1 THEN 1 ELSE 0 END) AS pct_of_installments_1,
    avg(CASE WHEN installments =   2 THEN 1 ELSE 0 END) AS pct_of_installments_2,
    avg(CASE WHEN installments =   3 THEN 1 ELSE 0 END) AS pct_of_installments_3,
    avg(CASE WHEN installments =   4 THEN 1 ELSE 0 END) AS pct_of_installments_4,
    avg(CASE WHEN installments =   5 THEN 1 ELSE 0 END) AS pct_of_installments_5,
    avg(CASE WHEN installments =   6 THEN 1 ELSE 0 END) AS pct_of_installments_6,
    avg(CASE WHEN installments =   7 THEN 1 ELSE 0 END) AS pct_of_installments_7,
    avg(CASE WHEN installments =   8 THEN 1 ELSE 0 END) AS pct_of_installments_8,
    avg(CASE WHEN installments =   9 THEN 1 ELSE 0 END) AS pct_of_installments_9,
    avg(CASE WHEN installments =  10 THEN 1 ELSE 0 END) AS pct_of_installments_10,
    avg(CASE WHEN installments =  11 THEN 1 ELSE 0 END) AS pct_of_installments_11,
    avg(CASE WHEN installments =  12 THEN 1 ELSE 0 END) AS pct_of_installments_12,
    avg(CASE WHEN installments = 999 THEN 1 ELSE 0 END) AS pct_of_installments_999,
    mode() WITHIN GROUP (ORDER BY installments) AS top_1_installments,
    
    mode() WITHIN GROUP (ORDER BY city_id) AS top_1_city_id,
    
    min(purchase_amount) AS min_purchase_amount,
    avg(purchase_amount) AS avg_purchase_amount,
    max(purchase_amount) AS max_purchase_amount,
    sum(purchase_amount) AS sum_purchase_amount,
    mode() WITHIN GROUP (ORDER BY purchase_amount) AS top_1_purchase_amount,
    
    mode() WITHIN GROUP (ORDER BY state_id) AS top_1_state_id,
    mode() WITHIN GROUP (ORDER BY subsector_id) AS top_1_subsector_id
  FROM historical_transactions
  GROUP BY card_id
) TO STDOUT WITH CSV HEADER
