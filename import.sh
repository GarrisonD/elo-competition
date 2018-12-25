create table historical_transactions (authorized_flag varchar,
                                      card_id varchar,
                                      city_id varchar,
                                      category_1 varchar,
                                      installments int,
                                      category_3 varchar,
                                      merchant_category_id varchar,
                                      merchant_id varchar,
                                      month_lag int,
                                      purchase_amount double precision,
                                      purchase_date timestamp,
                                      category_2 double precision,
                                      state_id int,
                                      subsector_id int);

COPY historical_transactions FROM '/home/data-scientist/elo-merchant-category-recommendation/historical_transactions.csv' CSV HEADER;
