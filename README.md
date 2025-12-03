# web-scrapping


📖 Bank Review Data ETL & Analysis Project

This project focuses on extracting, transforming, and loading (ETL) bank application review data into a PostgreSQL database, followed by executing analytical queries to verify data integrity and derive key insights on customer sentiment and ratings.

🛠️ Project Setup

Prerequisites

You need the following installed to run this project:

Python 3.x

PostgreSQL Server (running on localhost:5432)

Required Python Libraries:

# Install core libraries and the database adapter
pip install pandas psycopg2-binary ast

# (Optional) Install tabulate if you want formatted markdown output in the console
pip install tabulate


Required Data:

./data/raw/app_info.csv (Bank/App information)

./data/processed/reviews_with_sentiment.csv (Processed reviews with sentiment analysis)

Database Configuration

The project assumes the following database credentials and structure. These parameters were used in the successful ETL script:

Parameter

Value

Database Name

bank_reviews

User

admin

Password

admin123

Host

localhost

Port

5432

PostgreSQL Permissions Note:
If you encounter permission denied errors, ensure the admin user has the necessary privileges, specifically:

GRANT CONNECT ON DATABASE bank_reviews TO admin;
GRANT USAGE ON SCHEMA public TO admin;
GRANT CREATE ON SCHEMA public TO admin;
ALTER ROLE admin SET search_path = public, "$user";


📦 Database Schema

The ETL script creates two tables to store the normalized data.

banks Table

Stores metadata about the banking applications.

Column

Data Type

Constraint

Description

bank_code

VARCHAR(50)

PRIMARY KEY

Unique identifier for the bank.

bank_name

VARCHAR(255)

NOT NULL

Full name of the bank.

app_name

VARCHAR(255)



Name of the mobile application.

reviews Table

Stores the detailed review data and the processed sentiment results.

Column

Data Type

Constraint

Description

review_id

VARCHAR(100)

PRIMARY KEY

Unique ID for the review.

bank_code

VARCHAR(50)

FOREIGN KEY

Links to the banks table.

review_text

TEXT



The full text of the customer review.

rating

INTEGER



Customer rating (1-5).

sentiment_label

VARCHAR(50)



Sentiment classification (e.g., Positive, Negative).

sentiment_score

NUMERIC(5, 4)



Confidence score of the sentiment.

identified_themes

TEXT[]



Array of themes/topics identified in the review.

🚀 Data Ingestion (ETL)

The data ingestion script uses the psycopg2.extras.execute_batch function for fast, reliable bulk loading of data, handling conflict resolution and complex data type conversion (string-list to TEXT[] array).

🔍 Verification & Analysis Queries

The final Python script executes these three SQL queries to verify data integrity and derive initial insights.

1. Count Reviews per Bank (Integrity Check)

This query validates the foreign key relationship and provides a breakdown of data volume per bank.

SELECT 
    b.bank_name, 
    COUNT(r.review_id) AS total_reviews
FROM 
    reviews r
JOIN 
    banks b ON b.bank_code = r.bank_code
GROUP BY 
    b.bank_name
ORDER BY
    total_reviews DESC;


2. Average Rating per Bank (Core KPI)

Calculates the average rating for each bank, essential for comparing performance.

SELECT 
    b.bank_name, 
    ROUND(AVG(r.rating)::numeric, 2) AS average_rating,
    COUNT(r.review_id) AS review_count
FROM 
    reviews r
JOIN 
    banks b ON b.bank_code = r.bank_code
GROUP BY 
    b.bank_name
ORDER BY
    average_rating DESC;


3. Count Reviews by Sentiment Label (Sentiment Distribution)

Provides an overview of the sentiment polarity (Positive, Negative, Neutral) across the entire review dataset.

SELECT 
    sentiment_label, 
    COUNT(*) AS count_of_reviews
FROM 
    reviews
GROUP BY 
    sentiment_label
ORDER BY 
    count_of_reviews DESC;
