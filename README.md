```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.io as pio
pio . renderers . default = 'notebook'
from sqlalchemy import create_engine

# Veritabanı bağlantısını oluşturma
# SQL sorguları ile oluşturulan tablolar çekildi.
engine = create_engine('postgresql+psycopg2://postgres:6587@localhost:5432/postgres')

# SQL sorgularını tanımlama
queries = [
    """
    SELECT 
    p.product_id,
    p.product_name,
    COUNT(DISTINCT o.order_id) AS number_of_orders
FROM 
    order_details AS od
INNER JOIN 
    products AS p ON od.product_id = p.product_id
INNER JOIN 
    orders AS o ON od.order_id = o.order_id
GROUP BY 
    p.product_id, p.product_name
HAVING 
    COUNT(DISTINCT o.order_id ) > 1
ORDER BY 
    number_of_orders DESC;

    """,
    """
    WITH CustomerOrderCounts AS (
    SELECT 
        c.customer_id,
        c.company_name,
        COUNT(o.order_id) AS total_orders
    FROM 
        customers AS c
    LEFT JOIN 
        orders AS o ON c.customer_id = o.customer_id
    GROUP BY 
        c.customer_id, c.company_name
),
TopCustomers AS (
    SELECT 
        customer_id,
        company_name,
        total_orders
    FROM 
        CustomerOrderCounts
    ORDER BY 
        total_orders DESC
)
SELECT 
    tc.customer_id,
    tc.company_name,
    tc.total_orders,
    c.country AS customer_country
FROM 
    TopCustomers AS tc
INNER JOIN 
    customers AS c ON tc.customer_id = c.customer_id
ORDER BY 
    tc.total_orders DESC;


    """,
    """
    WITH CustomerPurchases AS (
    SELECT 
        c.customer_id,
        c.company_name,
        COUNT(o.order_id) AS number_of_orders,
        SUM(od.quantity * od.unit_price) AS total_spent
    FROM 
        customers AS c
    INNER JOIN 
        orders AS o ON c.customer_id = o.customer_id
    INNER JOIN 
        order_details AS od ON o.order_id = od.order_id
    GROUP BY 
        c.customer_id, c.company_name
)
SELECT 
    customer_id,
    company_name,
    number_of_orders,
    to_char(total_spent, 'FM$999,999,999.00') AS total_spent_formatted,
    CASE
        WHEN total_spent > 50000 THEN 'High Value'
        WHEN total_spent BETWEEN 30000 AND 50000 THEN 'Medium Value'
        ELSE 'Low Value'
    END AS customer_segment
FROM 
    CustomerPurchases
ORDER BY 
    total_spent DESC;

   """,
    """
    WITH WeekDays AS (
    SELECT 'Monday' AS day_name, 1 AS day_number
    UNION SELECT 'Tuesday', 2
    UNION SELECT 'Wednesday', 3
    UNION SELECT 'Thursday', 4
    UNION SELECT 'Friday', 5
    UNION SELECT 'Saturday', 6
    UNION SELECT 'Sunday', 7
),
OrderCounts AS (
    SELECT
        TO_CHAR(o.order_date, 'Day') AS order_day,
        TO_CHAR(o.order_date, 'D')::int AS day_number,
        COUNT(o.order_id) AS number_of_orders
    FROM
        orders AS o
    GROUP BY
        TO_CHAR(o.order_date, 'Day'),
        TO_CHAR(o.order_date, 'D')
)
SELECT
    wd.day_name AS order_day,
    wd.day_number,
    COALESCE(oc.number_of_orders, 0) AS number_of_orders
FROM
    WeekDays AS wd
LEFT JOIN
    OrderCounts AS oc ON wd.day_number = oc.day_number
ORDER BY
    wd.day_number;

    """,
    """
    SELECT
    TO_CHAR(o.order_date, 'YYYY-MM') AS order_month,
    COUNT(o.order_id) AS number_of_orders
FROM
    orders AS o
GROUP BY
    TO_CHAR(o.order_date, 'YYYY-MM')
ORDER BY
    order_month;

    """,
    """
    SELECT
    TO_CHAR(o.order_date, 'Month') AS order_month,
    COUNT(o.order_id) AS number_of_orders
FROM
    orders AS o
GROUP BY
    TO_CHAR(o.order_date, 'Month'),
    EXTRACT(MONTH FROM o.order_date)
ORDER BY
    EXTRACT(MONTH FROM o.order_date);

    """,
    """
    SELECT 
    s."supplier_id",
    ROUND(AVG(o."required_date" - o."order_date")) AS average_last_day,
    ROUND(AVG(o."shipped_date" - o."order_date")) AS average_processing_days,
    ROUND(AVG(o."required_date" - o."shipped_date")) AS average_delivery_days
FROM 
    "orders" o
JOIN 
    "order_details" od ON o."order_id" = od."order_id"
JOIN 
    "products" p ON od."product_id" = p."product_id"
JOIN 
    "suppliers" s ON p."supplier_id" = s."supplier_id"
GROUP BY 
    s."supplier_id"
ORDER BY 
    average_processing_days, average_delivery_days;

    """,
    """
    SELECT 
    DATE_PART('year', o."order_date") AS year,
    DATE_PART('month', o."order_date") AS month,
    TO_CHAR(SUM(od.quantity * od.unit_price * (1 - od.discount)), '$999,999,999.99') AS monthly_income
FROM 
    "orders" o
JOIN 
    "order_details" od ON o."order_id" = od."order_id"
GROUP BY 
    DATE_PART('year', o."order_date"),
    DATE_PART('month', o."order_date")
ORDER BY 
    year,
    month;

    """,
    """
    SELECT 
    c."category_name" AS category,
    DATE_PART('year', o."order_date") AS year,
    DATE_PART('month', o."order_date") AS month,
    TO_CHAR(SUM(od.quantity * od.unit_price * (1 - od.discount)), '$999,999,999.99') AS monthly_income
FROM 
    "orders" o
JOIN 
    "order_details" od ON o."order_id" = od."order_id"
JOIN 
    "products" p ON od."product_id" = p."product_id"
JOIN 
    "categories" c ON p."category_id" = c."category_id"
GROUP BY 
    c."category_name",
    DATE_PART('year', o."order_date"),
    DATE_PART('month', o."order_date")
ORDER BY 
    year,
    month,
    category;

    """,
    """
    SELECT 
    DATE_PART('year', o."order_date") AS year,
    TO_CHAR(SUM(od.quantity * od.unit_price * (1 - od.discount)), '$999,999,999.99') AS yearly_income
FROM 
    "orders" o
JOIN 
    "order_details" od ON o."order_id" = od."order_id"
GROUP BY 
    DATE_PART('year', o."order_date")
ORDER BY 
    year;

    """,
    """
    SELECT 
    c."category_name" AS category,
    DATE_PART('year', o."order_date") AS year,
    TO_CHAR(SUM(od.quantity * od.unit_price * (1 - od.discount)), '$999,999,999.99') AS yearly_income
FROM 
    "orders" o
JOIN 
    "order_details" od ON o."order_id" = od."order_id"
JOIN 
    "products" p ON od."product_id" = p."product_id"
JOIN 
    "categories" c ON p."category_id" = c."category_id"
GROUP BY 
    c."category_name",
    DATE_PART('year', o."order_date")
ORDER BY 
    year,
    category;

    """,
    """
 WITH ReorderedProducts AS (
    SELECT 
        od."product_id",
        p."product_name",
        DATE_PART('year', o."order_date") AS year,
        SUM(od."quantity") AS total_quantity
    FROM 
        "orders" o
    JOIN 
        "order_details" od ON o."order_id" = od."order_id"
    JOIN 
        "products" p ON od."product_id" = p."product_id"
    WHERE 
        EXISTS (
            SELECT 1
            FROM "order_details" od2
            WHERE od2."product_id" = od."product_id"
              AND od2."order_id" <> od."order_id"
        )
    GROUP BY 
        od."product_id",
        p."product_name",
        DATE_PART('year', o."order_date")
),
RankedProducts AS (
    SELECT
        product_id,
        product_name,
        year,
        total_quantity,
        RANK() OVER (PARTITION BY year ORDER BY total_quantity DESC) AS rnk
    FROM
        ReorderedProducts
)
, Top5ReorderedProducts AS (
    SELECT
        product_id,
        product_name,
        year
    FROM
        RankedProducts
    WHERE
        rnk <= 5
)
SELECT 
    p."product_name",
    tp.year,
    TO_CHAR(SUM(od."quantity" * od."unit_price" * (1 - od."discount")), '$999,999,999.99') AS yearly_income
FROM 
    "order_details" od
JOIN 
    "products" p ON od."product_id" = p."product_id"
JOIN 
    Top5ReorderedProducts tp ON p."product_id" = tp."product_id"
GROUP BY 
    p."product_name",
    tp.year
ORDER BY 
    tp.year,
    yearly_income DESC
	limit 10;


    """,
    """
    WITH max_o_d AS (
    SELECT customer_id,
           MAX(order_date) AS max_order_date
    FROM orders
    GROUP BY customer_id
),
recency AS (
    SELECT customer_id,
           max_order_date,
           ('1998-05-10'::date - max_order_date::date) AS recency
    FROM max_o_d
    
),
frequency AS (
    SELECT customer_id,
           COUNT(*) AS frequency
    FROM orders
    GROUP BY customer_id
),
monetary AS (
    SELECT o.customer_id,	
           ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount))::numeric, 0) AS monetary
    FROM order_details od
    JOIN orders o ON od.order_id = o.order_id
    GROUP BY o.customer_id
),
scores AS (
    SELECT
        r.customer_id,
        r.recency,
        NTILE(5) OVER(ORDER BY r.recency DESC) AS recency_score,
        f.frequency,
        NTILE(5) OVER(ORDER BY f.frequency DESC) AS frequency_score, 
        m.monetary,
        NTILE(5) OVER(ORDER BY m.monetary ASC) AS monetary_score
    FROM recency r	
    LEFT JOIN frequency f ON r.customer_id = f.customer_id
    LEFT JOIN monetary m ON f.customer_id = m.customer_id
),
monetary_frequency AS (
    SELECT customer_id,
           recency_score,
           frequency_score + monetary_score AS mon_fre_score
    FROM scores
),
rfm_score AS (
    SELECT customer_id,
           recency_score,	
           NTILE(5) OVER(ORDER BY mon_fre_score) AS mon_fre_score
    FROM monetary_frequency
)
SELECT 
    customer_id,
    recency_score,
    frequency_score,
    monetary_score,
    (recency_score + frequency_score + monetary_score) AS total_score,
    ((recency_score + frequency_score + monetary_score) / 3) AS average_score
FROM 
   scores;

    """,
    """
    WITH first_order AS (
    SELECT customer_id,
           MIN(DATE_TRUNC('month', order_date)) AS first_order_month
    FROM orders
    GROUP BY customer_id
),

orders_with_period AS (
    SELECT o.customer_id,
           o.order_id,
           DATE_TRUNC('month', o.order_date) AS order_month,
           fo.first_order_month,
           EXTRACT(YEAR FROM age(DATE_TRUNC('month', o.order_date), fo.first_order_month)) * 12 + 
           EXTRACT(MONTH FROM age(DATE_TRUNC('month', o.order_date), fo.first_order_month)) AS period
    FROM orders o
    JOIN first_order fo ON o.customer_id = fo.customer_id
)

SELECT 
    TO_CHAR(first_order_month, 'YYYY-MM') AS cohort_month,
    period,
    COUNT(DISTINCT owp.customer_id) AS num_customers,
    TO_CHAR(ROUND(SUM(od.quantity * od.unit_price * (1 - od.discount))::numeric, 2), 'FM$999,999,999.00') AS total_revenue
FROM 
    orders_with_period owp
JOIN
    order_details od ON owp.order_id = od.order_id
GROUP BY 
    first_order_month, period
ORDER BY 
    first_order_month, period;


    """,
    """
    SELECT 
    e.employee_id, 
    e.first_name, 
    e.last_name, 
    COUNT(o.order_id) AS order_count,
    TO_CHAR(SUM(od.unit_price * od.quantity * (1 - od.discount)), 'FM$999,999,999.00') AS total_revenue
FROM 
    employees e
LEFT JOIN 
    orders o ON e.employee_id = o.employee_id
LEFT JOIN 
    order_details od ON o.order_id = od.order_id
GROUP BY 
    e.employee_id, 
    e.first_name, 
    e.last_name
ORDER BY 
    total_revenue asc;

    """
]

# Sorgu sonuçlarını saklamak için bir sözlük oluşturma
dataframes = {}

# Sorguları çalıştırma ve sonuçları sözlüğe ekleme
for idx, query in enumerate(queries):
    df_name = f'df{idx + 1}'
    df = pd.read_sql(query, engine)
    dataframes[df_name] = df

# DataFrame'lerin isimlerini ve ilk birkaç satırını kontrol etme
for df_name, df in dataframes.items():
    print(f'{df_name}:\n', df.head(), '\n')
```

```python
#1-Product Analysis
#-Yeniden sipariş verilen ürünler ve sipariş sayıları



# df1'i alıp dikey çubuk grafik çizmek
def plot_vertical_bar_chart_with_labels(df):
    # Sipariş sayılarına göre df1'i sırala ve ilk 10 kaydı seç
    top_10_df = df.sort_values(by='number_of_orders', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(x='product_name', y='number_of_orders', data=top_10_df, palette='viridis')
    
    plt.title('Top 10 Product Orders')
    plt.xlabel('Product Name')
    plt.ylabel('Number of Orders')
    plt.xticks(rotation=45)
    
    # Veri etiketlerini eklemek
    for index, row in top_10_df.iterrows():
        barplot.text(row.name, row.number_of_orders, round(row.number_of_orders, 2), color='black', ha="center")

    plt.show()

# df1'in grafiğini çizmek
plot_vertical_bar_chart_with_labels(dataframes['df1'])
```

```python
# 2-Customer Analysis
    #2.1-En çok sipariş veren firmalar, sipariş sayısı ve ülkeleri 

def plot_country_orders(df):
    # Toplam siparişleri şirket adına göre grupluyoruz
    company_orders = df.groupby('company_name').agg({
        'total_orders': 'sum',
        'customer_country': 'first'  # İlk ülkeyi alıyoruz
    }).reset_index()
    
    # Toplam siparişlere göre sıralıyoruz ve ilk 10'u alıyoruz
    top_10_company_orders = company_orders.sort_values(by='total_orders', ascending=False).head(10)
    
    plt.figure(figsize=(14, 8))
    
    # Barplot oluşturuluyor
    bar_plot = sns.barplot(x='total_orders', y='company_name', data=top_10_company_orders, palette='Set1')
    
    # Her barın üzerine veri etiketlerini ekliyoruz
    for index, value in enumerate(top_10_company_orders['total_orders']):
        country = top_10_company_orders.iloc[index]['customer_country']
        bar_plot.text(value, index, f'{value} ({country})', va='center', ha='left', color='black', fontsize=10)
    
    # Başlık ve eksen etiketleri
    plt.title('Total Orders by Top 10 Companies')
    plt.xlabel('Total Orders')
    plt.ylabel('Company Name')
    
    # Grafiği göster
    plt.show()

# df2'nin grafiğini çizmek
df2 = dataframes['df2']
plot_country_orders(df2)
```


```python
#2.2-Toplam firma harcamaları 
#-50.000 ve üzeri high value- 30.000/50.000 arası medium- 30.000 altı low value


def plot_top_10_companies_spending(df):
    # 'total_spent_formatted' sütununu sayısal verilere dönüştürüyoruz
    df['total_spent'] = df['total_spent_formatted'].replace('[\$,]', '', regex=True).astype(float)
    
    # Toplam harcamaları şirket adına göre grupluyoruz
    company_spending = df.groupby('company_name').agg({
        'total_spent': 'sum'
    }).reset_index()
    
    # Toplam harcamalara göre sıralıyoruz ve ilk 10'u alıyoruz
    top_10_company_spending = company_spending.sort_values(by='total_spent', ascending=False).head(10)
    
    plt.figure(figsize=(14, 8))
    
    # Barplot oluşturuluyor
    bar_plot = sns.barplot(x='total_spent', y='company_name', data=top_10_company_spending, palette='Set2')
    
    # Her barın üzerine veri etiketlerini ekliyoruz
    for index, value in enumerate(top_10_company_spending['total_spent']):
        bar_plot.text(value, index, f'{value:,.2f}', va='center', ha='left', color='black', fontsize=10)
    
    # Başlık ve eksen etiketleri
    plt.title('Top 10 by Total Company Revenue')
    plt.xlabel('Total Spending')
    plt.ylabel('Company Name')
    
    # Grafiği göster
    plt.show()

# df3'ün grafiğini çizmek
df3 = dataframes['df3']
plot_top_10_companies_spending(df3)
```


```python
# 3. Order Analysis
# 3.1-Haftanın günleri bazlı order sayısı

def plot_order_day_vs_number_of_orders(df):
    plt.figure(figsize=(14, 8))
    
    # Dikey barplot oluşturuluyor
    bar_plot = sns.barplot(x='order_day', y='number_of_orders', data=df, palette='Set3')
    
    # Her barın üzerine veri etiketlerini ekliyoruz
    for index, value in enumerate(df['number_of_orders']):
        bar_plot.text(index, value, f'{value}', va='bottom', ha='center', color='black', fontsize=10)
    
    # Başlık ve eksen etiketleri
    plt.title('Number of Orders by Order Day')
    plt.xlabel('Order Day')
    plt.ylabel('Number of Orders')
    
    # X eksenindeki etiketlerin doğru şekilde gösterilmesi için
    plt.xticks(rotation=45)  # Etiketleri döndürerek daha okunaklı hale getir
    
    # Grafiği göster
    plt.show()

# df4'ün grafiğini çizmek
df4 = dataframes['df4']
plot_order_day_vs_number_of_orders(df4)
```


```python
#3.2-Yıllara göre aylık order sayısı

# df5'i dataframes sözlüğünden alalım
df5 = dataframes['df5']

# 'order_month' sütununu datetime formatına dönüştürme
df5['order_month'] = pd.to_datetime(df5['order_month'])

# 'order_month' sütununu sadece yıl ve ay olarak formatlama
df5['order_month'] = df5['order_month'].dt.to_period('M').astype(str)

# Görselleştirme fonksiyonu
def plot_order_month_vs_number_of_orders(df):
    plt.figure(figsize=(14, 8))
    
    # Dikey barplot oluşturuluyor
    bar_plot = sns.barplot(x='order_month', y='number_of_orders', data=df, palette='Set3')
    
    # Her barın üzerine veri etiketlerini ekliyoruz
    for index, value in enumerate(df['number_of_orders']):
        bar_plot.text(index, value, f'{value}', va='bottom', ha='center', color='black', fontsize=10)
    
    # Başlık ve eksen etiketleri
    plt.title('Number of Orders by Order Month')
    plt.xlabel('Order Month')
    plt.ylabel('Number of Orders')
    
    # X eksenindeki etiketlerin doğru şekilde gösterilmesi için
    plt.xticks(rotation=45)  # Etiketleri döndürerek daha okunaklı hale getir
    
    # Grafiği göster
    plt.show()

# df5'in grafiğini çizmek
plot_order_month_vs_number_of_orders(df5)
```


```python
# 4. Revenue Analysis
     #-Aylık gelir analizi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# df8'i alıyoruz
df8 = dataframes['df8']

# Yıl ve ay sütunlarını birleştirerek datetime sütunu oluşturma
df8['date'] = pd.to_datetime(df8[['year', 'month']].assign(day=1))

# Para birimi sembollerini kaldırma ve sayısal formata dönüştürme
df8['monthly_income'] = df8['monthly_income'].replace('[\$,]', '', regex=True).astype(float)

# Monthly income'u küçükten büyüğe sıralama
df8_sorted = df8.sort_values(by='monthly_income')

def format_currency(value):
    formatted_value = f"{value:,.2f}"
    # Sıfırları kaldırma
    if formatted_value.endswith('.00'):
        return formatted_value[:-3]
    return formatted_value

def plot_monthly_income(df):
    plt.figure(figsize=(14, 8))
    
    # Line plot oluşturuluyor
    line_plot = sns.lineplot(x='date', y='monthly_income', data=df, marker='o',color='gray')
    
    # Veri etiketlerini ekleme
    for i in range(len(df)):
        plt.text(df['date'].iloc[i], df['monthly_income'].iloc[i], 
                 format_currency(df['monthly_income'].iloc[i]),
                 color='black', ha='right', va='bottom')
    
    # Başlık ve eksen etiketleri
    plt.title('Monthly Income Over Time')
    plt.xlabel('Date')
    plt.ylabel('Monthly Income')
    
    # Grafiği göster
    plt.grid(True)
    plt.show()

# df8_sorted'in grafiğini çizmek
plot_monthly_income(df8_sorted)
```


```python
#4.1-Yıl bazlı gelir analizi
import plotly.express as px

# df10'u alıyoruz
df10 = dataframes['df10']

# Sütun isimlerini ve veri örneklerini yazdırma
print("Columns in df10:", df10.columns)
print("First few rows in df10:")
print(df10.head())

# Yıl bazında toplam yıllık gelirleri hesaplama
# Önce yearly_income sütunundaki $ ve virgülleri kaldırıp sayısal formata dönüştürüyoruz
df10['yearly_income'] = df10['yearly_income'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Yıl bazında toplam yıllık gelirleri hesaplama
yearly_income = df10.groupby('year')['yearly_income'].sum().reset_index()

# Yıllık gelirlerin kontrolü
print("Yearly income data:")
print(yearly_income)

# Donut grafiği oluşturma
fig = px.pie(yearly_income, values='yearly_income', names='year', hole=0.3)

# Başlık ve yerleşim düzenlemeleri
fig.update_layout(
    title_text='Yearly Income Distribution',
    title_x=0.5,  # Başlığı yatay olarak ortalar
    title_y=0.5,  # Başlığı dikey olarak ortalar (0.5, 0.5 grafiğin tam ortasıdır)
    title_font_size=24  # Başlık yazı tipi boyutu
)

# Grafiği gösterme
df10 = dataframes['df10']
fig.show()
```


```python
#4.2-Kategori bazlı yıllık gelir analizi
import plotly.express as px

# df11 veri çerçevesini alıyoruz
df11 = dataframes['df11']

# Sütun isimlerini ve veri örneklerini yazdırma
print("Columns in df11:", df11.columns)
print("First few rows in df11:")
print(df11.head())

# Yıllık gelirleri işlemek için öncelikle yearly_income sütunundaki $ ve virgülleri kaldırıp sayısal formata dönüştürüyoruz
df11['yearly_income'] = df11['yearly_income'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Kategori bazında toplam yıllık gelirleri hesaplama
category_yearly_income = df11.groupby('category')['yearly_income'].sum().reset_index()

# Kategori bazında toplam yıllık gelirleri kontrol etme
print("Category yearly income data:")
print(category_yearly_income)

# Donut grafiği oluşturma
fig = px.pie(category_yearly_income, values='yearly_income', names='category', hole=0.3)

# Başlık ve yerleşim düzenlemeleri
fig.update_layout(
    title_text='Category-wise Yearly Income Distribution',
    title_x=0.5,  # Başlığı yatay olarak ortalar
    title_y=0.5,  # Başlığı dikey olarak ortalar (0.5, 0.5 grafiğin tam ortasıdır)
    title_font_size=24  # Başlık yazı tipi boyutu
)

# Grafiği gösterme
fig.show()
```



```python
# 4.3- En Fazla Yeniden Satın Alınan Ürünlerde İlk 8 ve Gelirleri

import plotly.express as px
# df12'ü alıyoruz
df12 = dataframes['df12']


# df12'yi alıyoruz
df12 = dataframes['df12']

# Yıllar sütununu tam sayıya çevirin
df12['year'] = df12['year'].astype(int)

# Yıllık gelir verilerini temizleyip sayısal formata dönüştürme
df12['yearly_income'] = df12['yearly_income'].replace('[\$,]', '', regex=True).astype(float)

# Küme çubuk grafiği oluşturma (dikey)
fig = px.bar(
    df12,
    x='year',  # X ekseni: yıl
    y='yearly_income',  # Y ekseni: yıllık gelir
    color='product_name',  # Renk gruplama: ürün adı
    barmode='group',  # Çubukları grupla
    title='Yearly Income by Product and Year',
    labels={'yearly_income': 'Yearly Income', 'year': 'Year'}
)

# Başlık ve etiketlerin düzenlenmesi
fig.update_layout(
    title_text='Yearly Income by Product and Year',
    title_x=0.5,
    xaxis_title='Year',
    yaxis_title='Yearly Income'
)

# Veri etiketlerini ekleme
fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')

# Grafiği göster
fig.show()

fig.write_html("yearly_income_by_product_and_year.html")
```


```python
# 5. RFM Analysis

import plotly.express as px
# df13'ü alıyoruz
df13 = dataframes['df13']
# Ağaç haritası oluşturma
fig = px.treemap(
    df13,
    path=['customer_id'],
    values='average_score',  # Burada 'total_score' ya da diğer RFM skorlarından birini de kullanabilirsin
    color='average_score',  # Ağaç haritasındaki renkler için bir metrik belirle
    hover_data=['recency_score', 'frequency_score', 'monetary_score', 'total_score'],
    color_continuous_scale='Blues'
)

# Grafik başlığı ve düzeni ayarlama
fig.update_layout(title='Müşteri RFM Skorları Ağaç Haritası')

# Grafiği gösterme
fig.show()
```


```python
# 6. Employees Analysis

# df15'i alıyoruz
df15 = dataframes['df15']

# Total revenue'yu float'a çevirme
df15['total_revenue'] = df15['total_revenue'].replace('[\$,]', '', regex=True).astype(float)

# Total revenue'yu küçükten büyüğe sıralıyoruz
df15_sorted = df15.sort_values(by='total_revenue')

# Scatter plot oluşturma
fig = px.scatter(
    df15_sorted,
    x='order_count',
    y='total_revenue',
    text='employee_id',
    title='Employee Order Count vs Total Revenue',
    labels={'order_count': 'Order Count', 'total_revenue': 'Total Revenue'}
)

# Formatlama işlevi
def format_currency(value):
    return f"{value:,.0f}"  # Binlik ayırıcı ile formatlama

# Veri etiketlerini formatlamak
fig.update_traces(
    texttemplate=df15_sorted['total_revenue'].apply(format_currency),
    textposition='top center'
)

fig.update_layout(
    xaxis_title="Order Count",
    yaxis_title="Total Revenue",
    yaxis=dict(range=[df15_sorted['total_revenue'].min(), df15_sorted['total_revenue'].max()])
)

fig.show()
```

