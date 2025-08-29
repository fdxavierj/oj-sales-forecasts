import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv('OrangeJuiceQ42.csv')
print(df["sales"])

## prepare data
def standardize_matrix(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def log_transform(data):
    return np.log1p(data)

products_per_store = 11
observations_per_store = 102

stores = df['store'].unique()
stores_data = [df[df['store'] == store] for store in stores]
price_columns = [col for col in df.columns if col.startswith('price')]

def process_store_data(store_df, price_columns, products_per_store, observations_per_store):
    prices = store_df.iloc[:observations_per_store][price_columns].values
    sales = store_df['sales'].values.reshape(products_per_store, observations_per_store).T
    deals = store_df['deal'].values.reshape(products_per_store, observations_per_store).T
    features = store_df['feat'].values.reshape(products_per_store, observations_per_store).T

    return {
        'prices': prices,
        'prices_stand': standardize_matrix(prices),
        'sales': sales,
        'log_sales_stand': standardize_matrix(log_transform(sales)),
        'deals': deals,
        'features': features
    }

final_store_data = []

for store_df in stores_data:
    store_processed = process_store_data(store_df, price_columns, products_per_store, observations_per_store)
    final_store_data.append(store_processed)


# Specify the brand index (e.g., i = 0 for the first brand)
brand_index = 10 # Change this to select a different brand (0 to 10, since products_per_store = 11)

# Collect sales for brand i across all stores
total_sales_brand_i = []
total_prices_brand_i = []
for store_data in final_store_data:
    sales_brand_i = store_data["sales"][:, brand_index]  # Get sales for brand i (column i)
    total_sales_brand_i.append(sales_brand_i)
    prices_brand_i = store_data["prices"][:, brand_index]  # Get sales for brand i (column i)
    total_prices_brand_i.append(prices_brand_i)

# Concatenate sales into a single vector
total_sales_brand_i = np.concatenate(total_sales_brand_i)
total_prices_brand_i = np.concatenate(total_prices_brand_i)

# Compute the average
average_sales_brand_i = np.mean(total_sales_brand_i)
st_dev_sales_brand_i = np.std(total_sales_brand_i)

average_prices_brand_i = np.mean(total_prices_brand_i)
st_dev_prices_brand_i = np.std(total_prices_brand_i)

# Print results
print(f"Sales vector for brand {brand_index}:", total_sales_brand_i)
print(f"Average sales for brand {brand_index}:", average_sales_brand_i)
print(f"Standard dev of sales for brand {brand_index}:", st_dev_sales_brand_i)

print(f"Average price for brand {brand_index}:", average_prices_brand_i)
print(f"Standard dev of price for brand {brand_index}:", st_dev_prices_brand_i)

# Start date: first week of September 1989 (e.g. September 1st)
start_date = '1989-09-01'

# Generate weekly dates for 102 weeks
dates = pd.date_range(start=start_date, periods=102, freq='W')

# Create a DataFrame
season_df = pd.DataFrame({'date': dates})
season_df['month'] = season_df['date'].dt.month

# Define seasons:
# Spring: March (3) to May (5)
# Summer: June (6) to August (8)
# Autumn: September (9) to November (11)
# Winter: December (12), January (1), February (2)

season_df['spring'] = season_df['month'].isin([3, 4, 5]).astype(int)
season_df['summer'] = season_df['month'].isin([6, 7, 8]).astype(int)
season_df['autumn'] = season_df['month'].isin([9, 10, 11]).astype(int)
season_df['winter'] = season_df['month'].isin([12, 1, 2]).astype(int)

# Optional: remove the 'month' column
season_df = season_df.drop(columns='month')
# for s in range(10):
#     for i in range(products_per_store):
#         # Get the sales vector (length 102)
#         sales = final_store_data[s]["sales"][:, i]


#         season_df['sales'] = sales



#         # Aggregate total or average sales per season
#         seasonal_sales = {
#             'Spring': season_df.loc[season_df['spring'] == 1, 'sales'].mean(),
#             'Summer': season_df.loc[season_df['summer'] == 1, 'sales'].mean(),
#             'Autumn': season_df.loc[season_df['autumn'] == 1, 'sales'].mean(),
#             'Winter': season_df.loc[season_df['winter'] == 1, 'sales'].mean()
#         }

#         # Plot
#         plt.bar(seasonal_sales.keys(), seasonal_sales.values(), color=['#77c1f0', '#f4d35e', '#ff7f50', '#b5ead7'])
#         plt.ylabel("Average Weekly Sales")
#         plt.title(f"Average Sales per Season (Product {i}, Store {s})")
#         plt.grid(axis='y', linestyle='--', alpha=0.5)
#         plt.tight_layout()
#         plt.show()

#         promotions = final_store_data[s]["deals"][:, i]

#         # Add to the existing season_df
#         season_df['promotion'] = promotions

#         # Count number of weeks with promotion per season
#         promo_counts = {
#             'Spring': season_df.loc[season_df['spring'] == 1, 'promotion'].sum(),
#             'Summer': season_df.loc[season_df['summer'] == 1, 'promotion'].sum(),
#             'Autumn': season_df.loc[season_df['autumn'] == 1, 'promotion'].sum(),
#             'Winter': season_df.loc[season_df['winter'] == 1, 'promotion'].sum()
#         }

# Assuming final_store_data and season_df are available from the provided code
products_per_store = 11
num_stores = 10

# Aggregate sales across all stores for each product
seasonal_sales = np.zeros((products_per_store, 4))  # Columns: Spring, Summer, Autumn, Winter
for p in range(products_per_store):
    product_sales = np.zeros(102)
    for s in range(num_stores):
        product_sales += final_store_data[s]["sales"][:, p]
    
    season_df['sales'] = product_sales  # Assign once per product
    
    seasonal_sales[p, 0] = season_df.loc[season_df['spring'] == 1, 'sales'].sum()
    seasonal_sales[p, 1] = season_df.loc[season_df['summer'] == 1, 'sales'].sum()
    seasonal_sales[p, 2] = season_df.loc[season_df['autumn'] == 1, 'sales'].sum()
    seasonal_sales[p, 3] = season_df.loc[season_df['winter'] == 1, 'sales'].sum()

# Compute proportions
total_sales_per_product = seasonal_sales.sum(axis=1, keepdims=True)
seasonal_proportions = seasonal_sales / total_sales_per_product  # Ensure normalization

# Verify proportions sum to 1 (for debugging)
print("Proportions sum check:", np.sum(seasonal_proportions, axis=1))  # Should be close to 1 for each product

# Create stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.5
products = range(products_per_store)
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
colors = ['#d9d9d9', '#a6a6a6', '#737373', '#404040']  # Grayscale-friendly

bottom = np.zeros(products_per_store)
for i, season in enumerate(seasons):
    ax.bar(products, seasonal_proportions[:, i], bar_width, bottom=bottom, 
           label=season, color=colors[i])
    bottom += seasonal_proportions[:, i]

# Customize the plot
ax.set_xlabel('Product')
ax.set_ylabel('Proportion of Total Sales')
ax.set_title('')
ax.set_xticks(products)
ax.set_xticklabels([f'P{i+1}' for i in range(products_per_store)])
ax.legend(title='Season', loc='upper right')
ax.set_ylim(0, 1)  # Ensure y-axis is capped at 1
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('corrected_seasonal_sales_proportion.png', dpi=300, bbox_inches='tight')
plt.show()