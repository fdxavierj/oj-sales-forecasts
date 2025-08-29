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

### plot of sales of the first shop, simply to get an overview of the series

# sales = df.iloc[:, -1].values
# vectors = [sales[i*102:(i+1)*102] for i in range(11)]
# time = np.arange(102)

# plt.figure(figsize=(10, 6))
# for i, vector in enumerate(vectors):
#     plt.plot(time, vector, label=f'Vector {i+1}')

# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('11 Vectors from Last Column vs Time')
# plt.legend()
# plt.grid(True)

# plt.show()
### end of plotting



### prepare data
def standardize_matrix(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def log_transform(data):
    return np.log1p(data)

def create_season_dummies(length):
    week_indices = np.arange(length) % 52
    season_labels = np.zeros(length, dtype=int)

    # Assign seasons
    season_labels[(week_indices >= 0) & (week_indices <= 12)] = 0  # Winter
    season_labels[(week_indices >= 13) & (week_indices <= 25)] = 1  # Spring
    season_labels[(week_indices >= 26) & (week_indices <= 38)] = 2  # Summer
    season_labels[(week_indices >= 39) & (week_indices <= 51)] = 3  # Fall

    # Convert to dummies (drop one column to avoid dummy trap)
    dummies = np.zeros((length, 3))
    for i in range(3):  # 0 = Winter, 1 = Spring, 2 = Summer
        dummies[:, i] = (season_labels == i + 1).astype(int)

    return dummies

products_per_store = 11
observations_per_store = 102

stores = df['store'].unique()
stores_data = [df[df['store'] == store] for store in stores]
price_columns = [col for col in df.columns if col.startswith('price')]

def remove_outliers(sales, feat, deal):
    # thresholds = [40000, 10000, 20000, 65000, 50000, 100000, 40000, 10000, 15000, 80000, 17500, 
    #               40000, 14000, 10000, 60000, 50000, 8000, 20000, 10000, 20000, 50000, 25000,
    #               100000, 15000, 10000, 150000, 100000, 100000, 50000, 20000, 50000, 150000, 40000,
    #               50000, 100000, 15000, 100000, 75000, 100000, 30000, 10000, 20000, 80000, 20000,
    #               ]

    

    sales_clean = sales.copy()

    # Iterate through all time-product pairs
    for t in range(sales.shape[0]):
        for p in range(sales.shape[1]):
            Q1 = np.quantile(sales[:,p], 0.25)
            Q3 = np.quantile(sales[:,p], 0.75)
            IQR = Q3 - Q1
            outlier = (sales[t,p] < Q1 - 4 * IQR) | (sales[t,p] > Q3 + 4 * IQR)

            if outlier:
                feat_val = feat[t, p]
                deal_val = deal[t, p]

                # Find indices with the same feat-deal combo, excluding the current one
                match_mask = (feat == feat_val) & (deal == deal_val)
                matched_sales = np.where(match_mask, sales, np.nan)
                replacement_value = np.nanmedian(matched_sales, axis=0)[p]

                sales_clean[t, p] = replacement_value

    return sales_clean


def process_store_data(store_df, price_columns, products_per_store, observations_per_store):
    prices = store_df.iloc[:observations_per_store][price_columns].values
    sales = store_df['sales'].values.reshape(products_per_store, observations_per_store).T
    deals = store_df['deal'].values.reshape(products_per_store, observations_per_store).T
    features = store_df['feat'].values.reshape(products_per_store, observations_per_store).T
    season_dummies = create_season_dummies(observations_per_store)

    return {
        'prices': prices,
        'prices_stand': standardize_matrix(prices),
        'sales': remove_outliers(sales, features, deals),
        'log_sales_stand': standardize_matrix(log_transform(remove_outliers(sales, features, deals))),
        'deals': deals,
        'features': features,
        'season_dummies': season_dummies
    }

final_store_data = []

for store_df in stores_data:
    store_processed = process_store_data(store_df, price_columns, products_per_store, observations_per_store)
    final_store_data.append(store_processed)



# Plot only Product 9 (index 8) from Store 10 (index 9)
store_index = 8  # Store 9
product_index = 9  # Product 10

store_data = final_store_data[store_index]
sales_series = store_data['sales'][:, product_index]

plt.figure(figsize=(8, 4))
plt.plot(sales_series)
plt.title(f'Store {store_index + 1} - Product {product_index + 1} Sales')
plt.xlabel('Time (Weeks)')
plt.ylabel('Sales')
plt.grid(True)
plt.tight_layout()
plt.show()



for store_index, store_data in enumerate(final_store_data):
    sales_matrix = store_data['sales']  # shape: (102, 11)
    
    # Plot each product's sales separately
    for product_index in range(sales_matrix.shape[1]):
        plt.figure(figsize=(8, 4))
        plt.plot(sales_matrix[:, product_index])
        plt.title(f'Store {store_index + 1} - Product {product_index + 1} Sales')
        plt.xlabel('Time (Weeks)')
        plt.ylabel('Sales')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
### end of data preparation 