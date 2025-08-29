import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


######################################################## Deal/Promo Trends ########################################################

# df = pd.read_csv('OrangeJuiceQ42.csv') 

# #grouped = df.groupby(['store', 'brand', 'week'])['deal'].sum().reset_index()
# grouped = df.groupby(['store', 'brand', 'week'])['feat'].sum().reset_index()
# stores = grouped['store'].unique()

# for store in stores:
#     plt.figure(figsize=(10,6))
#     store_data = grouped[grouped['store'] == store]
#     #sns.lineplot(data=store_data, x='week', y='deal', hue='brand', marker='o')
#     #plt.title(f'Deal Value Over Time by Brand - Store {store}')
#     sns.lineplot(data=store_data, x='week', y='feat', hue='brand', marker='o')
#     plt.title(f'Feature Value Over Time by Brand - Store {store}')

#     plt.xlabel('Week')
#     #plt.ylabel('Deal Value')
#     plt.ylabel('Feature Value')
#     #plt.legend(title='Brand')
#     plt.legend(title='Feature')
#     plt.tight_layout()
#     plt.show()



# ######################################################## Store/SKU-Combo ########################################################

# df_results = pd.read_csv('hybrid.csv')
# group = df_results[(df_results['store'] == 9) & (df_results['product'] == 5)]

# # Sort by time
# group_sorted = group.sort_values('time')

# # Plot
# plt.figure(figsize=(10, 4))
# plt.plot(group_sorted['time'], group_sorted['y_true'], label='y_true', linewidth=2)
# plt.plot(group_sorted['time'], group_sorted['y_pred'], label='y_pred', linewidth=2, linestyle='--')

# plt.title('Store 9, Product 5')
# plt.xlabel('Time')
# plt.ylabel('Log Sales (Standardized)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()




# ######################################################## All Store/SKU-Combos ########################################################

df_results = pd.read_csv('hybrid.csv')
grouped = df_results.groupby(['store', 'product'])

for (store, product), group in grouped:
    group_sorted = group.sort_values('time')

    plt.figure(figsize=(10, 4))
    plt.plot(group_sorted['time'], group_sorted['y_true_sales'], label='y_true', linewidth=2)
    plt.plot(group_sorted['time'], group_sorted['y_pred_sales'], label='y_pred', linewidth=2, linestyle='--')

    plt.title(f'Store {store}, Product {product}')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()