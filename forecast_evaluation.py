import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

df = pd.read_csv("OrangeJuiceQ42.csv")

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


def volatility_plot():
    model_list = ["hybrid", "ridge", "lasso", "elasticNet", "xgboost"]
    
    for model in model_list:
        if model == "lasso" or model == "elasticNet":
            df2 = pd.read_csv(f"{model}.csv")
        else:
            df2 = pd.read_csv(f"{model}.csv", sep=";")
        standard_deviations = []
        forecast_errors = []

        for s in range(10):
            store_s = final_store_data[s]
            store_predictions = df2[df2["store"] == s]

            for i in range(products_per_store):
                sales_sku_i = store_s["sales"][:,i].reshape(-1,1)
                st_dev = np.std(sales_sku_i)
                standard_deviations.append(st_dev)
                product_predictions = store_predictions[store_predictions["product"] == i]
                errors = product_predictions.y_pred_sales - product_predictions.y_true_sales
                avg_errors = np.mean(errors)
                forecast_errors.append(avg_errors)
        
        plt.scatter(standard_deviations, forecast_errors)
        plt.title(f"{model.capitalize()} Model")
        plt.xlabel("Volatilities")
        plt.ylabel("Forecast errors")
        plt.savefig(f"Volatility_vs_forecast_{model}.jpg", format='jpg', dpi=300)
        plt.show()


def comparison_across_stores():
    model_list = ["hybrid", "ridge", "lasso", "elasticNet", "xgboost"]

    for model in model_list:
        if model == "lasso" or model == "elasticNet":
            df2 = pd.read_csv(f"{model}.csv")
        else:
            df2 = pd.read_csv(f"{model}.csv", sep=";")
        
        df2["loss"] = (df2["y_pred_sales"] - df2["y_true_sales"]).abs()
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="store", y="loss", data=df2)
        plt.xlabel("Store Number")
        plt.ylabel("Loss Value (Absolute Error)")
        plt.title(f"Performance Across Stores for {model.capitalize()} Model")
        plt.savefig(f"Performance_across_stores_{model}.jpg", format='jpg', dpi=300)
        plt.show()


def comparison_across_SKUs():
    model_list = ["hybrid", "ridge", "lasso", "elasticNet", "xgboost"]

    for model in model_list:
        if model == "lasso" or model == "elasticNet":
            df2 = pd.read_csv(f"{model}.csv")
        else:
            df2 = pd.read_csv(f"{model}.csv", sep=";")
        
        df2["loss"] = (df2["y_pred_sales"] - df2["y_true_sales"]).abs()
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="product", y="loss", data=df2)
        plt.xlabel("SKU")
        plt.ylabel("Loss Value (Absolute Error)")
        plt.title(f"Performance Across SKUs for {model.capitalize()} Model")
        plt.savefig(f"Performance_across_sku__{model}.jpg", format='jpg', dpi=300)
        plt.show()


def promotion_vs_prediction_difficulty():
    model_list = ["hybrid", "ridge", "lasso", "elasticNet", "xgboost"]
    
    for model in model_list:
        
        if model == "lasso" or model == "elasticNet":
            df2 = pd.read_csv(f"{model}.csv")
        else:
            df2 = pd.read_csv(f"{model}.csv", sep=";")

        percentage_features_deals = []
        forecast_errors = []

        for s in range(10):
            store_s = final_store_data[s]
            store_predictions = df2[df2["store"] == s]

            for i in range(products_per_store):
                product_predictions = store_predictions[store_predictions["product"] == i]
                errors = product_predictions.y_pred_sales - product_predictions.y_true_sales
                avg_errors = np.mean(errors)
                forecast_errors.append(avg_errors)
                features_deals = store_s["deals"][:,i] + store_s["features"][:,i]
                mean_features_deals = np.mean(features_deals)
                percentage_features_deals.append(mean_features_deals)
        
        plt.scatter(percentage_features_deals, forecast_errors)
        plt.xlabel("Ratio of promo/deals")
        plt.ylabel("Forecast errors")
        plt.title(f"{model.capitalize()} Model")
        plt.savefig(f"Promo_deal_ratio_{model}.jpg", format='jpg', dpi=300)
        plt.show()


def predictor_usage():
    predictors_list = ["hybrid_predictors", "lasso_predictors"]
    predictor_names = [...] # lijst met namen van predictors nog ff aanpassen zodra ik het van Leo heb

    for model in predictors_list:
        df = pd.read_csv(f"{model}.csv", sep=";")
        df2 = df.groupby(["product"])[predictor_names].mean()
        
        for i in range(products_per_store):
            plt.bar(predictor_names, df2.iloc[i], color='skyblue', edgecolor='black')
            plt.xlabel(f'Predictors for SKU "{predictor_names[i]}')
            plt.ylabel('Percent Usage')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.show()


# volatility_plot()
# comparison_across_SKUs()
# comparison_across_stores()
# promotion_vs_prediction_difficulty()
# predictor_usage()