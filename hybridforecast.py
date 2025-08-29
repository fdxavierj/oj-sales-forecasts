import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import statsmodels.api as sm

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
### end of data preparation 

### check for seasonality 
# for i in range(11):
#     plot_acf(sales_store_5[:,i], lags=52)
#     plt.show()
### end of seasonality check

### check for outliers
# for i in range(11):
#     prices_store_10.iloc[:, i].plot()
#     plt.show()
### end of outliers check


# ### Check for correlation between regressors

# regressors = pd.concat([pd.DataFrame(prices_store_1), pd.DataFrame(deals_store_1), pd.DataFrame(features_store_1)], axis=1)
# corr_matrix = regressors.corr()
# plt.figure(figsize=(14, 10))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
# plt.title("Correlation Matrix of Regressors")
# plt.tight_layout()
# plt.show()

# ### end of correlation check

############################################# Hybrid Subset Selection Method #############################################

def hybrid_stepwise_selection(X_train, y_train, verbose=True):
    # Get number of observations and predictors
    T, I = X_train.shape

    # List of all predictors not yet selected
    remaining = list(range(I))

    # List of predictors currently in the model
    selected = []

    # Start with infinitely bad aic
    best_aic = float('inf')

    # Improvement flag
    improved = True

    while improved and remaining:
        # Reset improvement flag
        improved = False

        # Store aics of candidates for this forward step
        candidate_aics = []

        # Try adding each remaining predictor one at a time
        for idx in remaining:
            # Add the current candidate predictor to the selected list
            predictors = selected + [idx]

            # Include intercept in the model
            X_candidate = X_train[:, predictors]

            # Fit OLS model
            model = sm.OLS(y_train, X_candidate).fit()

            # Compute aic using statsmodels built-in aic
            aic = model.aic

            # Save aic, candidate index, and model
            candidate_aics.append((aic, idx, model))

        # Sort candidates by aic
        candidate_aics.sort()
        #print(candidate_aics.item())

        # Get the best candidate based on aic
        best_add_aic, best_add_idx, best_model = candidate_aics[0]
        # print(best_add_aic)
        # print(best_aic)

        # If aic improves, accept the predictor
        if best_add_aic < best_aic:
            # Update selected and remaining lists
            selected.append(best_add_idx)
            remaining.remove(best_add_idx)

            # Update best aic
            best_aic = best_add_aic

            # Mark that improvement occurred
            improved = True

            # Print step details
            # if verbose:
            #     print(f"✅ Added predictor {best_add_idx}, aic: {best_aic:.3f}")

            # Backward elimination loop: try removing predictors
            while True:
                # Store aics from backward candidates
                delete_aics = []

                # Try removing each predictor one by one
                for idx in selected:
                    # Create reduced set without one predictor
                    reduced = [i for i in selected if i != idx]

                    # Fit model on reduced set
                    X_reduced = X_train[:, reduced]
                    #print(X_reduced)
                    if X_reduced.size == 0:
                        break

                    model = sm.OLS(y_train, X_reduced).fit()
                    aic = model.aic

                    # Store aic and predictor to potentially drop
                    delete_aics.append((aic, idx, model))
                if len(delete_aics) == 0:
                    break
                # Sort candidates for deletion by aic
                delete_aics.sort()

                # Get best candidate for removal
                best_del_aic, best_del_idx, del_model = delete_aics[0]
                # print(best_del_aic)
                # print(best_aic)

                # If removing improves aic, update selection
                if best_del_aic < best_aic:
                    selected.remove(best_del_idx)
                    remaining.append(best_del_idx)
                    best_aic = best_del_aic
                    improved = True

                else:
                    # Stop backward step if no further improvement
                    #print("No further improvement in backward step, stopping.")
                    break

    # Fit final model with all selected predictors
    final_model = sm.OLS(y_train, X_train[:, selected]).fit()

    return selected, final_model



### Hybrid Subset Selection regression with moving window

results = []

start_forecast = 3
end_forecast = 24
window_length = 70


for t in range(start_forecast, end_forecast + 1):
    for s in range(10): 
        store_s = final_store_data[s]
        print(t,s)

        for i in range(products_per_store):

            part1 = np.arange(102 - (window_length - (t - 1)), 102)
            part2 = np.arange(0, t - 1)
            training_indices = np.concatenate((part1, part2))
        
            prices = store_s['prices_stand'][training_indices, :]
            deals = store_s['deals'][training_indices, i].reshape(-1, 1)
            features = store_s['features'][training_indices, i].reshape(-1, 1)
            interaction = (deals * features)

            lag_1_index = training_indices - 1
            lag_2_index = training_indices - 2

            lag_1_sales = store_s['log_sales_stand'][lag_1_index, i].reshape(-1, 1)
            lag_2_sales = store_s['log_sales_stand'][lag_2_index, i].reshape(-1, 1)

            deal_lag = store_s['deals'][lag_1_index, i].reshape(-1, 1)
            feature_lag = store_s['features'][lag_1_index, i].reshape(-1, 1)
            interaction_lag = (deal_lag * feature_lag)

            seasonal_dummies = store_s['season_dummies'][training_indices, :]

            X_training = np.concatenate([
                prices, deals, features, interaction,
                lag_1_sales, lag_2_sales,
                deal_lag, feature_lag, interaction_lag,
                seasonal_dummies
            ], axis=1)

            y_training = store_s['log_sales_stand'][training_indices, i]

            selected, model = hybrid_stepwise_selection(X_training, y_training, verbose=False)

            x_next = np.concatenate([
                store_s['prices_stand'][t, :],
                [store_s['deals'][t, i]],
                [store_s['features'][t, i]],
                [(store_s['deals'][t, i] * store_s['features'][t, i])],
                [store_s['log_sales_stand'][(t - 1), i]],
                [store_s['log_sales_stand'][(t - 2), i]],
                [store_s['deals'][(t - 1), i]],
                [store_s['features'][(t - 1), i]],
                [(store_s['deals'][(t - 1), i] * store_s['features'][(t - 1), i])],
                store_s['season_dummies'][t, :]
            ])

            x_selected = x_next[selected].reshape(1, -1)
            y_pred = model.predict(x_selected)[0]
            y_true = store_s['log_sales_stand'][t + window_length+1, i]

            # Duan smearing estimator
            y_fitted = model.fittedvalues
            residuals = y_training - y_fitted

            mean_log_sales = np.mean(log_transform(store_s['sales'][:, i]))
            std_log_sales = np.std(log_transform(store_s['sales'][:, i]))

            y_pred_unstand = y_pred * std_log_sales + mean_log_sales
            y_fitted_unstand = y_fitted * std_log_sales + mean_log_sales
            residuals_unstand = y_training * std_log_sales + mean_log_sales - y_fitted_unstand

            smearing_factor = np.mean(np.exp(residuals_unstand))

            y_pred_sales = np.exp(y_pred_unstand) * smearing_factor
            y_true_sales = store_s['sales'][t, i]


            results.append({
                'store': s,
                'product': i,
                'time': t,
                'y_pred': y_pred.item(),
                'y_true': store_s['log_sales_stand'][t, i],
                'num_predictors': len(selected),
                'selected': selected,
                'y_pred_sales': y_pred_sales.item(),
                'y_true_sales': y_true_sales.item()
            })

start_second_part = 2
end_second_part = 31

for t in range(start_second_part, end_second_part):
    for s in range(10):
        print(t,s)
        store_s = final_store_data[s]
        for i in range(products_per_store):
            prices = store_s['prices_stand'][t:t+window_length, :]                    
            deals = store_s['deals'][t:t+window_length, i].reshape(-1, 1)
            features = store_s['features'][t:t+window_length, i].reshape(-1, 1)
            interaction = (store_s['deals'][t:t+window_length, i] * store_s['features'][t:t+window_length, i]).reshape(-1, 1)

            lag_1_sales = store_s['log_sales_stand'][t-1:t+window_length-1, i].reshape(-1, 1)   
            lag_2_sales = store_s['log_sales_stand'][t-2:t+window_length-2, i].reshape(-1, 1)    

            deal_lag = store_s['deals'][t-1:t+window_length-1, i].reshape(-1, 1)          
            feature_lag = store_s['features'][t-1:t+window_length-1, i].reshape(-1, 1)
            interaction_lag = deal_lag * feature_lag

            seasonal_dummies = store_s['season_dummies'][t:t+window_length, :]

            X_training = np.concatenate([prices, deals, features, interaction,
                                      lag_1_sales, lag_2_sales, deal_lag,
                                      feature_lag, interaction_lag, seasonal_dummies], axis=1)
            y_training = store_s['log_sales_stand'][t:t+window_length, i]


            selected, model = hybrid_stepwise_selection(X_training, y_training, verbose=False)

            x_next = np.concatenate([
            store_s['prices_stand'][t+window_length+1, :],
            [store_s['deals'][t+window_length+1, i]],
            [store_s['features'][t+window_length+1, i]],
            [store_s['deals'][t+window_length+1, i] * store_s['features'][t+window_length+1, i]],
            [store_s['log_sales_stand'][t+window_length, i]],
            [store_s['log_sales_stand'][t+window_length-1, i]],
            [store_s['deals'][t+window_length, i]],
            [store_s['features'][t+window_length, i]],
            [store_s['deals'][t+window_length, i] * store_s['features'][t+window_length, i]],
            store_s['season_dummies'][t + window_length + 1, :]
            ])

            x_selected = x_next[selected].reshape(1, -1)
            y_pred = model.predict(x_selected)[0]
            y_true = store_s['log_sales_stand'][t + window_length+1, i]

            
            # Duan smearing estimator

            y_fitted = model.fittedvalues

            residuals = y_training - y_fitted


            mean_log_sales = np.mean(log_transform(store_s['sales'][:, i]))
            std_log_sales = np.std(log_transform(store_s['sales'][:, i]))

            y_pred_unstand = y_pred * std_log_sales + mean_log_sales
            y_fitted_unstand = y_fitted * std_log_sales + mean_log_sales
            residuals_unstand = y_training * std_log_sales + mean_log_sales - y_fitted_unstand

            smearing_factor = np.mean(np.exp(residuals_unstand))

            y_pred_sales = np.exp(y_pred_unstand) * smearing_factor
            y_true_sales = store_s['sales'][t + window_length+1, i]
            
            results.append({
                'store': s,
                'product': i,
                'time': t + window_length + 1,
                'y_pred': y_pred,
                'y_true': y_true,
                'num_predictors': len(selected),
                'selected': selected,
                'y_pred_sales': y_pred_sales,
                'y_true_sales': y_true_sales,
            })

### Save results
df_results = pd.DataFrame(results)
df_results.to_csv("hybrid.csv", index=False)

### end of moving window forecasting