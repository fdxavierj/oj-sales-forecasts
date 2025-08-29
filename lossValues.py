import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
import csv
import matplotlib.pyplot as plt


df_rw = pd.read_csv(
    "randomWalk.csv",
    names=["store","product","time","y_randomWalk"],
    skiprows=1  # skip original header row
)
for k in ["store","product","time"]:
    df_rw[k] = df_rw[k].astype(str)

df_rw = df_rw.sort_values(["store","product","time"])

seasons = ['fall', 'winter', 'spring', 'summer']

def assign_season(time_val):
    idx = (int(time_val) // 13) % 4
    return seasons[idx]


def transform_raw_results(model_name):
    # 1) peek header
    fname = f"{model_name}.csv"
   
    with open(fname, "r") as f:
            header_line = f.readline()
    
    try:
        dialect = csv.Sniffer().sniff(header_line, delimiters=[",",";"])
        sep = dialect.delimiter
    except csv.Error:
        
        sep = ","   # fallback if sniffer can’t decide

    # 2) Split on the detected sep
    header = header_line.strip().split(sep)
    first_three = header[:3]
    last_two    = header[-2:]
    use_names   = first_three + last_two

    # 3) Read only those columns, using the same sep
    raw = pd.read_csv(
        fname,
        sep=sep,
        usecols=use_names
    )

    # print(raw)

    # now raw.columns == first_three + last_two
    # coerce keys to string
    for k in ["store","product","time"]:
        raw[k] = raw[k].astype(str)
    raw = raw.sort_values(["store","product","time"])

    # merge — same as before
    results = pd.merge(raw, df_rw,
                       on=["store","product","time"],
                       how="left")

    # strip off any [ ] in y_pred_sales
    results["y_pred_sales"] = (
        results["y_pred_sales"]
          .astype(str)
          .str.strip()
          .str.replace(r"[\[\]]", "", regex=True)
    )

    # numeric conversion
    results["y_true_sales"] = pd.to_numeric(results["y_true_sales"], errors="coerce")
    results["y_pred_sales"] = pd.to_numeric(results["y_pred_sales"], errors="coerce")
    results["y_randomWalk"] = pd.to_numeric(results["y_randomWalk"], errors="coerce")

    #convert time to numeric
    results["time"] = pd.to_numeric(results["time"], errors="coerce")

    results["season"] = results["time"].astype(int).apply(assign_season)


    # final NaN check
    if results.isnull().any().any():
        bad = results[results.isnull().any(axis=1)]
        print("Rows with NaN values:\n", bad)
        raise ValueError(f"Data contains NaNs for model '{model_name}'")

    return results


# Calculate loss values PER MODEL
def mse(g):
    return np.mean((g.y_true_sales - g.y_pred_sales)**2)

def rel_rmse(g):
    rmse_m = np.sqrt(np.mean((g.y_true_sales - g.y_pred_sales)**2))
    rmse_b = np.sqrt(np.mean((g.y_true_sales - g.y_randomWalk)**2))
    return rmse_m / rmse_b

def mspe(g):
    return np.mean(((g.y_true_sales - g.y_pred_sales) / g.y_true_sales)**2)

def qlike(g):
    r = g.y_true_sales / g.y_pred_sales
    return np.mean(r - np.log(r) - 1)

def loss_calculator(data, group_cols=["store","product"]):
    loss_df = (
        data
        .groupby(group_cols)
        .apply(lambda g: pd.Series({
            "mse":      mse(g),
            "rel_rmse": rel_rmse(g),
            "mspe":     mspe(g),
            "qlike":    qlike(g)
        }))
        .reset_index()
    )

    # print(loss_df)

    return loss_df


def get_loss_tables(modelList):
    result_dict = {f"{model}_results": loss_calculator(transform_raw_results(model)) for model in modelList}

    # Combine all model‐specific loss_dfs into one long DataFrame and calculate the ranks and aggregate these ranks 
    loss_tables = []
    for model_name, loss_df in result_dict.items():
        tmp = loss_df.copy()
        tmp["model"] = model_name.replace("_results","")   # clean up the name
        loss_tables.append(tmp)

    all_losses = pd.concat(loss_tables, ignore_index=True)

    # print(all_losses)



    metrics = ["mse", "rel_rmse", "mspe", "qlike"]

    for m in metrics:
        all_losses[f"{m}_rank"] = (
            all_losses
            .groupby(["store", "product"])[m]
            .rank(method="dense", ascending=True)
        )

    avg_rank_per_store = (
        all_losses
        .groupby(["store", "model"])[[f"{m}_rank" for m in metrics]]
        .mean()
        .reset_index()
    )

    avg_rank_per_product = (
        all_losses
        .groupby(["product", "model"])[[f"{m}_rank" for m in metrics]]
        .mean()
        .reset_index()
    )

    global_avg_rank = (
        all_losses
        .groupby("model")[[f"{m}_rank" for m in metrics]]
        .mean()
    )

    print("▶ Average rank per store (first few rows):\n", avg_rank_per_store, "\n")
    print("▶ Avg rank per product (first few rows):\n", avg_rank_per_product, "\n")
    print("▶ Global avg rank over all store×product:\n", global_avg_rank)

    return avg_rank_per_store, avg_rank_per_product, global_avg_rank

# average rank by season
def get_avg_rank_by_season(modelList):
    loss_tables = []
    for model_name in modelList:
        df = transform_raw_results(model_name)
        loss_df = loss_calculator(df, group_cols=["season", "store", "product"])
        loss_df["model"] = model_name
        loss_tables.append(loss_df)

    all_losses = pd.concat(loss_tables, ignore_index=True)

    # Rank within each (season, store, product) group
    metrics = ["mse", "rel_rmse", "mspe", "qlike"]
    for m in metrics:
        all_losses[f"{m}_rank"] = (
            all_losses
            .groupby(["season", "store", "product"])[m]
            .rank(method="dense", ascending=True)
        )

    # Average ranks across (season, model)
    avg_rank_per_season = (
        all_losses
        .groupby(["season", "model"])[[f"{m}_rank" for m in metrics]]
        .mean()
        .reset_index()
    )

    avg_rank_per_season["avg_rank"] = avg_rank_per_season[
        [f"{m}_rank" for m in metrics]
    ].mean(axis=1)

    return avg_rank_per_season

##########################################################################################
# GETTING THE SCORING TABLES
##########################################################################################

# will use squared forecast erros like in the paper
def panel_dm_test(dfA, dfB, alpha=0.05):
    # merge and compute d_it
    merged = (
      dfA[['store','product','time','y_pred_sales','y_true_sales']]
        .rename(columns={'y_pred_sales':'pred_A','y_true_sales':'y'})
      .merge(
        dfB[['store','product','time','y_pred_sales']],
        on=['store','product','time']
      )
        .rename(columns={'y_pred_sales':'pred_B'})
    )
    
    merged['d_it'] = (merged.y - merged.pred_A)**2 - (merged.y - merged.pred_B)**2

    #NOTE ???? what if there is no store or product in merged?
    # would it be different if we want SEASONALITY 
    n = merged[['store','product']].drop_duplicates().shape[0]

    # this will be the average across the grouping we choose
    R_t = merged.groupby('time')['d_it'].mean() * np.sqrt(n)

    # Newey-West variance
    T = len(R_t)
    L = int(np.floor(T**(1/3)))
    rt = R_t.values
    mu = rt.mean()
    gamma0 = np.mean((rt - mu)**2)
    gamma = sum(
      (1 - l/(L+1)) *
      np.cov(rt[l:], rt[:-l], bias=True)[0,1]
      for l in range(1, L+1)
    )
    sigma = np.sqrt(gamma0 + 2*gamma)

    # test statistic & one-sided p
    J = np.sqrt(T) * (mu) / sigma
    p_one = norm.cdf(J)  
    return J, p_one

def compute_scores(all_data, model_names, group_col, alpha=0.05, universal=False):
    if universal:
        # universal test across all data
        scores = {m: 0 for m in model_names}
        for m1, m2 in itertools.combinations(model_names, 2):
            df1, df2 = all_data[m1], all_data[m2]
            J, p = panel_dm_test(df1, df2, alpha=alpha)
            if p < alpha:
                # model with lower mean d_it wins
                winner = m1 if J < 0 else m2
                scores[winner] += 1

                loser = m1 if J > 0 else m2
                scores[loser] -= 1

        return pd.DataFrame(scores.items(), columns=['model', 'score'])
    
    else:
        groups = all_data[model_names[0]][group_col].unique()
        scores = {(g,m): 0 for g in groups for m in model_names}

        for m1, m2 in itertools.combinations(model_names, 2):
            df1, df2 = all_data[m1], all_data[m2]
            for g in groups:
                sub1 = df1[df1[group_col]==g]
                sub2 = df2[df2[group_col]==g]
                J, p = panel_dm_test(sub1, sub2, alpha=alpha)
                if p < alpha:
                    # model with lower mean d_it wins
                    winner = m1 if J < 0 else m2
                    scores[(g, winner)] += 1

                    # think of adding a loser count
                    loser = m1 if J > 0 else m2
                    scores[(g, loser)] -= 1

        # to DataFrame
        return (
        pd.DataFrame([
            {group_col: g, 'model': m, 'score': s}
            for (g,m), s in scores.items()
        ])
        )


##################################################################################
# LET'S RUN 
##################################################################################

modelList = ["ridge", "lasso","elasticNet","hybrid","xgboost"]
all_data = { m: transform_raw_results(m) for m in modelList }

# Still need to properly think of how to handle the time column and ADD IN SEASONALITY
# Assign seasonality: time 0-12 is 'fall', then every next 13 values is the next season in order
seasons = ['fall', 'winter', 'spring', 'summer']

def assign_season(time_val):
    idx = (int(time_val) // 13) % 4
    return seasons[idx]

all_data = {
    m: df.assign(season=df['time'].astype(int).apply(assign_season))
    for m, df in all_data.items()
}

by_all = compute_scores(all_data, modelList, group_col=None, universal=True)
by_store   = compute_scores(all_data, modelList, group_col='store')
by_product = compute_scores(all_data, modelList, group_col='product')
by_season  = compute_scores(all_data, modelList, group_col='season')


# 4) merge or inspect
print("By all:\n", by_all)
# print("By store:\n", by_store)
# print("By product:\n", by_product)
# print("By season:\n", by_season)

##################################################################
# RUN for loss values
##################################################################

# modelList = ["hybrid", "ridge", "lasso", "elasticNet", "xgboost"]
# # modelList = ["ridge", "hybrid"]
# print(transform_raw_results("ridge"))
# get_loss_tables(modelList)

#####################################################################
# VIUSLALIZATION
#####################################################################
modelList = ["hybrid", "ridge", "lasso", "elasticNet", "xgboost"]
# modelList = ["ridge", "hybrid"]
print(transform_raw_results("ridge"))
avg_rank_per_store, avg_rank_per_product, global_avg_rank = get_loss_tables(modelList)

avg_rank_per_season = get_avg_rank_by_season(modelList)


modelList = ["ridge", "lasso","elasticNet","hybrid","xgboost"]
all_data = { m: transform_raw_results(m) for m in modelList }

# Still need to properly think of how to handle the time column and ADD IN SEASONALITY
# Assign seasonality: time 0-12 is 'fall', then every next 13 values is the next season in order
seasons = ['fall', 'winter', 'spring', 'summer']

def assign_season(time_val):
    idx = (int(time_val) // 13) % 4
    return seasons[idx]

all_data = {
    m: df.assign(season=df['time'].astype(int).apply(assign_season))
    for m, df in all_data.items()
}

by_all = compute_scores(all_data, modelList, group_col=None, universal=True)
by_store   = compute_scores(all_data, modelList, group_col='store')
by_product = compute_scores(all_data, modelList, group_col='product')
by_season  = compute_scores(all_data, modelList, group_col='season')

# for store/product, take the mean across the four "_rank" cols
# BE CAREFUL WITH THE AVERAGING OUT.....
rank_cols = [c for c in avg_rank_per_store.columns if c.endswith("_rank")]
avg_rank_per_store["avg_rank"] = avg_rank_per_store[rank_cols].mean(axis=1)

rank_cols = [c for c in avg_rank_per_product.columns if c.endswith("_rank")]
avg_rank_per_product["avg_rank"] = avg_rank_per_product[rank_cols].mean(axis=1)

# global_avg_rank is a DataFrame indexed by model with the four rank‐cols
global_df = global_avg_rank.reset_index().rename(columns={"index":"model"})
global_df["avg_rank"] = global_df[rank_cols].mean(axis=1)

# create a 1×5 DataFrame for the global average ranks:
global_heat = pd.DataFrame([ global_df.set_index('model')['avg_rank'] ])
global_heat.index = ['Global']   # single row labeled “Global”

heatmap_data = {
    'Universal DM Score':   by_all.assign(All='All').pivot(index='All',   columns='model', values='score'),
    'Score by Store':       by_store.pivot(index='store',   columns='model', values='score'),
    'Score by Product':     by_product.pivot(index='product', columns='model', values='score'),
    'Score by Season':      by_season.pivot(index='season',  columns='model', values='score'),
    'Avg Rank by Store':    avg_rank_per_store.pivot(index='store',   columns='model', values='avg_rank'),
    'Avg Rank by Product':  avg_rank_per_product.pivot(index='product', columns='model', values='avg_rank'),
    'Global Average Rank':  global_heat, 
    'Avg Rank by Season': avg_rank_per_season.pivot(index='season', columns='model', values='avg_rank')
}

for title, dfh in heatmap_data.items():
    fig, ax = plt.subplots(
        figsize=( max(6, dfh.shape[1]*0.5),
                  max(2, dfh.shape[0]*0.4) )
    )
    im = ax.imshow(dfh.values)
    ax.set_xticks(range(dfh.shape[1]))
    ax.set_xticklabels(dfh.columns, rotation=45, ha='right')
    ax.set_yticks(range(dfh.shape[0]))
    ax.set_yticklabels(dfh.index.astype(str))
    ax.set_title(title)
    plt.colorbar(im, ax=ax, orientation='vertical', label='Value')
    plt.tight_layout()

plt.show()