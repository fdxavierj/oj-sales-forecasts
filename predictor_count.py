import pandas as pd
import ast


def extract_csv(csv_file, model_name):
    df = pd.read_csv(csv_file)

    if model_name == "lasso":
        if 'selected' not in df.columns:
            raise ValueError("The CSV file must contain a 'predictors' column.")

        print(f"Type of 'selected' column: {type(df['selected'].iloc[0])}")

        df["selected"] = df['selected'].apply(lambda x: x[:1] + x[1:].replace(' ', '', 1) if isinstance(x, str) and len(x) > 1 and x[1] == ' ' else x)

        df["selected"] = df['selected'].apply(lambda x: (x.replace('  ', ',')) if isinstance(x, str) else x)
        df["selected"] = df['selected'].apply(lambda x: (x.replace(' ', ',')) if isinstance(x, str) else x)

    df["selected"] = df['selected'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    print(df)
    
    return df, model_name

def extract_predictor_count(df, model_name):

    num_predictions = 29

    exploded = df.explode("selected")
    print(exploded)

    exploded['count'] = 1

    df_test = exploded.groupby(["store", "product", "selected"]).sum().reset_index()

    print(df_test)

    df_test['count'] = df_test['count'] / num_predictions
    
    pivot = df_test.pivot_table(
        index=['store', 'product'],
        columns='selected',
        values='count',
        fill_value=0   # missing combinations → 0 times selected
    )

    pivot = pivot.reset_index()
    print(pivot)

    # Rename the columns to predictor names
    predictor_names = ['price1', 'price2', 'price3', 'price4', 'price5', 'price6', 'price7', 'price8', 'price9', 'price10', 'price11', 
                       'deal', 'feature', 'interaction', 'lag_1_sales', 'lag_2_sales',
                       'deal_lag', 'feature_lag', 'interaction_lag']
    pivot.columns = ['store', 'product'] + predictor_names
    print(pivot)

    # Save the pivot table to a CSV file
    # pivot.to_csv(f"{model_name}_predictors.csv", index=False)
    return []

df, model_name = extract_csv("lasso.csv", "lasso")
extract_predictor_count(df, model_name)