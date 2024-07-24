import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split



def prepare_image_dataset(
    x_csv = 'data/table/filtered_X_cols_cleaned.csv',
    y_csv = 'data/table/final_Y.csv',
    test_size = 0.2,
    random_state = 42,
    device = None
):
    
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Load data
    df_X = pd.read_csv(x_csv)
    df_Y = pd.read_csv(y_csv)
    
    # Check for missing values in the 'path' column of df_X
    missing_path_indices = df_X[df_X['path'].isnull()].index

    # Drop rows with missing 'path' values from both df_X and df_Y
    df_X = df_X.drop(missing_path_indices)
    df_Y = df_Y.drop(missing_path_indices)

    # Encode labels
    y_encoder = LabelEncoder()
    df_Y['GT'] = y_encoder.fit_transform(df_Y['GT'])

    # Split IDs into train and test based on NACCID (Patients)
    train_ids, test_ids = train_test_split(df_X['NACCID'].unique(), test_size=test_size, random_state=random_state)
        
    # Filter the datasets based on the split IDs
    train_df_X = df_X[df_X['NACCID'].isin(train_ids)]
    train_df_Y = df_Y[df_Y['NACCID'].isin(train_ids)]

    test_df_X = df_X[df_X['NACCID'].isin(test_ids)]
    test_df_Y = df_Y[df_Y['NACCID'].isin(test_ids)]
    
    y_train = torch.tensor(train_df_Y['GT'].values, dtype=torch.long)
    y_test = torch.tensor(test_df_Y['GT'].values, dtype=torch.long)

    train_image_paths = train_df_X[['path']]
    test_image_paths = test_df_X[['path']]

    return train_image_paths, y_train, test_image_paths, y_test



def prepare_tabular_dataset(
    x_csv = 'data/table/filtered_X_cols_cleaned.csv',
    x_template = 'data/table/filtered_X_cols_template.json',
    y_csv = 'data/table/final_Y.csv',
    test_size = 0.2,
    random_state = 42,
    device = None,
    drop_missing_images = False,
):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load JSON metadata
    with open(x_template, 'r') as file:
        column_metadata = json.load(file)

    # Load data
    df_X = pd.read_csv(x_csv)
    df_Y = pd.read_csv(y_csv)
    
    if drop_missing_images:
        # Check for missing values in the 'path' column of df_X
        missing_path_indices = df_X[df_X['path'].isnull()].index

        # Drop rows with missing 'path' values from both df_X and df_Y
        df_X = df_X.drop(missing_path_indices)
        df_Y = df_Y.drop(missing_path_indices)

    # Define categorical and continuous columns
    categorical_columns = [col for col, info in column_metadata.items() if info['type'] in ['C', 'O']]
    continuous_columns = [col for col, info in column_metadata.items() if info['type'] == 'N']

    # Handle categorical data
    # Create a mask for categorical data
    cat_mask = (df_X[categorical_columns].isnull()).astype(int)
    df_X[categorical_columns] = df_X[categorical_columns].fillna(-1)  # Fill NaNs with -1 for categorical data
    encoders = {col: LabelEncoder().fit(df_X[col]) for col in categorical_columns}
    for col in categorical_columns:
        df_X[col] = encoders[col].transform(df_X[col])

    # Handle continuous data
    # Create a mask for numerical data
    num_mask = (df_X[continuous_columns].isnull()).astype(int)
    for col in continuous_columns:
        df_X[col] = df_X[col].fillna(df_X[col].mean())  # Fill NaNs with the mean for continuous data

    # Encode labels
    y_encoder = LabelEncoder()
    df_Y['GT'] = y_encoder.fit_transform(df_Y['GT'])

    # Split IDs into train and test based on NACCID (Patients)
    train_ids, test_ids = train_test_split(df_X['NACCID'].unique(), test_size=test_size, random_state=random_state)

    # Filter the datasets based on the split IDs
    train_df_X = df_X[df_X['NACCID'].isin(train_ids)]
    train_df_Y = df_Y[df_Y['NACCID'].isin(train_ids)]
    train_cat_mask = cat_mask[df_X['NACCID'].isin(train_ids)]
    train_num_mask = num_mask[df_X['NACCID'].isin(train_ids)]

    test_df_X = df_X[df_X['NACCID'].isin(test_ids)]
    test_df_Y = df_Y[df_Y['NACCID'].isin(test_ids)]
    test_cat_mask = cat_mask[df_X['NACCID'].isin(test_ids)]
    test_num_mask = num_mask[df_X['NACCID'].isin(test_ids)]

    # Remove NACCID column
    train_df_X = train_df_X.drop(['NACCID', 'path', 'NACCMRFI'], axis=1, errors='ignore')
    test_df_X = test_df_X.drop(['NACCID', 'path', 'NACCMRFI'], axis=1, errors='ignore')

    # Split categorical and continuous data
    train_df_X_cat = train_df_X[categorical_columns]
    train_df_X_cont = train_df_X[continuous_columns]
    test_df_X_cat = test_df_X[categorical_columns]
    test_df_X_cont = test_df_X[continuous_columns]

    # Convert DataFrames to tensors for training set
    X_train_cat = torch.tensor(train_df_X_cat.values, dtype=torch.long, device=device)
    X_train_cont = torch.tensor(train_df_X_cont.values, dtype=torch.float, device=device)
    y_train = torch.tensor(train_df_Y['GT'].values, dtype=torch.long, device=device)
    cat_mask_train = torch.tensor(train_cat_mask[categorical_columns].values, dtype=torch.long, device=device)
    num_mask_train = torch.tensor(train_num_mask[continuous_columns].values, dtype=torch.long, device=device)

    # Convert DataFrames to tensors for testing set
    X_test_cat = torch.tensor(test_df_X_cat.values, dtype=torch.long, device=device)
    X_test_cont = torch.tensor(test_df_X_cont.values, dtype=torch.float, device=device)
    y_test = torch.tensor(test_df_Y['GT'].values, dtype=torch.long, device=device)
    cat_mask_test = torch.tensor(test_cat_mask[categorical_columns].values, dtype=torch.long, device=device)
    num_mask_test = torch.tensor(test_num_mask[continuous_columns].values, dtype=torch.long, device=device)
    
    num_categories = [len(encoder.classes_) for encoder in encoders.values()]
    num_continuous = len(continuous_columns)
        
    return X_train_cat, X_train_cont, y_train, cat_mask_train, num_mask_train, X_test_cat, \
            X_test_cont, y_test, cat_mask_test, num_mask_test, num_categories, num_continuous, categorical_columns, continuous_columns



def prepare_combined_dataset(
    x_csv = 'data/table/filtered_X_cols_cleaned.csv',
    x_template = 'data/table/filtered_X_cols_template.json',
    y_csv = 'data/table/final_Y.csv',
    test_size = 0.2,
    random_state = 42,
):
    # Load JSON metadata
    with open(x_template, 'r') as file:
        column_metadata = json.load(file)

    # Load data
    df_X = pd.read_csv(x_csv)
    df_Y = pd.read_csv(y_csv)
    
    # Check for missing values in the 'path' column of df_X
    missing_path_indices = df_X[df_X['path'].isnull()].index

    # Drop rows with missing 'path' values from both df_X and df_Y
    df_X = df_X.drop(missing_path_indices)
    df_Y = df_Y.drop(missing_path_indices)

    # Define categorical and continuous columns
    categorical_columns = [col for col, info in column_metadata.items() if info['type'] in ['C', 'O']]
    continuous_columns = [col for col, info in column_metadata.items() if info['type'] == 'N']

    # Handle categorical data
    # Create a mask for categorical data
    cat_mask = (df_X[categorical_columns].isnull()).astype(int)
    df_X[categorical_columns] = df_X[categorical_columns].fillna(-1)  # Fill NaNs with -1 for categorical data
    encoders = {col: LabelEncoder().fit(df_X[col]) for col in categorical_columns}
    for col in categorical_columns:
        df_X[col] = encoders[col].transform(df_X[col])

    # Handle continuous data
    # Create a mask for numerical data
    num_mask = (df_X[continuous_columns].isnull()).astype(int)
    for col in continuous_columns:
        df_X[col] = df_X[col].fillna(df_X[col].mean())  # Fill NaNs with the mean for continuous data

    # Encode labels
    y_encoder = LabelEncoder()
    df_Y['GT'] = y_encoder.fit_transform(df_Y['GT'])

    # Split IDs into train and test based on NACCID (Patients)
    train_ids, test_ids = train_test_split(df_X['NACCID'].unique(), test_size=test_size, random_state=random_state)

    # Filter the datasets based on the split IDs
    train_df_X = df_X[df_X['NACCID'].isin(train_ids)]
    train_df_Y = df_Y[df_Y['NACCID'].isin(train_ids)]
    train_cat_mask = cat_mask[df_X['NACCID'].isin(train_ids)]
    train_num_mask = num_mask[df_X['NACCID'].isin(train_ids)]

    test_df_X = df_X[df_X['NACCID'].isin(test_ids)]
    test_df_Y = df_Y[df_Y['NACCID'].isin(test_ids)]
    test_cat_mask = cat_mask[df_X['NACCID'].isin(test_ids)]
    test_num_mask = num_mask[df_X['NACCID'].isin(test_ids)]

    train_image_paths = train_df_X[['path']]
    test_image_paths = test_df_X[['path']]
    
    # Remove NACCID column
    train_df_X = train_df_X.drop(['NACCID', 'path', 'NACCMRFI'], axis=1, errors='ignore')
    test_df_X = test_df_X.drop(['NACCID', 'path', 'NACCMRFI'], axis=1, errors='ignore')

    # Split categorical and continuous data
    train_df_X_cat = train_df_X[categorical_columns]
    train_df_X_cont = train_df_X[continuous_columns]
    test_df_X_cat = test_df_X[categorical_columns]
    test_df_X_cont = test_df_X[continuous_columns]

    # Convert DataFrames to tensors for training set
    X_train_cat = torch.tensor(train_df_X_cat.values, dtype=torch.long)
    X_train_cont = torch.tensor(train_df_X_cont.values, dtype=torch.float)
    y_train = torch.tensor(train_df_Y['GT'].values, dtype=torch.long)
    cat_mask_train = torch.tensor(train_cat_mask[categorical_columns].values, dtype=torch.long)
    num_mask_train = torch.tensor(train_num_mask[continuous_columns].values, dtype=torch.long)

    # Convert DataFrames to tensors for testing set
    X_test_cat = torch.tensor(test_df_X_cat.values, dtype=torch.long)
    X_test_cont = torch.tensor(test_df_X_cont.values, dtype=torch.float)
    y_test = torch.tensor(test_df_Y['GT'].values, dtype=torch.long)
    cat_mask_test = torch.tensor(test_cat_mask[categorical_columns].values, dtype=torch.long)
    num_mask_test = torch.tensor(test_num_mask[continuous_columns].values, dtype=torch.long)

    num_categories = [len(encoder.classes_) for encoder in encoders.values()]
    num_continuous = len(continuous_columns)
    
    return train_image_paths, X_train_cat, X_train_cont, y_train, cat_mask_train, num_mask_train, test_image_paths, \
            X_test_cat, X_test_cont, y_test, cat_mask_test, num_mask_test, num_categories, num_continuous, categorical_columns, continuous_columns
