import numpy as np
import pandas as pd


def simulate_data(num_disease, num_control, disease_lambda_range, control_lambda_range, disease_binary_prob,
                  control_binary_prob, num_features_1, num_features_2, num_features_3):
    # Generate indices for individuals in both groups
    disease_indices = np.arange(num_disease)
    control_indices = np.arange(num_control)

    # Generate Poisson parameters for each individual
    disease_lambdas = np.random.uniform(low=disease_lambda_range[0], high=disease_lambda_range[1], size=num_disease)
    control_lambdas = np.random.uniform(low=control_lambda_range[0], high=control_lambda_range[1], size=num_control)

    # Sample instances for each individual in the disease group
    disease_data = []
    for i, lam in zip(disease_indices, disease_lambdas):
        num_instances = np.random.poisson(lam)
        disease_data.extend([(i, 'disease')] * num_instances)

    # Sample instances for each individual in the control group
    control_data = []
    for i, lam in zip(control_indices, control_lambdas):
        num_instances = np.random.poisson(lam)
        control_data.extend([(i, 'control')] * num_instances)

    # Create DataFrames for both groups
    disease_df = pd.DataFrame(disease_data, columns=['individual_index', 'group'])
    control_df = pd.DataFrame(control_data, columns=['individual_index', 'group'])

    # Adjust individual indices to ensure uniqueness across both groups
    control_df['individual_index'] += num_disease

    # Combine the data into a single DataFrame
    combined_df = pd.concat([disease_df, control_df], ignore_index=True)

    # Simulate binary variable
    disease_binary = np.random.choice([1, 2], size=len(disease_df), p=[disease_binary_prob, 1 - disease_binary_prob])
    control_binary = np.random.choice([1, 2], size=len(control_df), p=[control_binary_prob, 1 - control_binary_prob])

    # Add the binary variable to the DataFrame
    combined_df['binary_variable'] = np.concatenate([disease_binary, control_binary])

    # Simulate the first set of feature values based on the binary variable
    features_1 = []
    for _ in range(num_features_1):
        feature_values = []
        for binary_value in combined_df['binary_variable']:
            if binary_value == 1:
                n = np.random.poisson(5) + 1  # Poisson distribution for the first parameter, ensuring n > 0
                p = np.random.beta(5, 2)  # Beta distribution for the second parameter
            else:
                n = np.random.poisson(10) + 1  # Poisson distribution for the first parameter, ensuring n > 0
                p = np.random.beta(2, 5)  # Beta distribution for the second parameter
            feature_value = np.random.negative_binomial(n=n, p=p)
            feature_values.append(feature_value)
        features_1.append(feature_values)

    # Add the first set of feature values to the DataFrame
    for i, feature_values in enumerate(features_1):
        combined_df[f'feature_{i + 1}'] = feature_values

    # Simulate the second set of feature values based on the binary variable
    features_2 = []
    for _ in range(num_features_2):
        feature_values = []
        for binary_value in combined_df['binary_variable']:
            if binary_value == 1:
                n = np.random.poisson(10) + 1  # Poisson distribution for the first parameter, ensuring n > 0
                p = np.random.beta(2, 5)  # Beta distribution for the second parameter
            else:
                n = np.random.poisson(5) + 1  # Poisson distribution for the first parameter, ensuring n > 0
                p = np.random.beta(5, 2)  # Beta distribution for the second parameter
            feature_value = np.random.negative_binomial(n=n, p=p)
            feature_values.append(feature_value)
        features_2.append(feature_values)

    # Add the second set of feature values to the DataFrame
    for i, feature_values in enumerate(features_2):
        combined_df[f'feature_{num_features_1 + i + 1}'] = feature_values

    # Simulate the third set of feature values based on the binary variable
    features_3 = []
    for _ in range(num_features_3):
        feature_values = []
        for binary_value in combined_df['binary_variable']:
            if binary_value == 1:
                n = np.random.poisson(5) + 1  # Poisson distribution for the first parameter, ensuring n > 0
                p = np.random.beta(5, 2)  # Beta distribution for the second parameter
            else:
                n = np.random.poisson(5) + 1  # Poisson distribution for the first parameter, ensuring n > 0
                p = np.random.beta(5, 2)  # Beta distribution for the second parameter
            feature_value = np.random.negative_binomial(n=n, p=p)
            feature_values.append(feature_value)
        features_3.append(feature_values)

    # Add the second set of feature values to the DataFrame
    for i, feature_values in enumerate(features_3):
        combined_df[f'feature_{num_features_1 + num_features_2 + i + 1}'] = feature_values

    return combined_df


# Parameters
# num_disease = 100  # Number of disease individuals
# num_control = 100  # Number of control individuals
# disease_lambda_range = (5, 15)  # Range of Poisson parameters for disease group
# control_lambda_range = (3, 10)  # Range of Poisson parameters for control group
# disease_binary_prob = 0.7  # Probability of binary variable being 1 for disease group
# control_binary_prob = 0.4  # Probability of binary variable being 1 for control group
# num_features_1 = 50  # Number of features to simulate with lower mean for binary variable 1
# num_features_2 = 50  # Number of features to simulate with higher mean for binary variable 1
# num_features_3 = 900  # Number of features to simulate with same mean for binary variable 1

# Simulate data
# simulated_data = simulate_data(num_disease, num_control, disease_lambda_range, control_lambda_range,
#                                disease_binary_prob, control_binary_prob, num_features_1, num_features_2, num_features_3)
