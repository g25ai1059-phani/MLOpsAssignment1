from sklearn.tree import DecisionTreeRegressor

from misc import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model
)

# Load data
df = load_data()

# Split data
X_train, X_test, y_train, y_test = preprocess_data(df)

# Create model
model = DecisionTreeRegressor()

# Train model
model = train_model(model, X_train, y_train)

# Evaluate
mse = evaluate_model(model, X_test, y_test)

print("Decision Tree MSE:", mse)