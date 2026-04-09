import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

housing_data = pd.read_csv('./Housing.csv')
housing_data.head()
housing_data.info()
housing_data.describe(include='all')
housing_data.isnull().sum()
housing_data.duplicated().sum()
housing_data.nunique()

cat_col = housing_data.select_dtypes(include=['str']).columns
print(f"Categorical columns: {list(cat_col)}")

# converting categorical values to numeric
# mainroad: No=1  Yes=0
housing_data['mainroad'].unique()
# guestroom: No=0  Yes=1
housing_data['guestroom'].unique()
# basement: No=0  Yes=1
housing_data['basement'].unique()
# hotwaterheating: No=0  Yes=1
housing_data['hotwaterheating'].unique()
# airconditioning: NO=1  Yes=0
housing_data['airconditioning'].unique()
# prefarea: NO=1  Yes=0
housing_data['prefarea'].unique()
# furnishingstatus: 0=furnished 1=semi-furnished 2=unfurnished
housing_data['furnishingstatus'].unique()

for col in cat_col:
  print(col)
  print((housing_data[col].unique()), list(range(housing_data[col].nunique())))
  
  housing_data[col] = housing_data[col].replace(housing_data[col].unique(), list(range(housing_data[col].nunique())))
  
print(housing_data.head())

# Training data: x_train is the input,
# y_train is the target Price (dollars)
X_train = np.array(housing_data.drop('price', axis=1).values)
y_train = np.array(housing_data['price'].values)
print(f"x_train = {X_train}")
print(f"y_train = {y_train}")

# Convert X_train to float to ensure math operations work
X_train = X_train.astype(float)

# Feature Scaling: Z-Score Normalization
def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def compute_cost(X, y, w, b): 
    """
    Computes the cost function for linear regression.
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter  
    
    Returns
        cost (scalar): cost
    """
    # number of training examples
    m = X.shape[0] 
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost


def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing


X_train, mu, sigma = zscore_normalize_features(X_train)

w_init = np.zeros(X_train.shape[1])
b_init = 0

iterations = 1000
alpha = 0.01

# Running gradient descent to fit the model
w_final, b_final, J_hist = gradient_descent(X_train ,y_train, w_init, b_init, compute_cost, compute_gradient, alpha, iterations)

print(f"(w,b) found by gradient descent: ({w_final},{b_final})")

# Plot cost versus iteration
plt.figure(figsize=(10, 6))
plt.plot(J_hist)
plt.title("Cost Function J per Iteration")
plt.ylabel('Cost')
plt.xlabel('Iteration Step')
plt.grid(True)
plt.show()

# Predicted vs. Actual Prices Plot
# Get predictions for the entire training set
y_pred = np.dot(X_train, w_final) + b_final
# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred, alpha=0.5, color='teal')
# Plot the "Perfect Prediction" line
max_val = max(max(y_train), max(y_pred))
min_val = min(min(y_train), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', lw=2, linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Housing Prices")
plt.grid(True)
plt.show()


feature_names = housing_data.drop('price', axis=1).columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, w_final, color='skyblue')
plt.xlabel("Weight Value (Importance)")
plt.title("Impact of Each Feature on House Price")
plt.show()


# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_train - y_pred) / y_train)) * 100
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
print(f"Model Accuracy (100 - MAPE): {100 - mape:.2f}%")


def predict_house_price(raw_data, w, b, mu, sigma):
    """
    Takes raw house features, normalizes them, and returns a predicted price.
    
    Args:
      raw_data (list or np.array): The 12 features of the house
      w, b: Your trained model parameters
      mu, sigma: Mean and StdDev from your training set
    """
    # 1. Convert to numpy array if it isn't one
    x_input = np.array(raw_data)
    
    # 2. Normalize using training statistics (Crucial!)
    x_norm = (x_input - mu) / sigma
    
    # 3. Compute prediction: y = wx + b
    prediction = np.dot(x_norm, w) + b
    
    return prediction

# --- TESTING IT ---

my_house1 = [7500, 4, 2, 2, 1, 1, 1, 0, 1, 2, 1, 0]
my_house2 = [3000, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2]

price1 = predict_house_price(my_house1, w_final, b_final, mu, sigma)
price2 = predict_house_price(my_house2, w_final, b_final, mu, sigma)

print(f"--- Prediction Result ---")
print(f"Estimated Market Value House1: ${price1:,.2f}")
print(f"Estimated Market Value House2: ${price2:,.2f}")