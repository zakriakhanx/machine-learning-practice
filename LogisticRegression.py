import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

titanic_data = pd.read_csv('./train.csv')

titanic_data.head()
titanic_data.info()
titanic_data.describe(include='all')
titanic_data.isnull().sum()
titanic_data.duplicated().sum()
titanic_data.nunique()

# Handling missing values
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
titanic_data.drop(columns=['Cabin'], inplace=True)
titanic_data.drop(columns=['Name', 'Ticket'], inplace=True)

# Encoding categorical variables
print(f"Categorical columns: {list(titanic_data.select_dtypes(include=['str']).columns)}")

# Sex: male = 0, female = 1
titanic_data['Sex'].unique()
titanic_data['Sex'] = titanic_data['Sex'].replace(titanic_data['Sex'].unique(), list(range(titanic_data['Sex'].nunique())))

# Embarked: S = 0, C = 1, Q = 2
titanic_data['Embarked'].unique()
titanic_data['Embarked'] = titanic_data['Embarked'].replace(titanic_data['Embarked'].unique(), list(range(titanic_data['Embarked'].nunique())))


# Feature scaling: Z-score normalization
def zscore_normalize_features(X):
    X = np.array(X, dtype=np.float64)   
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    # Avoid division by zero if sigma is 0
    sigma[sigma == 0] = 1
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


# Sigmoid function or Logistic function
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g

# Cost Function
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost

# Compute Gradient
def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar
        
    return dj_db, dj_dw

# Gradient Descent
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history


X_train = np.array(titanic_data.drop(columns=['Survived']))
y_train = np.array(titanic_data['Survived'])

#normalize features
X_train, mu, sigma = zscore_normalize_features(X_train)

# Initialize fitting parameters
initial_w = np.zeros(X_train.shape[1])
initial_b = 0.

iterations = 2000
alpha = 0.01

w, b, J_history = gradient_descent(X_train, y_train, initial_w, initial_b, alpha, iterations)

print(f"w: {w} b: {b}")


#plot cost vs iteration
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs. Iteration")
plt.show()


feature_names = titanic_data.drop(columns=['Survived']).columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, w, color='skyblue')
plt.xlabel("Weight Value (Importance)")
plt.title("Impact of Each Feature on Survival Prediction")
plt.show()


# Predict
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters w and b
    
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      
    Returns:
      p (ndarray (m,)): Predictions for X using a threshold at 0.5 
    """
    m, n = X.shape
    p = np.zeros(m)
    
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        p[i] = 1 if f_wb_i >= 0.5 else 0

    return p


# Get predictions
y_pred = predict(X_train, w, b)

# Calculate accuracy
accuracy = np.mean(y_pred == y_train) * 100
print(f"Training Accuracy: {accuracy:.2f}%")


def predict_passenger(raw_data, w, b, mu, sigma):
    """
    Takes raw passenger features, normalizes them, and returns a survival prediction.
    
    Args:
      raw_data (list or np.array): The features of the passenger
      w, b: Your trained model parameters
      mu, sigma: Mean and StdDev from your training set
    """
    # Convert to numpy array
    x_input = np.array(raw_data)
    
    # Normalize using training statistics
    x_norm = (x_input - mu) / sigma
    
    # Compute prediction probability
    probability = sigmoid(np.dot(x_norm, w) + b)
    
    # Classify based on threshold of 0.5
    prediction = 1 if probability >= 0.5 else 0
    
    return prediction, probability


# Columns: PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
passenger1 = [1, 3, 0, 22.0, 1, 0, 7.25, 0]  # Male, 3rd class, age 22
passenger2 = [2, 1, 1, 38.0, 1, 0, 71.28, 1] # Female, 1st class, age 38

pred1, prob1 = predict_passenger(passenger1, w, b, mu, sigma)
pred2, prob2 = predict_passenger(passenger2, w, b, mu, sigma)

print(f"--- Prediction Result ---")
print(f"Passenger 1: {'Survived' if pred1 == 1 else 'Did not survive'} (Probability: {prob1:.4f})")
print(f"Passenger 2: {'Survived' if pred2 == 1 else 'Did not survive'} (Probability: {prob2:.4f})")