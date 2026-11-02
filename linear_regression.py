import pandas as pd
import numpy as np

# ==========================================
# 1. Read Excel dataset
# ==========================================

df = pd.read_excel("iris.xlsx")

print("Dataset loaded successfully!")


# ==========================================
# 2. Convert Species into numbers
# ==========================================

df["Species_Code"] = df["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
})


# ==========================================
# 3. Select ONE input variable
# ==========================================

X = df["PetalLengthCm"].values

# Target variable
Y = df["Species_Code"].values


# ==========================================
# 4. Calculate means
# ==========================================

X_mean = np.mean(X)
Y_mean = np.mean(Y)


# ==========================================
# 5. Calculate slope
# ==========================================

numerator = np.sum((X - X_mean) * (Y - Y_mean))

denominator = np.sum((X - X_mean) ** 2)

b1 = numerator / denominator


# ==========================================
# 6. Calculate intercept
# ==========================================

b0 = Y_mean - b1 * X_mean


# ==========================================
# 7. Display equation
# ==========================================

print("\n================================")
print("SIMPLE LINEAR REGRESSION")
print("================================")

print("Intercept (b0):", b0)
print("Slope (b1):", b1)

print("\nRegression Equation:")
print("Species_Code =", b0, "+", b1, "* PetalLengthCm")


# ==========================================
# 8. Function to predict species
# ==========================================

def predict_species(value):

    value = round(value)

    if value <= 0:
        return "Iris-setosa"

    elif value == 1:
        return "Iris-versicolor"

    else:
        return "Iris-virginica"


# ==========================================
# 9. Take input from user
# ==========================================

print("\n================================")
print("ENTER FLOWER INFORMATION")
print("================================")

petal_length = float(input("Enter Petal Length: "))


# ==========================================
# 10. Predict
# ==========================================

prediction = b0 + b1 * petal_length


# ==========================================
# 11. Convert prediction to species
# ==========================================

species = predict_species(prediction)


# ==========================================
# 12. Display result
# ==========================================

print("\n================================")
print("PREDICTION")
print("================================")

print("Predicted numerical value:", prediction)
print("Predicted Species:", species)

print("================================")