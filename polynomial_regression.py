import pandas as pd
import numpy as np

# ==========================================
# 1. READ EXCEL DATASET
# ==========================================

df = pd.read_excel("iris.xlsx")

print("Dataset loaded successfully!")


# ==========================================
# 2. CONVERT SPECIES TO NUMBERS
# ==========================================

df["Species_Code"] = df["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
})


# ==========================================
# 3. TAKE 2 INPUT FEATURES
# ==========================================

X1 = df["PetalLengthCm"].values
X2 = df["PetalWidthCm"].values

# Target
Y = df["Species_Code"].values


# ==========================================
# 4. CREATE POLYNOMIAL FEATURES
# ==========================================

# Degree 2:
#
# 1
# X1
# X2
# X1^2
# X2^2

X_poly = np.column_stack([
    np.ones(len(X1)),
    X1,
    X2,
    X1 ** 2,
    X2 ** 2
])


# ==========================================
# 5. CALCULATE COEFFICIENTS
# ==========================================

# Normal Equation
#
# B = (X^T X)^-1 X^T Y

B = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ Y


# ==========================================
# 6. DISPLAY COEFFICIENTS
# ==========================================

print("\n==========================================")
print("POLYNOMIAL REGRESSION")
print("==========================================")

print("b0 =", B[0])
print("b1 =", B[1])
print("b2 =", B[2])
print("b3 =", B[3])
print("b4 =", B[4])


# ==========================================
# 7. DISPLAY EQUATION
# ==========================================

print("\nPolynomial Equation:")

print(
    "Y =",
    B[0],
    "+", B[1], "* PetalLength",
    "+", B[2], "* PetalWidth",
    "+", B[3], "* PetalLength^2",
    "+", B[4], "* PetalWidth^2"
)


# ==========================================
# 8. SPECIES PREDICTION FUNCTION
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
# 9. USER INPUT
# ==========================================

print("\n==========================================")
print("ENTER FLOWER INFORMATION")
print("==========================================")

petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))


# ==========================================
# 10. CREATE POLYNOMIAL INPUT
# ==========================================

new_X = np.array([
    1,
    petal_length,
    petal_width,
    petal_length ** 2,
    petal_width ** 2
])


# ==========================================
# 11. PREDICTION
# ==========================================

prediction = new_X @ B


# ==========================================
# 12. CONVERT PREDICTION TO SPECIES
# ==========================================

species = predict_species(prediction)


# ==========================================
# 13. DISPLAY RESULT
# ==========================================

print("\n==========================================")
print("PREDICTION RESULT")
print("==========================================")

print("Petal Length:", petal_length)
print("Petal Width:", petal_width)

print("Predicted numerical value:", prediction)
print("Predicted Species:", species)

print("==========================================")