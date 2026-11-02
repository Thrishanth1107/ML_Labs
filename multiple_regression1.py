import pandas as pd
import numpy as np

# ==========================================
# 1. READ DATASET
# ==========================================

df = pd.read_excel("iris.xlsx")

print("Dataset loaded successfully!")
print("Number of samples:", len(df))


# ==========================================
# 2. CONVERT SPECIES INTO NUMBERS
# ==========================================

df["Species_Code"] = df["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
})


# ==========================================
# 3. INPUT FEATURES
# ==========================================

X = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
].values


# ==========================================
# 4. TARGET VARIABLE
# ==========================================

Y = df["Species_Code"].values


# ==========================================
# 5. ADD INTERCEPT COLUMN
# ==========================================

X = np.column_stack((np.ones(len(X)), X))


# ==========================================
# 6. LINEAR REGRESSION
# ==========================================

# Normal Equation:
# beta = (X^T X)^-1 X^T Y

beta = np.linalg.pinv(X.T @ X) @ X.T @ Y


# ==========================================
# 7. DISPLAY MODEL COEFFICIENTS
# ==========================================

print("\n==========================================")
print("LINEAR REGRESSION MODEL")
print("==========================================")

print("Intercept:", beta[0])
print("Sepal Length coefficient:", beta[1])
print("Sepal Width coefficient:", beta[2])
print("Petal Length coefficient:", beta[3])
print("Petal Width coefficient:", beta[4])


# ==========================================
# 8. FUNCTION TO CONVERT NUMBER TO SPECIES
# ==========================================

def get_species(value):

    # Find nearest class
    value = int(round(value))

    if value <= 0:
        return "Iris-setosa"

    elif value == 1:
        return "Iris-versicolor"

    else:
        return "Iris-virginica"


# ==========================================
# 9. TAKE USER INPUT
# ==========================================

print("\n==========================================")
print("ENTER FLOWER MEASUREMENTS")
print("==========================================")

sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))


# ==========================================
# 10. CREATE NEW FLOWER
# ==========================================

new_flower = np.array([
    1,
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
])


# ==========================================
# 11. PREDICT SPECIES CODE
# ==========================================

prediction = new_flower @ beta


# ==========================================
# 12. CONVERT CODE TO SPECIES
# ==========================================

predicted_species = get_species(prediction)


# ==========================================
# 13. DISPLAY RESULT
# ==========================================

print("\n==========================================")
print("PREDICTION RESULT")
print("==========================================")

print("Predicted numerical value:", prediction)
print("Predicted Species:", predicted_species)

print("==========================================")