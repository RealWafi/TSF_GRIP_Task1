# Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the data
data = pd.read_csv("http://bit.ly/w-data")

# Exploratory plot
data.plot(x="Hours", y="Scores", style="o")

# Scaling, formatting and saving the plot
plt.title("Hours Studied vs Score")
plt.xlabel("Hours Studied")
plt.ylabel("Test Score")
plt.xlim(0, 10)
plt.ylim(0, 100)
plt.savefig("plain plot.png", dpi=500)
plt.show()  # Making sure this plot is also shown, not just the last one

# Splitting data into training and test sets, with the test set being 20% of all data
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Performing ordinary least squares linear regression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Defining the regression line and plotting it along with the data points
line = regressor.coef_ * x + regressor.intercept_
plt.scatter(x, y)
plt.plot(x, line)
plt.title("Hours Studied vs Score, With Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Test Score")
plt.xlim(0, 10)
plt.ylim(0, 100)
plt.savefig("regression plot.png", dpi=500)

# Comparison table between the actual points in the test set and their predicted values
predicted = regressor.predict(x_test)
comparison = pd.DataFrame({"Actual": y_test, "Predicted": predicted})
print("\n" + str(comparison))

# Predicting the score if a student studies 9.25 hours
test_value = [[9.25]]
own_prediction = regressor.predict(test_value)
print("\nPredicted score for 9.25 hours: {}\n".format(round(own_prediction[0])))

# Calculating the mean absolute error, which gives an indication about the accuracy of the model
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, predicted))
