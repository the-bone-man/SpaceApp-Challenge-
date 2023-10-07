import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'dataset.csv' with your dataset file)
data = pd.read_csv("C:\\Users\\Bijaya KC\\Documents\\pyy\\try.csv")

# Split the data into features (X) and the target variable (y)
X = data[['Latitude','Longitude','Wind_Speed','Wind_Direction','Elevation','Soil_Moisture','Precipitation','Temp_Celsius']]
y = data['Fire_Occurrence']

# Split the data into training and testing sets (adjust test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(report)

new_data = pd.DataFrame([['35.129','-120.462','4.2','220','830','12.6','0.0','36.0']],
                        columns=['Latitude', 'Longitude', 'Wind_Speed', 'Wind_Direction', 'Elevation', 'Soil_Moisture', 'Precipitation', 'Temp_Celsius'])

prediction = model.predict(new_data)
print(prediction)
if prediction[0] == 1:
    print("There is a risk of a forest fire.")
else:
    print(" no risk of a forest fire.")

# You can now use the trained model for making predictions on new data.
