#max accuracy : 34.17%
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# Initialize SQLite database
conn = sqlite3.connect('../../music_data.db')

# Load data from SQLite into a DataFrame
query = """
SELECT 
    t.title, 
    a.name as artist_name, 
    t.popularity, 
    t.danceability, 
    t.energy, 
    t.loudness, 
    t.speechiness, 
    t.acousticness, 
    t.instrumentalness, 
    t.liveness, 
    t.valence, 
    t.tempo,
    t.tags,
    a.genres_spotify, 
    a.genres_lastfm, 
    t.tier
FROM 
    tracks t 
JOIN 
    artists a 
ON 
    t.artist_id = a.artist_id
WHERE 
    t.tier IS NOT NULL
"""
df = pd.read_sql_query(query, conn)

# Data Preprocessing

# Encode the artist_name with LabelEncoder
label_encoder = LabelEncoder()
df['artist_name_encoded'] = label_encoder.fit_transform(df['artist_name'])

# Split the genres/tags into individual items and one-hot encode them
def split_and_one_hot_encode(df, column):
    # Split the column by comma and expand into multiple rows
    genres_expanded = df[column].str.get_dummies(sep=', ')
    return genres_expanded

genres_spotify_expanded = split_and_one_hot_encode(df, 'genres_spotify')
genres_lastfm_expanded = split_and_one_hot_encode(df, 'genres_lastfm')
genres_lastfm_expanded = split_and_one_hot_encode(df, 'tags')

# Combine the original dataframe with the one-hot encoded genres/tags
df = pd.concat([df, genres_spotify_expanded, genres_lastfm_expanded], axis=1)

# Drop the original genre/tag columns
df.drop(columns=['genres_spotify', 'genres_lastfm'], inplace=True)

# Encode the tier as a numeric value
df['tier'] = df['tier'].map({'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0})

# Drop columns not used as features
df.drop(columns=['title',  'artist_name'], inplace=True)

# Define features and target variable
X = df.drop(columns=['tier'])
y = df['tier']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Save the feature names used in training
X_train_columns = X_train.columns
np.save('X_train_columns.npy', X_train_columns)

# Save the trained model to disk
joblib.dump(rf, 'random_forest_classifier.pkl')

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the classifier
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Feature Importance
importance = rf.feature_importances_
features = X.columns
indices = np.argsort(importance)[::-1]


param_grid = {
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}


grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearch
print("Best Parameters:", grid_search.best_params_)

# Save the best model to disk
joblib.dump(grid_search.best_estimator_, 'best_random_forest_classifier.pkl')
# Save the LabelEncoder used for artist names
np.save('label_encoder_classes.npy', label_encoder.classes_)

# Predict with the best model
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

# Re-evaluate the best classifier
print("Classification Report (Best Model):\n", classification_report(y_test, y_pred_best))
print("Confusion Matrix (Best Model):\n", confusion_matrix(y_test, y_pred_best))
print("Accuracy Score (Best Model):", accuracy_score(y_test, y_pred_best))

# Plot Feature Importance of the best model
best_importance = best_rf.feature_importances_
best_indices = np.argsort(best_importance)[::-1]

plt.figure(figsize=(10, 8))
plt.title("Feature Importance (Best Model)")
plt.bar(range(X.shape[1]), best_importance[best_indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in best_indices], rotation=90)
plt.tight_layout()
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Close the database connection
conn.close()
