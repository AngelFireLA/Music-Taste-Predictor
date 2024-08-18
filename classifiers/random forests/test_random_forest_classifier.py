import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from music_data import get_combined_data

# Load the trained model
model = joblib.load('cv 15  updated parameters  test acc 0.35  acc 0.47.pkl')

# Load the LabelEncoder used in the training
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

# Load the feature names used in training
X_train_columns = np.load('X_train_columns.npy', allow_pickle=True)


# Function to split and one-hot encode genres/tags
def split_and_one_hot_encode(df, column):
    genres_expanded = df[column].str.get_dummies(sep=', ')
    return genres_expanded



# Example function to predict tier for a new song
def predict_tier_for_new_song(track_name, artist_name):
    # Get the combined data for the new song
    combined_data = get_combined_data(track_name, artist_name)

    if not combined_data:
        print("Failed to retrieve data for the song.")
        return

    # Prepare the data in the same format as training data
    features = pd.DataFrame([combined_data])

    # Add a check to handle unseen artist names
    def encode_artist_name(artist_name):
        if artist_name in label_encoder.classes_:
            return label_encoder.transform([artist_name])[0]
        else:
            # Handle unseen artist by assigning a default value (e.g., the most frequent artist label)
            # Or you can assign a new label, such as len(label_encoder.classes_) (requires model retraining)
            print(f"Artist '{artist_name}' not seen during training. Encoding as 'Unknown Artist'.")
            return -1  # You can use any default value here

    # Encode the artist_name with the same LabelEncoder used during training
    features['artist_name_encoded'] = encode_artist_name(artist_name)

    # One-hot encode the genres and tags
    genres_spotify_expanded = split_and_one_hot_encode(features, 'artist_genres_spotify')
    genres_lastfm_expanded = split_and_one_hot_encode(features, 'artist_tags')

    # Combine the original dataframe with the one-hot encoded genres/tags
    features = pd.concat([features, genres_spotify_expanded, genres_lastfm_expanded], axis=1)

    # Drop columns not used as features
    features.drop(columns=['track_name', 'artist_name', 'artist_genres_spotify', 'artist_tags'], inplace=True)

    # Ensure all necessary columns are present
    missing_cols = set(X_train_columns) - set(features.columns)
    for col in missing_cols:
        features[col] = 0

    # Reorder columns to match the training data
    features = features[X_train_columns]

    # Predict the tier
    predicted_tier = model.predict(features)

    tier_map = {5: 'S', 4: 'A', 3: 'B', 2: 'C', 1: 'D', 0: 'E'}
    print(f"The predicted tier for {track_name} by {artist_name} is {tier_map[predicted_tier[0]]}")


# Example usage
if __name__ == "__main__":
    predict_tier_for_new_song("Survive", "Vanze/Neon Dreams")
