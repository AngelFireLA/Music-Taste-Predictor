import os
import re
import sqlite3
import time

import mutagen
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.id3 import ID3, TIT2, TPE1
from music_data import get_combined_data, save_data_to_db

# Initialize SQLite database
conn = sqlite3.connect('music_data.db')
cursor = conn.cursor()

# Path to the root folder containing rank folders
ROOT_FOLDER = 'music'

# Mapping folder names to tier ranks
RANK_MAPPING = {
    'Rank S': 'S',
    'Rank A': 'A',
    'Rank B': 'B',
    'Rank C': 'C',
    'Rank D': 'D',
    'Rank E': 'E'
}


def classify_and_populate_tracks(root_folder):
    for rank_folder, tier in RANK_MAPPING.items():
        folder_path = os.path.join(root_folder, rank_folder)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist, skipping.")
            continue
        i = 0
        for filename in os.listdir(folder_path):
            i+=1
            print(i)
            if filename.endswith(('.mp3', '.flac', '.m4a', '.wav')):  # Add other extensions if needed
                track_name = clean_filename(filename)
                artist_name = get_artist_from_metadata(os.path.join(folder_path, filename))

                if track_name and artist_name:
                    print(f"Processing: {track_name} by {artist_name}, Tier: {tier}")
                    combined_data = get_combined_data(track_name, artist_name)
                    if combined_data:
                        combined_data['tier'] = tier
                        save_data_to_db(track_name, artist_name, combined_data)
                    else:
                        print(f"Failed to retrieve data for {track_name} by {artist_name}")
                else:
                    print(f"Could not process {filename}, skipping.")


def clean_filename(filename):
    """
    Clean the filename by removing trailing (1), (2), etc., but keeping other parts.
    Returns the cleaned track name without the file extension.
    """
    track_name = os.path.splitext(filename)[0]
    track_name = re.sub(r'\(\d\)$', '', track_name).strip()  # Remove trailing (1), (2), (3), etc.
    return track_name


def get_artist_from_metadata(file_path):
    """
    Extract the artist(s) from the file's metadata. Handles different file formats.
    Returns the artist name(s) as a string.
    """
    try:
        if file_path.endswith('.mp3'):
            audio = MP3(file_path, ID3=ID3)
            artist = audio.get('TPE1')  # ID3 tag for artist
            if artist:
                return artist.text[0]
        elif file_path.endswith('.flac'):
            audio = FLAC(file_path)
            artist = audio.get('artist')
            if artist:
                return artist[0]
        elif file_path.endswith('.m4a'):
            audio = MP4(file_path)
            artist = audio.get('\xa9ART')  # MP4 tag for artist
            if artist:
                return artist[0]
        elif file_path.endswith('.wav'):
            # WAV files don't typically contain metadata, so return None
            return None

        if artist:
            # Handle multiple artists separated by common delimiters
            artist = re.split(r'[\\/]', artist)
            return ', '.join([a.strip() for a in artist])
    except Exception as e:
        print(f"Error reading metadata from {file_path}: {e}")
        return None


# Example usage
if __name__ == "__main__":
    classify_and_populate_tracks(ROOT_FOLDER)

# Close the database connection
conn.close()
