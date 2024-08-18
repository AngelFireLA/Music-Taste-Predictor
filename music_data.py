import os

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import sqlite3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
LASTFM_API_KEY = os.getenv('LASTFM_API_KEY')
LASTFM_USERNAME = os.getenv('LASTFM_USERNAME')

# Authenticate with Spotify
auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)
sp.recommendations()
# Initialize SQLite database
conn = sqlite3.connect('music_data.db')
cursor = conn.cursor()

# Create tables if they do not exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS artists (
        artist_id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        spotify_url TEXT,
        lastfm_url TEXT,
        genres_spotify TEXT,
        genres_lastfm TEXT,
        playcount INTEGER,
        listeners INTEGER,
        user_artist_playcount INTEGER
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS tracks (
        track_id INTEGER PRIMARY KEY,
        title TEXT,
        album_name TEXT,
        artist_id INTEGER,
        release_date TEXT,
        duration_ms INTEGER,
        tier TEXT,
        popularity INTEGER,
        isrc TEXT,
        spotify_url TEXT,
        lastfm_url TEXT,
        danceability REAL,
        energy REAL,
        key INTEGER,
        loudness REAL,
        mode INTEGER,
        speechiness REAL,
        acousticness REAL,
        instrumentalness REAL,
        liveness REAL,
        valence REAL,
        tempo REAL,
        time_signature INTEGER,
        playcount_user INTEGER,
        track_tags TEXT,
        FOREIGN KEY(artist_id) REFERENCES artists(artist_id)
    )
''')


def get_spotify_data(track_name, artist_name):
    results = sp.search(q=f'track:{track_name} artist:{artist_name}', type='track', limit=1)
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        track_id = track['id']

        # Track Info
        track_info = {
            'track_name': track['name'],
            'album_name': track['album']['name'],
            'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
            'release_date': track['album']['release_date'],
            'track_duration_ms': track['duration_ms'],
            'popularity': track['popularity'],
            'isrc': track['external_ids'].get('isrc'),
            'spotify_url': track['external_urls']['spotify'],
            'album_cover_art': track['album']['images'],
            'track_uri': track['uri']
        }

        # Audio Features
        audio_features = sp.audio_features(track_id)[0]
        if audio_features is not None:
            track_info.update({
                'danceability': audio_features['danceability'],
                'energy': audio_features['energy'],
                'key': audio_features['key'],
                'loudness': audio_features['loudness'],
                'mode': audio_features['mode'],
                'speechiness': audio_features['speechiness'],
                'acousticness': audio_features['acousticness'],
                'instrumentalness': audio_features['instrumentalness'],
                'liveness': audio_features['liveness'],
                'valence': audio_features['valence'],
                'tempo': audio_features['tempo'],
                'time_signature': audio_features['time_signature']
            })

        # Artist Info (Spotify)
        artist_id = track['artists'][0]['id']
        artist_info = sp.artist(artist_id)
        track_info.update({
            'artist_genres_spotify': ', '.join(artist_info['genres']),
            'artist_spotify_url': artist_info['external_urls']['spotify']
        })

        return track_info
    else:
        return None


def get_user_artist_playcount(artist_name):
    total_playcount = 0
    page = 1

    while True:
        top_artists_url = f"http://ws.audioscrobbler.com/2.0/?method=user.getTopArtists&user={LASTFM_USERNAME}&api_key={LASTFM_API_KEY}&format=json&page={page}"
        response = requests.get(top_artists_url)
        data = response.json()
        if 'topartists' in data and 'artist' in data['topartists']:
            artists = data['topartists']['artist']
            for artist in artists:
                if artist['name'].lower() == artist_name.lower():
                    total_playcount = int(artist['playcount'])
                    return total_playcount

            # Check if there are more pages to fetch
            if int(data['topartists']['@attr']['totalPages']) > page:
                page += 1
            else:
                break
        else:
            break

    return total_playcount


def get_lastfm_data(track_name, artist_name):
    track_info = {}

    # Track Info (Last.fm)
    track_url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={LASTFM_API_KEY}&artist={artist_name}&track={track_name}&username={LASTFM_USERNAME}&format=json"
    track_response = requests.get(track_url)
    track_data = track_response.json().get('track')
    print(track_data)
    if track_data:

        track_info.update({
            'playcount_user': int(track_data['userplaycount']) if 'userplaycount' in track_data else None,
            'loved': int(track_data['userloved']) if 'userloved' in track_data else None,
            'track_tags': ', '.join(tag['name'] for tag in track_data['toptags']['tag']),
            'track_listeners': int(track_data['listeners']),
            'track_playcount': int(track_data['playcount']),
            'track_lastfm_url': track_data['url']
        })

    # Artist Info (Last.fm)
    artist_url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getInfo&api_key={LASTFM_API_KEY}&artist={artist_name}&format=json"
    artist_response = requests.get(artist_url)
    artist_data = artist_response.json().get('artist')

    if artist_data:
        track_info.update({
            'artist_tags': ', '.join(tag['name'] for tag in artist_data['tags']['tag']),
            'artist_playcount': int(artist_data['stats']['playcount']),
            'artist_listeners': int(artist_data['stats']['listeners']),
            'artist_lastfm_url': artist_data['url'],
            'user_artist_playcount': get_user_artist_playcount(artist_name)
        })

    return track_info


def save_data_to_db(track_name, artist_name, combined_data):
    # Check if the artist exists
    cursor.execute("SELECT artist_id FROM artists WHERE name = ?", (artist_name,))
    artist_row = cursor.fetchone()

    if artist_row:
        artist_id = artist_row[0]
    else:
        # Insert new artist
        cursor.execute('''
            INSERT INTO artists (name, spotify_url, lastfm_url, genres_spotify, genres_lastfm, playcount, listeners, user_artist_playcount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            artist_name,
            combined_data.get('artist_spotify_url'),
            combined_data.get('artist_lastfm_url'),
            combined_data.get('artist_genres_spotify'),
            combined_data.get('artist_tags'),
            combined_data.get('artist_playcount'),
            combined_data.get('artist_listeners'),
            combined_data.get('user_artist_playcount')
        ))
        artist_id = cursor.lastrowid

    # Check if the track exists
    cursor.execute("SELECT * FROM tracks WHERE title = ? AND artist_id = ?", (track_name, artist_id))
    track_row = cursor.fetchone()

    if track_row:
        print(f"The track '{track_name}' by '{artist_name}' already exists in the database.")
        confirm = input("Do you want to overwrite it? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return

    # Insert or update track
    cursor.execute('''
        INSERT OR REPLACE INTO tracks (
            title, tier, album_name, artist_id, release_date, duration_ms, popularity, isrc, spotify_url, lastfm_url, 
            danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, 
            tempo, time_signature, playcount_user, track_tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        track_name,
        combined_data.get('tier'),
        combined_data.get('album_name'),
        artist_id,
        combined_data.get('release_date'),
        combined_data.get('track_duration_ms'),
        combined_data.get('popularity'),
        combined_data.get('isrc'),
        combined_data.get('spotify_url'),
        combined_data.get('track_lastfm_url'),
        combined_data.get('danceability'),
        combined_data.get('energy'),
        combined_data.get('key'),
        combined_data.get('loudness'),
        combined_data.get('mode'),
        combined_data.get('speechiness'),
        combined_data.get('acousticness'),
        combined_data.get('instrumentalness'),
        combined_data.get('liveness'),
        combined_data.get('valence'),
        combined_data.get('tempo'),
        combined_data.get('time_signature'),
        combined_data.get('playcount_user'),
        combined_data.get('track_tags'),
    ))

    conn.commit()
    print(f"The track '{track_name}' by '{artist_name}' has been saved to the database.")


def get_combined_data(track_name, artist_name):
    spotify_data = get_spotify_data(track_name, artist_name)
    lastfm_data = get_lastfm_data(track_name, artist_name)

    if spotify_data and lastfm_data:
        combined_data = {**spotify_data, **lastfm_data}
        return combined_data
    else:
        return {**lastfm_data}


# # Example usage
track_name = "Believer"
artist_name = "Imagine Dragons"
# combined_data = get_combined_data(track_name, artist_name)
#
# if combined_data:
#     save_data_to_db(track_name, artist_name, combined_data)
# else:
#     print("Failed to retrieve data for the given track and artist.")
#
# # Close the database connection
# conn.close()