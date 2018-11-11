# Referenced from https://sotiriskakanos.com/2017/08/05/using-spotifys-web-api-with-python/
#Import required libraries
import spotipy
import webbrowser
from spotipy.oauth2 import SpotifyClientCredentials

#Authorize API using credentials
client_credentials_manager = SpotifyClientCredentials(client_id='c05f99ee0a2d4c059c835f8f4cbe2c73', client_secret='596ed849c5a44891af5fdd1f74ba8365')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

playlists = sp.user_playlists('li006zvua0ttqvdzzns2gwtu5') #My User ID/Name

print(playlists)
# playlists = sp.user_playlists('dale_xi')
# playlists = sp.user_playlists('MartinStoichkov')

# while playlists:
#     for i, playlist in enumerate(playlists['items']):
#         print("%4d %s %s" % (i + 1 + playlists['offset'], playlist['uri'],  playlist['name']))
#     if playlists['next']:
#         playlists = sp.next(playlists)
#     else:
#         playlists = None

def show_tracks(tracks):
    for i, item in enumerate(tracks['items']):
        track = item['track']
        print("   %d %32.32s %s" % (i, track['artists'][0]['name'], track['name']))

for playlist in playlists['items']:
        if playlist['owner']['id'] == 'li006zvua0ttqvdzzns2gwtu5':
            print(playlist['name'])
            print(playlist['external_urls']['spotify'])
            webbrowser.open(playlist['external_urls']['spotify'])
            print('  total tracks', playlist['tracks']['total'])
            results = sp.user_playlist('li006zvua0ttqvdzzns2gwtu5', playlist['id'], fields="tracks,next")
            tracks = results['tracks']
            show_tracks(tracks)
            while tracks['next']:
                tracks = sp.next(tracks)
                show_tracks(tracks)