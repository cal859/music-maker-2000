import os
import requests
import time

from bs4 import BeautifulSoup
from lxml import etree
import streamlit as st

FIRST_PAGE_URL = "https://www.metrolyrics.com/{artist}-alpage-1.html"
BASE_ARTIST_URL = "https://www.metrolyrics.com/{artist}-lyrics.html"
LYRICS_STORE = "./src/get_lyrics/lyrics_store"

if os.environ.get("IS_STREAMLIT"):
    PRINT_FN = st.write
else:
    PRINT_FN = print


def get_song_lyrics(url):
    lyrics = ""
    a = requests.get(url)
    a = BeautifulSoup(a.text, "html.parser")
    x = a.find_all("div", id="lyrics-body-text")[0]
    lyrics += " <SONGSTART> "
    for v in x.find_all("p"):
        lyrics += v.text
        lyrics += "\n"
    lyrics += " <SONGEND> "
    return lyrics


def get_songs(url):
    songs = []
    a = requests.get(url)
    a = BeautifulSoup(a.text, "html.parser")
    a = a.find_all("tbody")[0]
    for t in a.find_all("a"):
        songs.append(t["href"])
    return songs


def validate_artist_name(artist):
    return artist.lower().strip().replace(" ", "-")


def save_lyrics(file, artist):
    with open(f"{LYRICS_STORE}/{artist}.txt", "w") as out:
        out.write(file)


def check_cache(artist):
    if os.path.isfile(f"{LYRICS_STORE}/{artist}.txt"):
        PRINT_FN("file found in cache")
        out_f = ""
        with open(f"{LYRICS_STORE}/{artist}.txt", "r") as ini:
            out_f = ini.read()
        return out_f
    PRINT_FN("No file found in cache")
    return None


def get_lyrics_urls(artist: str):
    a = requests.get(BASE_ARTIST_URL.format(artist=artist))
    htmlparser = etree.HTMLParser()
    tree = etree.fromstring(a.text, htmlparser)
    song_pages = tree.xpath(
        "/html/body/div[2]/div[3]/div[2]/div[2]/div[1]/div[3]/div/div[1]/div[2]/p/span/a"
    )
    urls = [x.attrib["href"] for x in song_pages]
    if len(urls) == 0:
        # in the case where there is only one page of lyrics
        # then there are no additional pages to find
        # so no data is returned.
        # If this page doesn't exist, then we will handle
        # this error later
        # TODO: actualy handle the above case
        urls = [FIRST_PAGE_URL.format(artist=artist)]
    return urls


def get_lyrics(artist, use_cache=True):
    artist = validate_artist_name(artist)
    all_lyrics = ""
    i = 1
    total_songs = 0

    if use_cache:
        PRINT_FN("Checking cache...")
        cache_file = check_cache(artist)
        if cache_file:
            return cache_file
    PRINT_FN("Getting lyrics...")
    urls = get_lyrics_urls(artist)

    for url in urls:
        songs = get_songs(url)
        for s in songs:
            all_lyrics += get_song_lyrics(s)
            all_lyrics += "\n\n"
        PRINT_FN("processed {} page".format(i))
        i += 1
        total_songs += len(songs)

    save_lyrics(all_lyrics, artist)

    PRINT_FN("Lyrics found for {} songs".format(total_songs))

    return all_lyrics