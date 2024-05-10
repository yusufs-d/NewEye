from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.effects import speedup
from pydub.playback import play
from pygame import mixer, time as tm
import os


def text_to_speech_and_save(text, filename, lang='en',  path="sounds",speed=1.0):
    # Full path to save the file
    full_path = os.path.join(path, filename)

    # Convert text to speech
    tts = gTTS(text=text, lang=lang)
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    
    # Load audio into pydub
    audio = AudioSegment.from_file(buffer, format="mp3")
    
    # Adjust the speed of the audio if not 1.0
    if speed != 1.0:
        audio = speedup(audio, playback_speed=speed)
    
    # Save the modified audio to an MP3 file at the specified location
    audio.export(full_path, format="mp3")


def play_audio(filename, path="sounds"):
    full_path = os.path.join(path, filename)
    mixer.init()
    if not mixer.music.get_busy():  # Eğer zaten bir ses oynatılmıyorsa
        mixer.music.load(full_path)
        mixer.music.play()
        while mixer.music.get_busy():  # Oynatma bitene kadar döngüde kal
            tm.Clock().tick(10)

def play_multiple_audio(filename1, filename2, path = "sounds"):
    play_audio(filename1,path)
    play_audio(filename2,path)

def buildSounds(sounds : list,speed : float, path = "sounds"):
    for sound in sounds:
        text_to_speech_and_save(
            sound[0],
            sound[1]+".mp3",
            speed=speed, 
            path=path)




