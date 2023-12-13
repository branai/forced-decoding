import wave
import string

#
# split_test_data.py: This script was used for testing (split audio given the youtube transcript)
#


wav_file_path = "HS_QUIZ_TRUE.wav"
file_path = "HS_QUIZ_TRANS.txt"

# Clean section of transcript
def clean_text(input_string):
    # Host name appears in transcript
    if ">> COSTA: " in input_string:
        input_string = input_string.replace('>> ', '')

    # Remove arrows for changing speakers
    if ">>" in input_string:
        input_string = input_string.replace('>> ', '')

    # Remove punctuation
    translation_table = str.maketrans("", "", string.punctuation.replace("'", ""))
    result_string = input_string.translate(translation_table)
    return result_string

# Get section of audio
def get_sound_data(start_seconds, end_seconds):
    with wave.open(wav_file_path, 'rb') as wave_file:
        frame_rate = wave_file.getframerate()
        start_frame = int(start_seconds * frame_rate)
        end_frame = int(end_seconds * frame_rate)
        wave_file.setpos(start_frame)
        sound_data = wave_file.readframes(end_frame - start_frame)
        return frame_rate, list(sound_data)


# Get audio corresponding to each piece of transcript
# Returns [ [transcript, sound_data] ...]
def split_trans_audio():

    # Get lines of transcript
    with open(file_path, 'r') as file:
        lines = file.readlines()

    trans_frame = []

    # Lines are timestamp followed by transcript
    for i in range(0, len(lines)-2, 2):

        # Convert MM:SS to seconds (end time is after actual transcript)
        start_min, start_sec = lines[i].strip().split(":") 
        end_min, end_sec = lines[i+2].strip().split(":")
        time_frame = [int(start_min) * 60 + int(start_sec), int(end_min) * 60 + int(end_sec)+2]
        
        partial_trans = lines[i+1].strip()

        # Just skip over these cases
        if "(" in partial_trans:
            continue

        if any(char.isdigit() for char in partial_trans):
            continue

        partial_trans = clean_text(partial_trans)
        # Transcript followed by timeframe
        trans_frame.append([partial_trans.upper(), time_frame])

    trans_sounds = []

    for i in range(len(trans_frame)):

        # Get sound data of time frame
        fr, sound_data = get_sound_data(trans_frame[i][1][0], trans_frame[i][1][1])

        # Transcript followed by sound data
        words_sound = [trans_frame[i][0], sound_data]
        trans_sounds.append(words_sound)

    print(len(trans_sounds), len(trans_frame))
    return trans_sounds, fr

trans_audio, fr = split_trans_audio()

for curr in trans_audio:
    print("Transcript:", curr[0])
    print("# Frames in Audio", len(curr[1]))
    print()
    pass
