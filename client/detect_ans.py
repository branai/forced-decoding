import pyaudio
import threading
import time
import requests
import sys

api_url = ''
if len(sys.argv) > 1:
    api_url = f"http://{sys.argv[1]}/api/getConfidence"
    print("API URL:", api_url)
else:
    print("Usage: python3 detect_ans.py <api-ip>")
    exit()


qa = {
    'q': "This painting depicts a scene from the Bible in which Jesus Christ shares a meal with his disciples.",
    'a': 'The Last Supper'
}
qa1 = {
    'q': "This painting by Ludolf Backhuysen from the 1600s shows three Dutch cargo ships in violent waters.",
    'a': "Ships in Distress off a Rocky Coast"
}

# parameters for audio recording
audio_format = pyaudio.paInt16 #16-bit PCM
num_channels = 1  # mono
sampling_rate = 44100
chunk_size = 1024


# stream for audio input
p = pyaudio.PyAudio()
stream = p.open(format=audio_format,
                channels=num_channels,
                rate=sampling_rate,
                input=True,
                frames_per_buffer=chunk_size)


# for recorded frames
frames = []

stop_recording = False

# keyboard input / api call thread
def input_thread():

    # prompt player
    print()
    print(qa['q'])
    print()
    print("======= RECORDING =======")
    input("Press Enter to submit answer...\n")
    print("Sending...")
    time.sleep(1)
    global stop_recording
    stop_recording = True

    payload = {
        "sound": list(b"".join(frames)), 
        "ground_truth": qa['a'].upper(), 
        "fr": sampling_rate
    }

    response = requests.post(api_url, json=payload)

    if response.status_code == 200:

        if response.text != "0\n":

            # probabilities given to us as seperate lines
            prob_arr = [float(p) for p in response.text.split("\n")[:-1]]
            print(prob_arr)

            # our measure of correctness (one bad word is enough to be incorrect)
            correct = True
            for curr in prob_arr:
                if curr < 0.1:
                    correct = False
                    break

            if correct:
                print("CORRECT")
            else:
                print("INCORRECT")


# recording audio thread
def record_thread():
    while not stop_recording:
        data = stream.read(chunk_size)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()


input_thread = threading.Thread(target=input_thread)
input_thread.start()

record_thread = threading.Thread(target=record_thread)
record_thread.start()


# join threads
record_thread.join()
input_thread.join()