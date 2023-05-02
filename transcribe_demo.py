#! python3.7

import argparse
import io
import json
import os
import speech_recognition as sr
import openai

from firebase import firebase
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

import config

app_url = 'https://conversationsummarizer-11571-default-rtdb.firebaseio.com/'
fb = firebase.FirebaseApplication(app_url, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users. 
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=24000)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    #print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                #Write wav_data to a wav file
                with open("audio.wav", "wb") as f:
                    f.write(wav_data.getbuffer())
                #No need to convert to mp3, just use the wav file
                #Convert wav to mp3
                #stream = ffmpeg.input("audio.wav")
                #stream = ffmpeg.output(stream, "audio.mp3")
                #ffmpeg.run(stream)
                #Transcribe the audio
                text = transcribe("audio.wav")


                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                # Flush stdout.
                print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    full_transcription = ""
    print("\n\nTranscription:")
    for line in transcription:
        print(line)
        full_transcription = full_transcription + line

    summary = summarize(full_transcription)

    print(summary)

    # create a reference to the conversations collection
    conversations_ref = firebase.FirebaseApplication(app_url + '/conversations', None)


    # store the full transcription and the summary as a document
    result = conversations_ref.post("/", json.loads(summary))

    # print the result
    print(result)



# A function which uses the openai bindings to transcribe audio using the whisper api
def transcribe(audio_file):
    # Load your OpenAI API key and set the environment variable
    openai.api_key = config.api_key  # Replace with your actual API key


    # Open the converted mp3 file and pass it to the OpenAI Whisper API
    with open("audio.wav", "rb") as f:
        model_engine = "whisper-1"
        response = openai.Audio.transcribe(model=model_engine, file=f)

    # Extract the transcribed text from the response
    transcribed_text = response["text"]

    return transcribed_text

# A function which uses the openai bindings to summarize the given text
def summarize(text):
    # Load your OpenAI API key and set the environment variable
    openai.api_key = config.api_key  # Replace with your actual API key

    # Create a list of messages
    example1 = '{"confidence": "0.75", "participants": "3", "summary": "Colleagues discussed concerns about the new product\'s marketing and development. The team leader reassured them that everything was on track for a successful launch. However, some parts of the conversation were unclear due to transcription errors, including the timeline for the product and details on budget allocation.", "title": "Discussion on New Product Launch"}'
    example2 = 'example2 = \'{"confidence": "0.9", "participants": "2", "Two participants discuss using a Raspberry Pi as a main computer. One expresses their dislike for it, while the other suggests using SSH to control it. The conversation includes some confusion over the Raspberry Pi\'s use, but there were no obvious transcription errors.", "title": "Using Raspberry Pi as Main Computer"}'

    messages = [
        {
            "role": "system",
            "content": "Objective: You will receive a message representing a conversation between one or more individuals." +
                        "First, produce a score which indicates your confidence in the summary. This should essentially be a measure of how correct you think the transcription was, and how much context you think you have for the summary. If the conversation seems to not make sense in places or has one or more seemingly random phrasings, it's probably a transcription error and confidence should reflect this as well."
                        "Next, produce an integer for the estimated number of participants in the converstaion. Label this 'participants'" +
                        "Next, produce an internal monologue which reinforces important points in the conversation and identifies focal points. Also, note points which are likely to have been transcription problems." +
                        "Next, produce a summary of the conversation which highlights each important point. Do not make leaps in logic beyond your context, and instead note when you do not have context or did not understand what was going on."
                        "Finally, come up with a title for the conversation."
        },
        {
            "role": "system",
            "content": "Format (*EXTREMELY IMPORTANT*): JSON"
        },
        {
            "role": "system",
            "content": "Extra-Details: Ignore any non-english words or characters as if they did not exist." +
            "Also, ensure that each field contains a non-empty string." +
            "Additionally, if you find probably transcription errors, you should note them." +
            "Finally, ALWAYS format your response in JSON. Otherwise, the program will CRASH"
        },
        {
            "role": "system",
            "content": "Example One (NOTICE THE JSON FORMAT): " + example1
        },
        {
            "role": "system",
            "content": "Example Two: " + example2
        },
        {
            "role": "user",
            "content": "Here is your conversation: " + text
        }
    ]

    # Call the ChatCompletion endpoint with the gpt-3.5-turbo model and the messages list
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.73
    )

    output = completion.choices[0].message.content;

    return output


if __name__ == "__main__":
    main()
