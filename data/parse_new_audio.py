import numpy as np 
import librosa as ls 
import random
import math
import os
import warnings
warnings.filterwarnings("ignore")


def main():
    # get 30 second blocks of audio using ls.load
    # get the path from the user
    path = os.getcwd()

    # create new CSV file
    new_file_name = input("Enter the name of the new CSV file: ")
    write_file = open(str(new_file_name) + '.csv', 'w')

    print("Finding files in cur directory")
    files = ls.util.find_files(path)
    print("Here are the files found in your current directory:")
    [print("\033[96m", file.split('/')[-1], '\033[0m', '\n') for file in files]

    path = input("Enter the name of the audio file: ")

    # load the audio file
    print("Loading audio file", path)
    audio, sr = ls.load(path)
    print("Audio loaded")

    # get the duration of the audio file
    duration = ls.get_duration(y=audio, sr=sr)
    print("Duration: ", duration)

    # get the start and end times for each block 
    start_times = np.arange(0, duration, 30)

    # convert the start times to frames
    frames_array = start_frames = ls.time_to_frames(start_times, sr=sr)

    # pick a random block to extract (except the last one)
    start_frame_index = random.randint(0, len(frames_array) - 2)
    start_frame = frames_array[start_frame_index]

    # analyze until next 30 second block and extract features
    end_frame = frames_array[start_frame_index + 1]

    # within 30 seconds, divide into 3 second blocks
    three_sec = math.floor((end_frame - start_frame) / 10)
    print(three_sec)

    for i in range(10):
        # extract filename
        filename = path.split('/')[-1]

        this_start_frame = start_frame + (three_sec * i)
        this_end_frame = this_start_frame + three_sec

        # extract chroma_stft
        chroma_stft = ls.feature.chroma_stft(y=audio[this_start_frame:this_end_frame], sr=sr, n_fft=1024, hop_length=2)
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_var = np.var(chroma_stft)

        # extract rms
        rms = ls.feature.rms(y=audio[this_start_frame:this_end_frame])
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)

        # extract spectral centroid
        spectral_centroid = ls.feature.spectral_centroid(y=audio[this_start_frame:this_end_frame], sr=sr, n_fft=1024, hop_length=8)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_var = np.var(spectral_centroid)

        # extract spectral bandwidth
        spectral_bandwidth = ls.feature.spectral_bandwidth(y=audio[this_start_frame:this_end_frame], sr=sr, n_fft=1024, hop_length=8)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_var = np.var(spectral_bandwidth)

        # print out all data
        print(f"Filename: {filename}")
        print(f"Chroma_stft_mean: {chroma_stft_mean}")
        print(f"Chroma_stft_var: {chroma_stft_var}")
        print(f"RMS_mean: {rms_mean}")
        print(f"RMS_var: {rms_var}")
        print(f"Spectral_centroid_mean: {spectral_centroid_mean}")
        print(f"Spectral_centroid_var: {spectral_centroid_var}")
        print(f"Spectral_bandwidth_mean: {spectral_bandwidth_mean}")
        print(f"Spectral_bandwidth_var: {spectral_bandwidth_var}")

        # store as row in CSV
        write_file.write(f"{filename},{chroma_stft_mean},{chroma_stft_var},{rms_mean},{rms_var},{spectral_centroid_mean},{spectral_centroid_var},{spectral_bandwidth_mean},{spectral_bandwidth_var}\n")

    write_file.close()
    print("data written to", new_file_name + '.csv in the current directory')
main()