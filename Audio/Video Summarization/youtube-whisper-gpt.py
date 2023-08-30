import os
import shutil
import librosa
import openai
import soundfile as sf
import youtube_dl
from youtube_dl import DownloadError

openai.api_key = "Enter your api key"


def find_audio_files(path, extension=".mp3"):

    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                audio_files.append(os.path.join(root, f))
    return audio_files


def youtube_to_mp3(youtube_url, output_dir):

    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        # "verbose": True
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Downloading video from {youtube_url}")

    try:
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])
    except DownloadError:
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])

    audio_filename = find_audio_files(output_dir)[0]
    return audio_filename


def chunk_audio(filename, segment_length, output_dir):

    print(f"Chunking audio to {segment_length} second segments")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    audio, sr = librosa.load(filename, sr=44100)
    duration = librosa.get_duration(y=audio, sr=sr)
    num_segments = int(duration / segment_length) + 1
    print(f"Chunking {num_segments} chunks ...")

    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        segment = audio[start:end]
        sf.write(os.path.join(output_dir, f"segment_{i}.wav"), segment, sr)

    chunked_audio_files = find_audio_files(output_dir)
    return sorted(chunked_audio_files)


def transcribe_audio(audio_files, output_file, model="whisper-1"):

    print("Converting audio to text ...")

    transcripts = []
    for audio_file in audio_files:
        audio = open(audio_file, "rb")
        response = openai.Audio.transcribe(model, audio)
        transcripts.append(response["text"])

    if output_file is not None:
        with open(output_file, "w") as file:
            for transcript in transcripts:
                file.write(transcript + "\n")

    return transcripts


def summarize(chunks, system_prompt, model="gpt-3.5-turbo", output_file=None):

    print(f"Summarizing with {model} ...")

    summaries = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model,
            message=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ]
        )
        summary = response["choices"][0]["message"]["content"]
        summaries.append(summary)

    if output_file is not None:
        with open(output_file, "w") as file:
            for summary in summaries:
                file.write(summary + "\n")

    return summaries


def summarize_youtube_video(youtube_url, outputs_dir):

    raw_audio_dir = f"{outputs_dir}/raw_audio/"
    chunks_dir = f"{outputs_dir}/audio_chunks/"
    transcripts_file = f"{outputs_dir}/transcripts.txt"
    summary_file = f"{outputs_dir}/summary.txt"
    segment_length = 10 * 60

    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir)
        os.mkdir(outputs_dir)

    audio_filename = youtube_to_mp3(youtube_url, output_dir=raw_audio_dir)

    chunked_audio_files = chunk_audio(
        audio_filename,
        segment_length=segment_length,
        output_dir=chunks_dir
    )

    transcriptions = transcribe_audio(chunked_audio_files, transcripts_file)

    system_prompt = """
        You are a helpful assistant that summarizes youtube videos.
        You are provided chunks of raw audio that were transcribed from the video's audio.
        Summarize the current chunk to succint and clear bullet points of its contents.
    """

    summaries = summarize(
        transcriptions,
        system_prompt=system_prompt,
        output_file=summary_file
    )

    system_prompt_tldr = """
        You are a helpful assistant that summarizes youtube videos.
        Someone has already summarized the video to key points.
        Summarize the key points to one or two sentences that capture the essence of the video.
    """

    long_summary = "\n".join(summaries)
    short_summary = summarize(
        [long_summary],
        system_prompt=system_prompt_tldr,
        output_file=summary_file
    )[0]

    return long_summary, short_summary


youtube_url = "Enter the youtube video link"
outputs_dir = "./youtube-whisper-gpt-outputs/"

long_summary, short_summary = summarize_youtube_video(youtube_dl, outputs_dir)

print("Summaries:")
print("=" * 80)
print("Long summary:")
print("=" * 80)
print(long_summary)
print()

print("=" * 80)
print("Video - TLDR")
print("=" * 80)
print(short_summary)
