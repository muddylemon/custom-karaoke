import argparse
import os

import demucs.api
import torch
import whisper
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip

from moviepy.config import change_settings

from whisper.utils import get_writer
import platform

TRANSCRIPTION_MODEL = "medium.en"
NUM_PASSES = 1
VOCAL_VOLUME = 0.05
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
TEXT_WIDTH = 1200
TEXT_COLOR = "#FFFFFF"
TEXT_STROKE_COLOR = "#000000"
TEXT_STROKE_WIDTH = 0.5
FONT_SIZE = 40
FONT = "./fonts/kg.ttf"


if platform.system() == "Darwin":
    imagemagick_path = "/opt/homebrew/bin/magick"
elif platform.system() == "Windows":
    imagemagick_path = "C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/magick.exe"
else:
    raise NotImplementedError("Unsupported operating system")

change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})


def video_to_mp3(video_path: str):
    """Converts a video file to an mp3 file."""
    print(f"Converting video to mp3 -> {video_path}")
    audio_path = video_path.replace(".mp4", ".mp3")
    if os.path.exists(audio_path):
        return audio_path

    audio = AudioFileClip(video_path)
    audio.write_audiofile(audio_path, logger="bar")
    print(f"Audio saved to: {audio_path}")
    return audio_path


def separate_stems(audio_file_path: str) -> tuple[str, str]:
    """Separates vocals and music from an audio file."""

    if not os.path.exists("./stems"):
        os.makedirs("./stems")

    audio_filename = audio_file_path.split("/")[-1]

    if os.path.exists(f"./stems/vocals_{audio_filename}"):
        return f"./stems/vocals_{audio_filename}", f"./stems/music_{audio_filename}"

    print(f"Separating vocals from {audio_file_path}")

    separator = demucs.api.Separator(progress=True, jobs=4)

    _, separated = separator.separate_audio_file(audio_file_path)

    for stem, source in separated.items():
        demucs.api.save_audio(
            source, f"./stems/{stem}_{audio_filename}", samplerate=separator.samplerate)

    demucs.api.save_audio(
        separated["other"] + separated["drums"] + separated["bass"], f"./stems/music_{audio_filename}", samplerate=separator.samplerate)

    return f"./stems/vocals_{audio_filename}", f"./stems/music_{audio_filename}"


def transcribe(audiofile_path: str, num_passes: int = 1) -> str:
    """
    Converts an MP3 file to a transcript using Whisper

    Args:
        audiofile_path (str): The file path of the MP3 file to be processed.
        num_passes (int): Number of transcription passes to perform.
    Returns:
        str: The path to the SRT file containing the transcript from the last pass.
    """
    try:

        subtitle_path = os.path.join("./subtitles", os.path.splitext(
            os.path.basename(audiofile_path))[0] + '.srt')

        if os.path.exists(subtitle_path):
            return subtitle_path

        if not os.path.exists("./subtitles"):
            os.makedirs("./subtitles")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = whisper.load_model(TRANSCRIPTION_MODEL).to(device)

        last_result = None
        for i in range(num_passes):
            print(f"Transcription pass {i + 1} of {num_passes}...")
            current_result = model.transcribe(
                audiofile_path, verbose=True, language="en", word_timestamps=True)
            last_result = current_result

        if last_result is None:
            raise ValueError("No transcription results obtained.")

        srt_writer = get_writer("srt", "./subtitles")
        srt_writer(last_result, audiofile_path, highlight_words=True)

        return subtitle_path

    except Exception as e:
        print(f"Error converting MP3 to transcript: {e}")
        return ""


def create(video_path: str):
    """Creates a karaoke video from the separated audio files and the original video file.

    Args:
        video_path (str): The path to the original video file.

    Returns:
        str: The filename of the created karaoke video.
    """

    audio_path = video_to_mp3(video_path)

    vocals_path, music_path = separate_stems(audio_path)

    music = AudioFileClip(music_path).set_fps(44100)
    vocals_audio = AudioFileClip(vocals_path).volumex(
        VOCAL_VOLUME).set_fps(44100)

    combined_audio = CompositeAudioClip([music, vocals_audio])

    background_video = VideoFileClip(video_path, target_resolution=(VIDEO_HEIGHT, VIDEO_WIDTH)).set_fps(
        30).set_duration(combined_audio.duration)

    dimmed_background_video = background_video.fl_image(
        lambda image: (image * 0.3).astype("uint8"))

    subtitles_file = transcribe(vocals_path, NUM_PASSES)

    def generator(txt):
        """Generates the subtitles for the karaoke video.

        Args:
            txt (str): The text to be added to the subtitles.

        Returns:
            TextClip: The subtitle text clip.
        """

        return TextClip(
            txt,
            font=FONT,
            fontsize=FONT_SIZE,
            color=TEXT_COLOR,
            stroke_color=TEXT_STROKE_COLOR,
            stroke_width=TEXT_STROKE_WIDTH,
            size=(TEXT_WIDTH, None),
            method='pango'
        )

    subtitles = SubtitlesClip(subtitles_file, generator)

    result = CompositeVideoClip([
        dimmed_background_video,
        subtitles.set_position(('center', 'center'), relative=True)
    ]).set_audio(combined_audio)

    filename = f"karaoke_{os.path.basename(video_path)}"
    if not os.path.exists("./output"):
        os.makedirs("./output")
    result.write_videofile(f"./output/{filename}", fps=30, threads=4)

    return filename


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create a karaoke video from a video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "video_path", help="Path to the video file.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    video_path = args.video_path.replace("\\", "/")

    print(f"Creating karaoke video..")

    create(video_path)


if __name__ == "__main__":
    main()
