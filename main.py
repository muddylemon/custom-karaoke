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


if platform.system() == "Darwin":
    imagemagick_path = "/opt/homebrew/bin/magick"
elif platform.system() == "Windows":
    imagemagick_path = "C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/magick.exe"
else:
    raise NotImplementedError("Unsupported operating system")

print(f"Using ImageMagick path: {imagemagick_path}")

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


def separate_tracks(audio_file_path: str) -> tuple[str, str]:
    """Separates vocals and music from an audio file."""

    if not os.path.exists("./separated"):
        os.makedirs("./separated")

    audio_filename = audio_file_path.split("/")[-1]

    if os.path.exists(f"./separated/vocals_{audio_filename}"):
        return f"./separated/vocals_{audio_filename}", f"./separated/music_{audio_filename}"

    separator = demucs.api.Separator(progress=True, jobs=4)

    _, separated = separator.separate_audio_file(audio_file_path)

    for stem, source in separated.items():
        demucs.api.save_audio(
            source, f"./separated/{stem}_{audio_filename}", samplerate=separator.samplerate)

    demucs.api.save_audio(
        separated["other"] + separated["drums"] + separated["bass"], f"./separated/music_{audio_filename}", samplerate=separator.samplerate)

    return f"./separated/vocals_{audio_filename}", f"./separated/music_{audio_filename}"


def transcribe(audiofile_path: str, num_passes: int = 1, output_directory: str = "./subtitles") -> str:
    """
    Converts an MP3 file to a transcript using Whisper, assuming the last pass provides the best result.

    Args:
        audiofile_path (str): The file path of the MP3 file to be processed.
        num_passes (int): Number of transcription passes to perform.
        output_directory (str): Directory to save the output files.

    Returns:
        str: The path to the SRT file containing the transcript from the last pass.
    """
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if os.path.exists(f"{output_directory}/{os.path.splitext(os.path.basename(audiofile_path))[0]}.srt"):
            return f"{output_directory}/{os.path.splitext(os.path.basename(audiofile_path))[0]}.srt"

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = whisper.load_model("medium.en").to(device)

        last_result = None
        for i in range(num_passes):
            print(f"Transcription pass {i + 1} of {num_passes}...")
            current_result = model.transcribe(
                audiofile_path, verbose=True, language="en", word_timestamps=True)
            last_result = current_result

        if last_result is None:
            raise ValueError("No transcription results obtained.")

        srt_writer = get_writer("srt", output_directory)
        srt_writer(last_result, audiofile_path, highlight_words=True)

        txt_writer = get_writer("txt", output_directory)
        txt_writer(last_result, audiofile_path)

        subtitle_path = os.path.join(output_directory, os.path.splitext(
            os.path.basename(audiofile_path))[0] + '.srt')
        return subtitle_path

    except Exception as e:
        print(f"Error converting MP3 to transcript: {e}")
        return ""


def create(vocals_path: str, music_path: str, video_path: str):

    audio = AudioFileClip(music_path).set_fps(44100)
    vocals_audio = AudioFileClip(vocals_path).volumex(0.05).set_fps(44100)

    combined_audio = CompositeAudioClip([audio, vocals_audio])

    background_video = VideoFileClip(video_path, target_resolution=(720, 1280)).set_fps(
        30).set_duration(combined_audio.duration)

    def generator(txt):
        fontsize = 24 if txt == "instrumental" else 48
        fontcolor = "#aeedad" if txt == "instrumental" else "#FFEEFF"
        return TextClip(txt, font="./fonts/dv.ttf", fontsize=fontsize, color=fontcolor, stroke_color="#000000", stroke_width=0.5, size=(1240, None), method='pango')

    subtitles_path = transcribe(vocals_path)
    print(f"Subtitles path: {subtitles_path}")

    subtitles = SubtitlesClip(subtitles_path, generator)

    dimmed_background_video = background_video.fl_image(
        lambda image: (image * 0.3).astype("uint8"))

    result = CompositeVideoClip([
        dimmed_background_video,
        subtitles.set_position(('center', 'center'), relative=True)
    ]).set_audio(combined_audio)

    # Save the karaoke video
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
    # dewindowize
    video_path = args.video_path.replace("\\", "/")
    print(f"\nProcessing {video_path}.")

    audio_path = video_to_mp3(video_path)

    vocals_path, music_path = separate_tracks(audio_path)

    print(f"Creating karaoke video... {vocals_path} {music_path} {video_path}")

    create(vocals_path, music_path, video_path)


if __name__ == "__main__":
    main()
