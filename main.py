import argparse
import gc
import os

import demucs.api
import torch
import whisperx
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip

from moviepy.config import change_settings


from whisper.utils import get_writer

change_settings(
    {"IMAGEMAGICK_BINARY": "C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/magick.exe"})


def video_to_mp3(video_path: str):
    """Converts a video file to an mp3 file."""
    print(f"Converting video to mp3 -> {video_path}")
    audio = AudioFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".mp3")
    audio.write_audiofile(audio_path, logger="bar")
    print(f"Audio saved to: {audio_path}")
    return audio_path


def separate_tracks(audio_file_path: str) -> tuple[str, str]:
    """Separates vocals and music from an audio file."""

    if not os.path.exists("./separated"):
        os.makedirs("./separated")

    audio_filename = audio_file_path.split("/")[-1]

    separator = demucs.api.Separator(progress=True, jobs=4)

    _, separated = separator.separate_audio_file(audio_file_path)

    for stem, source in separated.items():
        demucs.api.save_audio(
            source, f"./separated/{stem}_{audio_filename}", samplerate=separator.samplerate)

    demucs.api.save_audio(
        separated["other"] + separated["drums"] + separated["bass"], f"./separated/music_{audio_filename}", samplerate=separator.samplerate)

    return f"./separated/vocals_{audio_filename}", f"./separated/music_{audio_filename}"


def transcribe(audio_file: str) -> dict:
    """Transcribe an audio file using Whisper."""
    device = "cuda"
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float16"

    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    model_dir = "./models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(
        audio, batch_size=batch_size)

    gc.collect()
    torch.cuda.empty_cache()

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device)

    result = whisperx.align(result["segments"], model_a,
                            metadata, audio, device, print_progress=True)

    return result


def write_subtitles(subtitles: dict, vocals_path: str):
    """Write subtitles to a video file."""

    output_directory = "./subtitles"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    srt_writer = get_writer("srt", output_directory)
    srt_writer(subtitles, vocals_path)

    subtitle_path = os.path.join(output_directory, os.path.splitext(
        os.path.basename(vocals_path))[0] + '.srt')

    return subtitle_path


def create(vocals_path: str, music_path: str, video_path: str):

    audio = AudioFileClip(music_path).set_fps(44100)
    vocals_audio = AudioFileClip(vocals_path).volumex(0.075).set_fps(44100)

    combined_audio = CompositeAudioClip([audio, vocals_audio])

    background_video = VideoFileClip(video_path, target_resolution=(720, 1280)).set_fps(
        30).set_duration(combined_audio.duration)

    text_container_size = (background_video.w - 2 *
                           (background_video.w // 20), None)

    font = "fonts/dv.tff"

    default_font_size = min(max(int(background_video.w / 18), 15), 45)

    def generator(txt):
        if txt == "instrumental":
            fontsize = default_font_size // 2
            fontcolor = "#aeedad"
        else:
            fontsize = default_font_size
            fontcolor = "#FFEEFF"

        return TextClip(
            txt,
            font=font,
            fontsize=fontsize,
            color=fontcolor,
            stroke_color="#000000",
            stroke_width=0.5,
            size=text_container_size,
            method="caption",
            align='center'
        )

    segments = transcribe(vocals_path)
    subtitles_path = write_subtitles(segments, vocals_path)

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
