import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create a karaoke video from a video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "video_path", help="Path to the video file.")
    parser.add_argument(
        "--skip-subtitles",
        action="store_true",
        help="Skip generating subtitles.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    # dewindowize
    video_path = args.video_path.replace("\\", "/")
    print(video_path)

    # audio_path = video_to_mp3(video_path)
    # vocals_path, music_path = separate_tracks(audio_path)

    # create_karaoke_video(vocals_path, music_path, video_path)


if __name__ == "__main__":
    main()
