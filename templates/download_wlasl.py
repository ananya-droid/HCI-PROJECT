import json
import subprocess
from pathlib import Path
import urllib.request


# ==========================================
# SETTINGS
# ==========================================

METADATA_FILE = Path(
    "WLASL/selected_metadata.json"
)

OUTPUT_DIR = Path(
    "WLASL/videos"
)


# ==========================================
# CHECK METADATA
# ==========================================

if not METADATA_FILE.exists():

    print(
        "ERROR: selected_metadata.json not found."
    )

    print(
        "Run download_wlasl.py first."
    )

    raise SystemExit


with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


# ==========================================
# CREATE OUTPUT
# ==========================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print()
print("==============================")
print("WLASL VIDEO DOWNLOADER")
print("==============================")
print()

total = 0
downloaded = 0
failed = 0


# ==========================================
# DOWNLOAD
# ==========================================

for item in data:

    gloss = item["gloss"].lower()

    instances = item.get(
        "instances",
        []
    )

    sign_dir = (
        OUTPUT_DIR /
        gloss
    )

    sign_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print("------------------------------")
    print(
        f"SIGN: {gloss.upper()}"
    )
    print(
        f"Videos: {len(instances)}"
    )
    print("------------------------------")


    for index, instance in enumerate(
        instances
    ):

        total += 1


        video_id = str(
            instance.get(
                "video_id",
                index
            )
        )


        # ----------------------------------
        # Determine URL
        # ----------------------------------

        url = instance.get(
            "url"
        )


        if not url:

            print(
                f"[SKIP] {video_id} "
                f"no URL"
            )

            failed += 1

            continue


        output_file = (
            sign_dir /
            f"{video_id}.mp4"
        )


        # Already downloaded

        if output_file.exists():

            print(
                f"[EXISTS] {gloss}/{video_id}"
            )

            downloaded += 1

            continue


        print()
        print(
            f"[{index + 1}/{len(instances)}] "
            f"{gloss}/{video_id}"
        )


        # ----------------------------------
        # Try yt-dlp
        # ----------------------------------

        command = [
            "yt-dlp",
            "--no-warnings",
            "--quiet",
            "--no-progress",
            "-o",
            str(output_file),
            url
        ]


        try:

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )


            if (
                result.returncode == 0
                and output_file.exists()
            ):

                print(
                    "  SUCCESS"
                )

                downloaded += 1

            else:

                print(
                    "  FAILED"
                )

                if result.stderr:

                    print(
                        result.stderr[:300]
                    )

                failed += 1


        except FileNotFoundError:

            print()
            print(
                "ERROR: yt-dlp is not installed."
            )

            print()
            print(
                "Run:"
            )

            print(
                "pip install yt-dlp"
            )

            raise SystemExit


# ==========================================
# SUMMARY
# ==========================================

print()
print("==============================")
print("DOWNLOAD COMPLETE")
print("==============================")

print(
    f"Total references : {total}"
)

print(
    f"Downloaded       : {downloaded}"
)

print(
    f"Failed/skipped   : {failed}"
)

print()
print(
    f"Videos stored in:"
)

print(
    OUTPUT_DIR
)