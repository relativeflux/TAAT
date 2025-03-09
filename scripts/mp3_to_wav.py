import os
import argparse
from pydub import AudioSegment

parser = argparse.ArgumentParser(description="Tape Archive Analysis Toolkit (TAAT)")
parser.add_argument("--src", type=str, required=True, help="Source mp3 file.")
parser.add_argument("--dest", type=str, help="Destination wav file.")
args = parser.parse_args()

# files                                                                     
src = args.src

(name, _) = os.path.splitext(src)
dst = args.dest or f"{name}.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")
