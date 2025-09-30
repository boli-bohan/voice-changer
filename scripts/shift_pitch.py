#!/usr/bin/env python3
"""
Audio pitch shifting script for the voice changer app.

Usage:
    python shift_pitch.py input_file.wav output_file.wav [pitch_shift_semitones]

Arguments:
    input_file: Path to the input WAV file
    output_file: Path for the output WAV file
    pitch_shift_semitones: Number of semitones to shift (default: 4.0)
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# Suppress deprecated audioread warnings
warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*audioread.*", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def shift_pitch(input_file: str, output_file: str, pitch_shift_semitones: float = 4.0) -> bool:
    """Apply a pitch shift to the provided audio file.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Destination path for the processed audio file.
        pitch_shift_semitones (float): Number of semitones to shift the pitch.

    Returns:
        bool: ``True`` if processing succeeds, ``False`` otherwise.
    """
    try:
        logger.info(f"Loading audio from: {input_file}")

        # Load audio file
        audio_data, sample_rate = librosa.load(input_file, sr=None, mono=False)

        logger.info(f"Audio loaded: {audio_data.shape} at {sample_rate} Hz")
        logger.info(f"Applying pitch shift of {pitch_shift_semitones} semitones")

        # Apply pitch shifting
        if len(audio_data.shape) == 1:
            # Mono audio
            shifted_audio = librosa.effects.pitch_shift(
                audio_data,
                sr=sample_rate,
                n_steps=pitch_shift_semitones
            )
        else:
            # Stereo audio - process each channel separately
            shifted_audio = np.array([
                librosa.effects.pitch_shift(
                    audio_data[channel],
                    sr=sample_rate,
                    n_steps=pitch_shift_semitones
                )
                for channel in range(audio_data.shape[0])
            ])

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving processed audio to: {output_file}")

        # Save the processed audio
        sf.write(output_file, shifted_audio.T if len(shifted_audio.shape) > 1 else shifted_audio, sample_rate)

        logger.info("Pitch shifting completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return False


def main():
    """Parse CLI arguments and run the pitch shifting workflow."""
    parser = argparse.ArgumentParser(
        description="Apply pitch shifting to audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shift_pitch.py input.wav output.wav
  python shift_pitch.py input.wav output.wav 6.0
  python shift_pitch.py input.wav output.wav -2.0
        """
    )

    parser.add_argument("input_file", help="Input audio file path")
    parser.add_argument("output_file", help="Output audio file path")
    parser.add_argument(
        "pitch_shift",
        nargs="?",
        type=float,
        default=4.0,
        help="Pitch shift in semitones (default: 4.0)"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Validate pitch shift range (reasonable limits)
    if abs(args.pitch_shift) > 12:
        logger.warning(f"Large pitch shift detected: {args.pitch_shift} semitones")
        logger.warning("Results may sound unnatural")

    logger.info(f"Starting pitch shift: {args.input_file} -> {args.output_file}")
    logger.info(f"Pitch shift: {args.pitch_shift} semitones")

    success = shift_pitch(args.input_file, args.output_file, args.pitch_shift)

    if success:
        logger.info("Process completed successfully")
        sys.exit(0)
    else:
        logger.error("Process failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
