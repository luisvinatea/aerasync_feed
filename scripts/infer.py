#!/usr/bin/env python3
"""
Real-time inference script for AeraSync Feed project.
Runs audio classification on Raspberry Pi using TensorFlow Lite model.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import librosa
import json
import time
import logging
from pathlib import Path
import threading
import queue
import signal
import sys

# For audio recording (you may need to install pyaudio: pip install pyaudio)
try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("PyAudio not available. Audio recording will be disabled.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioClassifier:
    """Real-time audio classifier using TensorFlow Lite."""

    def __init__(
        self,
        model_path,
        label_encoder_path=None,
        sample_rate=22050,
        chunk_duration=2.0,
        overlap=0.5,
    ):
        """
        Initialize the audio classifier.

        Args:
            model_path (str): Path to TensorFlow Lite model
            label_encoder_path (str): Path to label encoder JSON file
            sample_rate (int): Audio sample rate
            chunk_duration (float): Duration of audio chunks in seconds
            overlap (float): Overlap between chunks (0-1)
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap

        # Calculate chunk parameters
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.hop_samples = int(self.chunk_samples * (1 - overlap))

        # Load TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]
        self.output_shape = self.output_details[0]["shape"]

        logger.info(f"Model loaded: {model_path}")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Output shape: {self.output_shape}")

        # Load label encoder
        self.class_names = None
        if label_encoder_path and os.path.exists(label_encoder_path):
            with open(label_encoder_path, "r") as f:
                label_data = json.load(f)
                self.class_names = label_data.get("classes", None)
            logger.info(
                f"Loaded {len(self.class_names)} classes: {self.class_names}"
            )

        # Audio buffer for continuous processing
        self.audio_buffer = np.array([])
        self.is_running = False

    def extract_features(self, audio_data):
        """
        Extract MFCC features from audio data.

        Args:
            audio_data (np.array): Raw audio data

        Returns:
            np.array: Extracted features
        """
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=2048,
                hop_length=512,
            )

            # Pad or truncate to fixed length (128 time steps)
            target_length = 128
            if mfccs.shape[1] < target_length:
                mfccs = np.pad(
                    mfccs,
                    ((0, 0), (0, target_length - mfccs.shape[1])),
                    mode="constant",
                )
            else:
                mfccs = mfccs[:, :target_length]

            # Transpose to (time, features) and add batch dimension
            features = mfccs.T
            features = np.expand_dims(features, axis=0).astype(np.float32)

            return features

        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return None

    def classify_audio(self, audio_data):
        """
        Classify audio data using the TensorFlow Lite model.

        Args:
            audio_data (np.array): Raw audio data

        Returns:
            dict: Classification results
        """
        # Extract features
        features = self.extract_features(audio_data)

        if features is None:
            return None

        # Ensure features match expected input shape
        if features.shape != tuple(self.input_shape):
            logger.warning(
                f"Feature shape {features.shape} doesn't match expected {self.input_shape}"
            )
            return None

        try:
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]["index"], features
            )

            # Run inference
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = time.time() - start_time

            # Get output
            output_data = self.interpreter.get_tensor(
                self.output_details[0]["index"]
            )
            predictions = output_data[0]  # Remove batch dimension

            # Get predicted class
            predicted_class_idx = np.argmax(predictions)
            confidence = predictions[predicted_class_idx]

            predicted_class = (
                self.class_names[predicted_class_idx]
                if self.class_names
                else f"Class_{predicted_class_idx}"
            )

            return {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "probabilities": predictions.tolist(),
                "inference_time": inference_time,
            }

        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return None

    def classify_file(self, audio_file):
        """
        Classify audio from a file.

        Args:
            audio_file (str): Path to audio file

        Returns:
            dict: Classification results
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(audio_file, sr=self.sample_rate)

            logger.info(f"Loaded audio file: {audio_file}")
            logger.info(f"Duration: {len(audio_data) / sr:.2f} seconds")

            # Classify the audio
            result = self.classify_audio(audio_data)

            if result:
                logger.info(
                    f"Prediction: {result['predicted_class']} "
                    f"(confidence: {result['confidence']:.3f})"
                )

            return result

        except Exception as e:
            logger.error(f"Error processing file {audio_file}: {str(e)}")
            return None

    def start_realtime_classification(
        self, device_index=None, decision_callback=None
    ):
        """
        Start real-time audio classification.

        Args:
            device_index (int): Audio device index (None for default)
            decision_callback (function): Callback for classification decisions
        """
        if not PYAUDIO_AVAILABLE:
            logger.error(
                "PyAudio not available. Cannot start real-time classification."
            )
            return

        self.is_running = True

        # Audio recording parameters
        format = pyaudio.paFloat32
        channels = 1

        # Initialize PyAudio
        audio = pyaudio.PyAudio()

        try:
            # Open audio stream
            stream = audio.open(
                format=format,
                channels=channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
            )

            logger.info("Starting real-time audio classification...")
            logger.info("Press Ctrl+C to stop")

            while self.is_running:
                try:
                    # Read audio data
                    data = stream.read(1024, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.float32)

                    # Add to buffer
                    self.audio_buffer = np.append(
                        self.audio_buffer, audio_chunk
                    )

                    # Process when buffer has enough samples
                    if len(self.audio_buffer) >= self.chunk_samples:
                        # Extract chunk for classification
                        chunk = self.audio_buffer[: self.chunk_samples]

                        # Classify chunk
                        result = self.classify_audio(chunk)

                        if result:
                            logger.info(
                                f"Real-time prediction: {result['predicted_class']} "
                                f"(confidence: {result['confidence']:.3f}, "
                                f"time: {result['inference_time']:.3f}s)"
                            )

                            # Call decision callback if provided
                            if decision_callback:
                                decision_callback(result)

                        # Slide buffer
                        self.audio_buffer = self.audio_buffer[
                            self.hop_samples :
                        ]

                except Exception as e:
                    logger.error(f"Real-time processing error: {str(e)}")
                    break

        finally:
            # Clean up
            if "stream" in locals():
                stream.stop_stream()
                stream.close()
            audio.terminate()
            logger.info("Real-time classification stopped")


def feeding_decision_callback(result):
    """
    Example callback function for feeding decisions.

    Args:
        result (dict): Classification result
    """
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]

    # Example decision logic
    if predicted_class == "shrimp_chewing" and confidence > 0.7:
        logger.info(
            "🦐 FEEDING DECISION: ACTIVATE FEEDER - Shrimp chewing detected!"
        )
        # Here you would integrate with your feeder control system
        # e.g., GPIO control, API call, etc.
    elif predicted_class == "aerator_noise" and confidence > 0.8:
        logger.info(
            "🌊 FEEDING DECISION: HOLD FEEDING - Aerator noise detected"
        )
    else:
        logger.info(
            f"📊 FEEDING DECISION: UNCERTAIN - {predicted_class} ({confidence:.3f})"
        )


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("Stopping audio classification...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time audio classification for shrimp feeding"
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to TensorFlow Lite model (.tflite)",
    )
    parser.add_argument(
        "--labels", "-l", help="Path to label encoder JSON file"
    )
    parser.add_argument(
        "--file", "-f", help="Audio file to classify (instead of real-time)"
    )
    parser.add_argument(
        "--device", "-d", type=int, help="Audio device index for recording"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Audio chunk duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap between chunks 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Audio sample rate (default: 22050)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error(f"Model file does not exist: {args.model}")
        return

    # Initialize classifier
    classifier = AudioClassifier(
        model_path=args.model,
        label_encoder_path=args.labels,
        sample_rate=args.sample_rate,
        chunk_duration=args.duration,
        overlap=args.overlap,
    )

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    if args.file:
        # Classify single file
        result = classifier.classify_file(args.file)
        if result:
            print(f"\nClassification Result:")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Inference Time: {result['inference_time']:.3f}s")

            if classifier.class_names:
                print(f"\nAll Probabilities:")
                for i, (class_name, prob) in enumerate(
                    zip(classifier.class_names, result["probabilities"])
                ):
                    print(f"  {class_name}: {prob:.3f}")
    else:
        # Start real-time classification
        classifier.start_realtime_classification(
            device_index=args.device,
            decision_callback=feeding_decision_callback,
        )


if __name__ == "__main__":
    main()
