from __future__ import annotations

import logging

import pyaudiowpatch as pyaudio

logger = logging.getLogger(__name__)


def list_loopback_devices(pa: pyaudio.PyAudio) -> list[dict]:
    try:
        devices = list(pa.get_loopback_device_info_generator())
    except OSError:
        logger.error("WASAPI is not available on this system")
        return []
    for d in devices:
        logger.debug(
            "Found loopback: [%d] %s (%d Hz, %d ch)",
            d["index"], d["name"], int(d["defaultSampleRate"]), d["maxInputChannels"],
        )
    return devices


def get_default_loopback(pa: pyaudio.PyAudio) -> dict | None:
    try:
        device = pa.get_default_wasapi_loopback()
        logger.info(
            "Default loopback: [%d] %s (%d Hz, %d ch)",
            device["index"], device["name"],
            int(device["defaultSampleRate"]), device["maxInputChannels"],
        )
        return device
    except OSError:
        logger.error("WASAPI is not available on this system")
        return None
    except LookupError:
        logger.error("No loopback device found. Check audio output device.")
        return None


def get_default_input_device(pa: pyaudio.PyAudio) -> dict | None:
    try:
        info = pa.get_default_input_device_info()
        if info["maxInputChannels"] > 0:
            logger.info(
                "Default input: [%d] %s (%d Hz, %d ch)",
                info["index"], info["name"],
                int(info["defaultSampleRate"]), info["maxInputChannels"],
            )
            return info
    except OSError:
        logger.warning("No input device available")
    return None


def resolve_device(pa: pyaudio.PyAudio, saved_index: int | None) -> dict | None:
    if saved_index is not None:
        try:
            info = pa.get_device_info_by_index(saved_index)
            if info.get("isLoopbackDevice"):
                logger.info("Using saved device: [%d] %s", saved_index, info["name"])
                return info
        except Exception:
            pass
        logger.warning("Saved device index %d no longer valid, using default", saved_index)
    return get_default_loopback(pa)
