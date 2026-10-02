from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from typing import Any

logger = logging.getLogger(__name__)

LOOPBACK_SUFFIX = " [Loopback]"
CLSID_MM_DEVICE_ENUMERATOR = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
IID_IMM_DEVICE_ENUMERATOR = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
CLSCTX_ALL = 23
COINIT_APARTMENTTHREADED = 2
RPC_E_CHANGED_MODE = -2147417850
E_RENDER = 0
ROLE_CONSOLE = 0
ROLE_MULTIMEDIA = 1
RELEASE_SLOT = 2
GET_DEFAULT_AUDIO_ENDPOINT_SLOT = 4
GET_ID_SLOT = 5
S_OK = 0


def get_default_loopback(port_audio: Any) -> dict | None:
    try:
        device = port_audio.get_default_wasapi_loopback()
        logger.info(
            "Default loopback: [%d] %s (%d Hz, %d ch)",
            device["index"], device["name"],
            int(device["defaultSampleRate"]), device["maxInputChannels"],
        )
        return dict(device)
    except OSError:
        logger.error("WASAPI is not available on this system")
        return None
    except LookupError:
        logger.error("No loopback device found. Check audio output device.")
        return None


def find_render_device(port_audio: Any, loopback: dict) -> dict | None:
    render_name = str(loopback["name"]).removesuffix(LOOPBACK_SUFFIX)
    try:
        wasapi = port_audio.get_host_api_info_by_type(_wasapi_type())
    except (OSError, LookupError):
        return None
    for host_index in range(int(wasapi.get("deviceCount", 0))):
        device = port_audio.get_device_info_by_host_api_device_index(wasapi["index"], host_index)
        is_render = int(device.get("maxOutputChannels", 0)) > 0 and not device.get("isLoopbackDevice", False)
        if is_render and device["name"] == render_name:
            return dict(device)
    return None


def get_default_input_device(port_audio: Any) -> dict | None:
    try:
        info = port_audio.get_default_input_device_info()
        if info["maxInputChannels"] > 0:
            logger.info(
                "Default input: [%d] %s (%d Hz, %d ch)",
                info["index"], info["name"],
                int(info["defaultSampleRate"]), info["maxInputChannels"],
            )
            return dict(info)
    except OSError:
        logger.warning("No input device available")
    return None


def _wasapi_type() -> int:
    import pyaudiowpatch as pyaudio

    return int(pyaudio.paWASAPI)


class _Guid(ctypes.Structure):
    _fields_ = [
        ("data1", wintypes.DWORD),
        ("data2", wintypes.WORD),
        ("data3", wintypes.WORD),
        ("data4", ctypes.c_ubyte * 8),
    ]


def _guid(text: str) -> _Guid:
    guid = _Guid()
    ctypes.oledll.ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(guid))
    return guid


def _method(interface: ctypes.c_void_p, slot: int, *argument_types: Any) -> Any:
    vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argument_types)
    return prototype(vtable[slot])


def _release(interface: ctypes.c_void_p) -> None:
    if interface:
        _method(interface, RELEASE_SLOT)(interface)


class DefaultOutputWatcher:
    def __init__(self) -> None:
        self._enumerator = ctypes.c_void_p()
        self._initialized = False
        if sys.platform != "win32":
            return
        result = ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        self._initialized = result in (S_OK, 1)
        if result not in (S_OK, 1, RPC_E_CHANGED_MODE):
            logger.warning("CoInitializeEx failed (0x%08X), output changes will not be detected", result & 0xFFFFFFFF)
            return
        class_id = _guid(CLSID_MM_DEVICE_ENUMERATOR)
        interface_id = _guid(IID_IMM_DEVICE_ENUMERATOR)
        result = ctypes.windll.ole32.CoCreateInstance(
            ctypes.byref(class_id), None, CLSCTX_ALL, ctypes.byref(interface_id), ctypes.byref(self._enumerator),
        )
        if result != S_OK:
            logger.warning("Could not create the audio device enumerator (0x%08X)", result & 0xFFFFFFFF)
            self._enumerator = ctypes.c_void_p()

    def current_id(self) -> str | None:
        if not self._enumerator:
            return None
        identifiers = [self._endpoint_id(role) for role in (ROLE_CONSOLE, ROLE_MULTIMEDIA)]
        return "|".join(identifier or "" for identifier in identifiers)

    def _endpoint_id(self, role: int) -> str | None:
        device = ctypes.c_void_p()
        get_default = _method(
            self._enumerator, GET_DEFAULT_AUDIO_ENDPOINT_SLOT, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
        )
        if get_default(self._enumerator, E_RENDER, role, ctypes.byref(device)) != S_OK:
            return None
        identifier_pointer = ctypes.c_void_p()
        try:
            get_id = _method(device, GET_ID_SLOT, ctypes.POINTER(ctypes.c_void_p))
            if get_id(device, ctypes.byref(identifier_pointer)) != S_OK:
                return None
            return ctypes.wstring_at(identifier_pointer.value) if identifier_pointer.value else None
        finally:
            if identifier_pointer.value:
                ctypes.windll.ole32.CoTaskMemFree(identifier_pointer)
            _release(device)

    def close(self) -> None:
        _release(self._enumerator)
        self._enumerator = ctypes.c_void_p()
        if self._initialized:
            ctypes.windll.ole32.CoUninitialize()
            self._initialized = False
