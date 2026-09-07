"""Window resolution and bounding box locator using PyGetWindow / Win32 API."""

import sys
from typing import Any

from dta.core.logger import logger


class WindowFinder:
    """Utility class to find game windows and retrieve pixel bounding boxes."""

    DOFUS_CLASSES = (
        "iop",
        "cra",
        "eniripsa",
        "selotrop",
        "sacrier",
        "feca",
        "xelor",
        "panda",
        "osamodas",
        "ecaflip",
        "enutrof",
        "sram",
        "sadida",
        "zobal",
        "steamer",
        "eliotrope",
        "hupper",
        "ouginak",
        "forgelance",
    )

    @classmethod
    def _is_dofus_title(cls, title: str) -> bool:
        """Check if window title matches Dofus game client patterns (e.g. 'Character - Class - Version - Release')."""
        t = title.lower().strip()
        if not t:
            return False
        if "dofus" in t:
            return True
        if t.endswith("- release") or "- release" in t:
            return True
        if any(f"- {c} -" in t or f" {c} " in f" {t} " for c in cls.DOFUS_CLASSES):
            return True
        return False

    @classmethod
    def find_window(cls, window_title: str) -> Any | None:
        """Find PyGetWindow window object matching window_title exact string, Dofus auto-detection, or substring."""
        target = window_title.strip().lower()
        if not target:
            return None

        try:
            import pygetwindow as gw  # Dynamic import to support runtime mocking and platform checks

            # Collect active, visible, non-minimized desktop windows
            get_all_fn: Any = getattr(gw, "getAllWindows", lambda: [])
            all_windows: list[Any] = get_all_fn()
            valid_windows: list[Any] = []

            if all_windows:
                for w in all_windows:
                    try:
                        visible = getattr(w, "visible", True)
                        minimized = getattr(w, "isMinimized", False)
                        w_width = int(getattr(w, "width", 1))
                        w_height = int(getattr(w, "height", 1))
                        if visible and not minimized and w_width > 0 and w_height > 0:
                            valid_windows.append(w)
                    except Exception:
                        valid_windows.append(w)

            # Priority 1: Exact window title match
            for w in valid_windows:
                title = str(getattr(w, "title", "")).strip().lower()
                if title == target:
                    return w

            # Priority 2: Automatic Dofus client pattern match (e.g. 'Character - Selotrop - 3.6.10.11 - Release')
            if target == "dofus" or "dofus" in target:
                for w in valid_windows:
                    title = str(getattr(w, "title", "")).strip()
                    if cls._is_dofus_title(title):
                        logger.info(f"Auto-detected Dofus client window: '{title}'")
                        return w

            # Priority 3: Substring search across active windows
            for w in valid_windows:
                title = str(getattr(w, "title", "")).strip().lower()
                if target in title:
                    return w

            # Priority 4: PyGetWindow native fallback
            title_matches = gw.getWindowsWithTitle(window_title)
            if title_matches:
                return title_matches[0]

        except ImportError:
            logger.warning("pygetwindow library is not installed/supported on this platform.")
        except Exception as e:
            logger.warning(f"Error querying window title '{window_title}': {e}")
        return None

    @classmethod
    def get_hwnd(cls, window_title: str) -> int | None:
        """Retrieve Win32 HWND window handle integer for target window."""
        window = cls.find_window(window_title)
        if window is None:
            return None
        try:
            hwnd = getattr(window, "_hWnd", None)
            if hwnd is not None and isinstance(hwnd, int):
                return hwnd
        except Exception:
            pass
        return None

    @classmethod
    def is_window_available(cls, window_title: str) -> bool:
        """Check if target window is currently open and visible."""
        window = cls.find_window(window_title)
        if window is None:
            return False
        try:
            return bool(getattr(window, "visible", True) and not getattr(window, "isMinimized", False))
        except Exception:
            return True

    @classmethod
    def get_window_bounds(cls, window_title: str) -> tuple[int, int, int, int] | None:
        """Get (left, top, width, height) pixel bounding box for target window client area."""
        window = cls.find_window(window_title)
        if window is None:
            return None

        # On Windows, extract exact inner Client Area via Win32 API to exclude title bars & borders
        if sys.platform == "win32":
            hwnd = cls.get_hwnd(window_title)
            if hwnd:
                try:
                    import ctypes
                    from ctypes import wintypes

                    class RECT(ctypes.Structure):
                        _fields_ = [
                            ("left", wintypes.LONG),
                            ("top", wintypes.LONG),
                            ("right", wintypes.LONG),
                            ("bottom", wintypes.LONG),
                        ]

                    class POINT(ctypes.Structure):
                        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

                    user32 = ctypes.windll.user32
                    rect = RECT()
                    if user32.GetClientRect(hwnd, ctypes.byref(rect)):
                        width = rect.right - rect.left
                        height = rect.bottom - rect.top
                        pt = POINT(0, 0)
                        user32.ClientToScreen(hwnd, ctypes.byref(pt))
                        left = pt.x
                        top = pt.y
                        if width > 0 and height > 0:
                            logger.info(
                                f"[DTA Window Finder]\n"
                                f"  Title: {window_title}\n"
                                f"  HWND: {hex(hwnd)} ({hwnd})\n"
                                f"  Resolution: {width}x{height}\n"
                                f"  Client Area: OK (left={left}, top={top})"
                            )
                            return (left, top, width, height)
                except Exception as exc:
                    logger.warning(f"Win32 Client Area query failed for HWND {hwnd}: {exc}")

        # Fallback to PyGetWindow bounds if Win32 API is not applicable/available
        try:
            left = int(getattr(window, "left", 0))
            top = int(getattr(window, "top", 0))
            width = int(getattr(window, "width", 0))
            height = int(getattr(window, "height", 0))
            if width > 0 and height > 0:
                return (left, top, width, height)
        except Exception as e:
            logger.warning(f"Failed to get bounds for window '{window_title}': {e}")
        return None

    @classmethod
    def get_window_resolution(cls, window_title: str) -> tuple[int, int] | None:
        """Get (width, height) pixel dimensions for target window."""
        bounds = cls.get_window_bounds(window_title)
        if bounds:
            return (bounds[2], bounds[3])
        return None

