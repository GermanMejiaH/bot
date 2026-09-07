"""Unit tests for DatasetCaptureService dataset generation, deduplication, and classification."""

import json
import os
import shutil
import tempfile
import time

import cv2
import numpy as np

from dta.config.settings import Settings
from dta.events.event_bus import EventBus
from dta.events.events import GameStateUpdated
from dta.models.detections import ResourceDetection
from dta.models.game_state import GameState
from dta.models.perception_state import PerceptionState
from dta.services.dataset_capture import DatasetCaptureService


def test_dataset_capture_directory_creation() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        dataset_dir = os.path.join(temp_dir, "test_dataset")
        service = DatasetCaptureService(dataset_dir=dataset_dir)

        assert service.dataset_dir == dataset_dir
        assert os.path.exists(dataset_dir)
        assert os.path.exists(os.path.join(dataset_dir, "raw"))
        assert os.path.exists(os.path.join(dataset_dir, "combat"))
        assert os.path.exists(os.path.join(dataset_dir, "exploration"))
        assert os.path.exists(os.path.join(dataset_dir, "validation"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataset_capture_phash_and_frame_difference() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        service = DatasetCaptureService(dataset_dir=temp_dir)

        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        img3 = np.full((100, 100, 3), 255, dtype=np.uint8)

        phash1 = service.compute_phash(img1)
        phash2 = service.compute_phash(img2)
        phash3 = service.compute_phash(img3)

        assert isinstance(phash1, str)
        assert len(phash1) == 16
        assert phash1 == phash2
        assert len(phash3) == 16

        diff_identical = service.compute_frame_difference(img1, img2)
        diff_distinct = service.compute_frame_difference(img1, img3)

        assert diff_identical == 0.0
        assert diff_distinct > 0.5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataset_capture_combat_classification() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        service = DatasetCaptureService(dataset_dir=temp_dir)

        # 1. Exploration GameState (PA=0, PM=0)
        p_state_exp = PerceptionState(
            frame_id="frame_exp_001",
            resources=ResourceDetection(pa=0, pm=0, hp=5000),
        )
        g_state_exp = GameState(frame_id="frame_exp_001", perception_state=p_state_exp)

        # 2. Combat GameState (PA=11, PM=6)
        p_state_combat = PerceptionState(
            frame_id="frame_combat_001",
            resources=ResourceDetection(pa=11, pm=6, hp=5000),
        )
        g_state_combat = GameState(frame_id="frame_combat_001", perception_state=p_state_combat)

        assert not service.is_combat_active(g_state_exp)
        assert service.is_combat_active(g_state_combat)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataset_capture_frame_saving_and_metadata() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        settings = Settings()
        settings.dataset.save_interval_seconds = 0.0
        settings.dataset.min_frame_difference = 0.01

        service = DatasetCaptureService(settings=settings, dataset_dir=temp_dir)

        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)

        p_state = PerceptionState(
            frame_id="frame_001",
            resources=ResourceDetection(pa=11, pm=6, hp=5279),
        )
        g_state = GameState(frame_id="frame_001", perception_state=p_state, frame_image=img)

        res = service.capture_frame(g_state)
        assert res is not None
        assert "image" in res
        assert "json" in res
        assert "raw_image" in res

        assert os.path.exists(res["image"])
        assert os.path.exists(res["json"])
        assert os.path.exists(res["raw_image"])
        assert "combat" in res["image"]

        with open(res["json"], encoding="utf-8") as f:
            data = json.load(f)

        assert data["frame_id"] == "frame_001"
        assert data["combat"] is True
        assert data["resolution"] == [1920, 1080]
        assert data["pa"] == 11
        assert data["pm"] == 6
        assert data["hp"] == 5279
        assert "phash" in data
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataset_capture_deduplication_skipping() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        settings = Settings()
        settings.dataset.save_interval_seconds = 0.0
        settings.dataset.min_frame_difference = 0.05

        service = DatasetCaptureService(settings=settings, dataset_dir=temp_dir)

        img_base = np.zeros((200, 200, 3), dtype=np.uint8)
        p_state = PerceptionState(frame_id="frame_base", resources=ResourceDetection(pa=0, pm=0))
        g_state1 = GameState(frame_id="frame_base", perception_state=p_state, frame_image=img_base)

        # First capture: saved
        res1 = service.capture_frame(g_state1)
        assert res1 is not None

        # Identical second frame: skipped due to deduplication
        g_state2 = GameState(frame_id="frame_dup", perception_state=p_state, frame_image=img_base)
        res2 = service.capture_frame(g_state2)
        assert res2 is None

        # Distinct third frame: saved
        img_distinct = np.full((200, 200, 3), 255, dtype=np.uint8)
        g_state3 = GameState(frame_id="frame_distinct", perception_state=p_state, frame_image=img_distinct)
        res3 = service.capture_frame(g_state3)
        assert res3 is not None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataset_capture_event_bus_integration() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        event_bus = EventBus()
        settings = Settings()
        settings.dataset.save_interval_seconds = 0.0

        service = DatasetCaptureService(settings=settings, event_bus=event_bus, dataset_dir=temp_dir)
        service.start_listening()
        assert service.is_listening

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        p_state = PerceptionState(frame_id="frame_ev_1", resources=ResourceDetection(pa=0, pm=0))
        g_state = GameState(frame_id="frame_ev_1", perception_state=p_state, frame_image=img)

        event_bus.publish(GameStateUpdated(state=g_state))
        time.sleep(0.05)

        assert service.captured_count == 1

        service.stop_listening()
        assert not service.is_listening
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
