"""Unit tests for DatasetCaptureService dataset generation, deduplication, and classification."""

import json
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dta.config.settings import Settings
from dta.events.event_bus import EventBus
from dta.events.events import GameStateUpdated
from dta.models.combat import CombatState
from dta.models.detections import ResourceDetection
from dta.models.game_state import GameState
from dta.models.perception_state import PerceptionState
from dta.services.dataset_capture import DatasetCaptureService
from scripts.analyze_dataset import analyze_dataset


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

        from dta.models.combat import CombatState

        # 2. Combat GameState (Turn Banner Active)
        p_state_combat = PerceptionState(
            frame_id="frame_combat_001",
            resources=ResourceDetection(pa=11, pm=6, hp=5000),
        )
        g_state_combat = GameState(frame_id="frame_combat_001", perception_state=p_state_combat, combat_state=CombatState())

        assert not service.is_combat_active(g_state_exp)
        assert service.is_combat_active(g_state_combat)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataset_capture_frame_saving_and_metadata() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        from dta.models.combat import CombatState

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
        g_state = GameState(frame_id="frame_001", perception_state=p_state, combat_state=CombatState(), frame_image=img)

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


def test_dataset_capture_metrics_and_report_generation() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        settings = Settings()
        settings.dataset.save_interval_seconds = 0.0
        settings.dataset.min_frame_difference = 0.05

        service = DatasetCaptureService(settings=settings, dataset_dir=temp_dir)

        img1 = np.zeros((200, 200, 3), dtype=np.uint8)
        img2 = np.full((200, 200, 3), 255, dtype=np.uint8)

        # Frame 1: Combat (Saved)
        p_combat = PerceptionState(frame_id="f1", resources=ResourceDetection(pa=11, pm=6))
        g1 = GameState(frame_id="f1", perception_state=p_combat, combat_state=CombatState(), frame_image=img1)
        service.capture_frame(g1)

        # Frame 2: Identical (Duplicate Skipped)
        g2 = GameState(frame_id="f2", perception_state=p_combat, combat_state=CombatState(), frame_image=img1)
        service.capture_frame(g2)

        # Frame 3: Exploration Distinct (Saved)
        p_exp = PerceptionState(frame_id="f3", resources=ResourceDetection(pa=0, pm=0))
        g3 = GameState(frame_id="f3", perception_state=p_exp, frame_image=img2)
        service.capture_frame(g3)

        metrics = service.get_metrics()
        assert metrics.total_frames_seen == 3
        assert metrics.total_frames_saved == 2
        assert metrics.duplicate_frames_skipped == 1
        assert metrics.combat_frames_saved == 1
        assert metrics.exploration_frames_saved == 1
        assert metrics.average_frame_difference > 0.0

        report_path = service.generate_session_report("report.json")
        assert os.path.exists(report_path)

        with open(report_path, encoding="utf-8") as rf:
            rep = json.load(rf)

        assert "metrics" in rep
        assert rep["metrics"]["total_frames_seen"] == 3
        assert rep["metrics"]["total_frames_saved"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataset_analysis_script() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        settings = Settings()
        settings.dataset.save_interval_seconds = 0.0
        service = DatasetCaptureService(settings=settings, dataset_dir=temp_dir)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        p_state = PerceptionState(frame_id="f_an", resources=ResourceDetection(pa=6, pm=3))
        g_state = GameState(frame_id="f_an", perception_state=p_state, frame_image=img)
        service.capture_frame(g_state)
        service.generate_session_report()

        analysis = analyze_dataset(temp_dir)
        assert analysis["total_images"] >= 1
        assert "combat" in analysis["category_counts"]
        assert analysis["report_data"] is not None
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
        assert os.path.exists(os.path.join(temp_dir, "report.json"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
