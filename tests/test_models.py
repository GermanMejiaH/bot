"""Unit tests for Stage 1 models (combat, game_state, detections)."""

import time

from dta.models.combat import Ally, CombatState, Enemy, Player, Position
from dta.models.detections import BoundingBox, CharacterDetection, DetectionBatch, ResourceDetection
from dta.models.game_state import GameState


def test_position_model() -> None:
    pos = Position(pixel_x=100, pixel_y=200, grid_x=5, grid_y=10, cell_id=142)
    assert pos.pixel_x == 100
    assert pos.pixel_y == 200
    assert pos.grid_x == 5
    assert pos.grid_y == 10
    assert pos.cell_id == 142


def test_combat_entities_and_state() -> None:
    pos_player = Position(pixel_x=100, pixel_y=150, cell_id=12)
    player = Player(id="player_01", name="Iop", hp=1000, max_hp=1000, position=pos_player, pa=12, pm=6)

    pos_enemy = Position(pixel_x=300, pixel_y=400, cell_id=45)
    enemy = Enemy(id="enemy_01", name="Gobball", hp=200, max_hp=200, position=pos_enemy, is_boss=False)

    pos_ally = Position(pixel_x=150, pixel_y=200, cell_id=20)
    ally = Ally(id="ally_01", name="Cra", hp=850, max_hp=850, position=pos_ally)

    obstacle = Position(pixel_x=500, pixel_y=500, cell_id=99)

    combat_state = CombatState(
        turn=3,
        pa=12,
        pm=6,
        map_width=14,
        map_height=20,
        obstacles=[obstacle],
        occupied_cells=[pos_player, pos_enemy, pos_ally],
        player=player,
        allies=[ally],
        enemies=[enemy],
    )

    assert combat_state.turn == 3
    assert combat_state.player is not None
    assert combat_state.player.pa == 12
    assert len(combat_state.enemies) == 1
    assert combat_state.enemies[0].name == "Gobball"
    assert len(combat_state.obstacles) == 1


def test_game_state_model() -> None:
    now = time.time()
    game_state = GameState(
        frame_id="frame_0001",
        timestamp=now,
        raw_frame_path="screenshots/raw/frame_0001.png",
        debug_frame_path="screenshots/debug/frame_0001.png",
    )

    assert game_state.frame_id == "frame_0001"
    assert game_state.timestamp == now
    assert game_state.combat_state is None


def test_detection_models() -> None:
    bbox = BoundingBox(x=10, y=20, w=30, h=40)
    assert bbox.centroid == (25, 40)

    char_det = CharacterDetection(label="enemy", bbox=bbox, confidence=0.95, centroid=bbox.centroid)
    res_det = ResourceDetection(pa=11, pm=5, hp=920)

    batch = DetectionBatch(frame_id="f1", timestamp=time.time(), characters=[char_det], resources=res_det)

    assert len(batch.characters) == 1
    assert batch.resources.pa == 11
    assert batch.characters[0].label == "enemy"
