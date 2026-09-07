"""Combat domain models for Dofus Tactical Assistant."""

from pydantic import BaseModel, Field


class Position(BaseModel):
    """Position representation storing both 2D pixel coordinates and isometric grid cells."""

    pixel_x: int
    pixel_y: int
    grid_x: int | None = None
    grid_y: int | None = None
    cell_id: int | None = None


class CombatEntity(BaseModel):
    """Base class for any character or monster present on the combat map."""

    id: str
    name: str
    hp: int = 0
    max_hp: int = 0
    position: Position
    states: list[str] = Field(default_factory=list)


class Player(CombatEntity):
    """Controlled player entity with current action and movement points."""

    pa: int = 0
    pm: int = 0


class Enemy(CombatEntity):
    """Opponent entity."""

    is_boss: bool = False


class Ally(CombatEntity):
    """Friendly or summons entity."""

    pass


class CombatState(BaseModel):
    """Comprehensive snapshot of the combat board state."""

    turn: int = 1
    pa: int = 0
    pm: int = 0
    map_width: int = 14
    map_height: int = 20
    obstacles: list[Position] = Field(default_factory=list)
    occupied_cells: list[Position] = Field(default_factory=list)
    player: Player | None = None
    allies: list[Ally] = Field(default_factory=list)
    enemies: list[Enemy] = Field(default_factory=list)
