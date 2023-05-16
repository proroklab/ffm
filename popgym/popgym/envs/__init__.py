"""Collection of POPGym environments
"""

import inspect
from importlib import find_loader
from typing import Any, Dict

import gymnasium as gym

from popgym.envs.autoencode import (
    Autoencode,
    AutoencodeEasy,
    AutoencodeHard,
    AutoencodeMedium,
)
from popgym.envs.battleship import (
    Battleship,
    BattleshipEasy,
    BattleshipHard,
    BattleshipMedium,
)
from popgym.envs.concentration import (
    Concentration,
    ConcentrationEasy,
    ConcentrationHard,
    ConcentrationMedium,
)
from popgym.envs.count_recall import (
    CountRecall,
    CountRecallEasy,
    CountRecallHard,
    CountRecallMedium,
)
from popgym.envs.higher_lower import (
    HigherLower,
    HigherLowerEasy,
    HigherLowerHard,
    HigherLowerMedium,
)
from popgym.envs.minesweeper import (
    MineSweeper,
    MineSweeperEasy,
    MineSweeperHard,
    MineSweeperMedium,
)
from popgym.envs.multiarmed_bandit import (
    MultiarmedBandit,
    MultiarmedBanditEasy,
    MultiarmedBanditHard,
    MultiarmedBanditMedium,
)
from popgym.envs.noisy_stateless_cartpole import (
    NoisyStatelessCartPole,
    NoisyStatelessCartPoleEasy,
    NoisyStatelessCartPoleHard,
    NoisyStatelessCartPoleMedium,
)
from popgym.envs.noisy_stateless_pendulum import (
    NoisyStatelessPendulum,
    NoisyStatelessPendulumEasy,
    NoisyStatelessPendulumHard,
    NoisyStatelessPendulumMedium,
)
from popgym.envs.repeat_first import (
    RepeatFirst,
    RepeatFirstEasy,
    RepeatFirstHard,
    RepeatFirstMedium,
)
from popgym.envs.repeat_previous import (
    RepeatPrevious,
    RepeatPreviousEasy,
    RepeatPreviousHard,
    RepeatPreviousMedium,
)
from popgym.envs.stateless_cartpole import (
    StatelessCartPole,
    StatelessCartPoleEasy,
    StatelessCartPoleHard,
    StatelessCartPoleMedium,
)
from popgym.envs.stateless_pendulum import (
    StatelessPendulum,
    StatelessPendulumEasy,
    StatelessPendulumHard,
    StatelessPendulumMedium,
)

#
# Simple envs
#
DIAGNOSTIC: Dict[gym.Env, Dict[str, Any]] = {
    Autoencode: {"id": "popgym-Autoencode-v0"},
    RepeatPrevious: {"id": "popgym-RepeatPrevious-v0"},
    RepeatFirst: {"id": "popgym-RepeatFirst-v0"},
    CountRecall: {"id": "popgym-CountRecall-v0"},
}

DIAGNOSTIC_EASY: Dict[gym.Env, Dict[str, Any]] = {
    AutoencodeEasy: {"id": "popgym-AutoencodeEasy-v0"},
    RepeatPreviousEasy: {"id": "popgym-RepeatPreviousEasy-v0"},
    RepeatFirstEasy: {"id": "popgym-RepeatFirstEasy-v0"},
    CountRecallEasy: {"id": "popgym-CountRecallEasy-v0"},
}

DIAGNOSTIC_MEDIUM: Dict[gym.Env, Dict[str, Any]] = {
    AutoencodeMedium: {"id": "popgym-AutoencodeMedium-v0"},
    RepeatPreviousMedium: {"id": "popgym-RepeatPreviousMedium-v0"},
    RepeatFirstMedium: {"id": "popgym-RepeatFirstMedium-v0"},
    CountRecallMedium: {"id": "popgym-CountRecallMedium-v0"},
}

DIAGNOSTIC_HARD: Dict[gym.Env, Dict[str, Any]] = {
    AutoencodeHard: {"id": "popgym-AutoencodeHard-v0"},
    RepeatPreviousHard: {"id": "popgym-RepeatPreviousHard-v0"},
    RepeatFirstHard: {"id": "popgym-RepeatFirstHard-v0"},
    CountRecallHard: {"id": "popgym-CountRecallHard-v0"},
}

ALL_DIAGNOSTIC = {**DIAGNOSTIC_EASY, **DIAGNOSTIC_MEDIUM, **DIAGNOSTIC_HARD}

#
# Control envs
#
CONTROL: Dict[gym.Env, Dict[str, Any]] = {
    StatelessCartPole: {"id": "popgym-StatelessCartPole-v0"},
    StatelessPendulum: {
        "id": "popgym-StatelessPendulum-v0",
    },
    # BipedalWalker: {"id": "popgym-BipedalWalker-v0"},
}

CONTROL_EASY: Dict[gym.Env, Dict[str, Any]] = {
    StatelessCartPoleEasy: {"id": "popgym-StatelessCartPoleEasy-v0"},
    StatelessPendulumEasy: {
        "id": "popgym-StatelessPendulumEasy-v0",
    },
    # BipedalWalkerEasy: {"id": "popgym-BipedalWalkerEasy-v0"},
}

CONTROL_MEDIUM: Dict[gym.Env, Dict[str, Any]] = {
    StatelessCartPoleMedium: {"id": "popgym-StatelessCartPoleMedium-v0"},
    StatelessPendulumMedium: {
        "id": "popgym-StatelessPendulumMedium-v0",
    },
    # BipedalWalkerMedium: {"id": "popgym-BipedalWalkerMedium-v0"},
}

CONTROL_HARD: Dict[gym.Env, Dict[str, Any]] = {
    StatelessCartPoleHard: {"id": "popgym-StatelessCartPoleHard-v0"},
    StatelessPendulumHard: {
        "id": "popgym-StatelessPendulumHard-v0",
    },
    # BipedalWalkerHard: {"id": "popgym-BipedalWalkerHard-v0"},
}

ALL_CONTROL = {**CONTROL_EASY, **CONTROL_MEDIUM, **CONTROL_HARD}

#
# Noisy envs
#
NOISY: Dict[gym.Env, Dict[str, Any]] = {
    NoisyStatelessCartPole: {"id": "popgym-NoisyStatelessCartPole-v0"},
    NoisyStatelessPendulum: {
        "id": "popgym-NoisyStatelessPendulum-v0",
    },
    MultiarmedBandit: {"id": "popgym-MultiarmedBandit-v0"},
    # NonstationaryBandit: {"id": "popgym-NonstationaryBandit-v0"},
}

NOISY_EASY: Dict[gym.Env, Dict[str, Any]] = {
    NoisyStatelessCartPoleEasy: {"id": "popgym-NoisyStatelessCartPoleEasy-v0"},
    NoisyStatelessPendulumEasy: {
        "id": "popgym-NoisyStatelessPendulumEasy-v0",
    },
    MultiarmedBanditEasy: {"id": "popgym-MultiarmedBanditEasy-v0"},
    # NonstationaryBanditEasy: {"id": "popgym-NonstationaryBanditEasy-v0"},
}

NOISY_MEDIUM: Dict[gym.Env, Dict[str, Any]] = {
    NoisyStatelessCartPoleMedium: {"id": "popgym-NoisyStatelessCartPoleMedium-v0"},
    NoisyStatelessPendulumMedium: {
        "id": "popgym-NoisyStatelessPendulumMedium-v0",
    },
    MultiarmedBanditMedium: {"id": "popgym-MultiarmedBanditMedium-v0"},
    # NonstationaryBanditMedium: {"id": "popgym-NonstationaryBanditMedium-v0"},
}

NOISY_HARD: Dict[gym.Env, Dict[str, Any]] = {
    NoisyStatelessCartPoleHard: {"id": "popgym-NoisyStatelessCartPoleHard-v0"},
    NoisyStatelessPendulumHard: {
        "id": "popgym-NoisyStatelessPendulumHard-v0",
    },
    MultiarmedBanditHard: {"id": "popgym-MultiarmedBanditHard-v0"},
    # NonstationaryBanditHard: {"id": "popgym-NonstationaryBanditHard-v0"},
}

ALL_NOISY = {**NOISY_EASY, **NOISY_MEDIUM, **NOISY_HARD}

#
# Game envs
#
GAME: Dict[gym.Env, Dict[str, Any]] = {
    HigherLower: {"id": "popgym-HigherLower-v0"},
    Battleship: {"id": "popgym-Battleship-v0"},
    Concentration: {"id": "popgym-Concentration-v0"},
    MineSweeper: {"id": "popgym-MineSweeper-v0"},
}

GAME_EASY: Dict[gym.Env, Dict[str, Any]] = {
    HigherLowerEasy: {"id": "popgym-HigherLowerEasy-v0"},
    BattleshipEasy: {"id": "popgym-BattleshipEasy-v0"},
    ConcentrationEasy: {"id": "popgym-ConcentrationEasy-v0"},
    MineSweeperEasy: {"id": "popgym-MineSweeperEasy-v0"},
}

GAME_MEDIUM: Dict[gym.Env, Dict[str, Any]] = {
    HigherLowerMedium: {"id": "popgym-HigherLowerMedium-v0"},
    BattleshipMedium: {"id": "popgym-BattleshipMedium-v0"},
    ConcentrationMedium: {"id": "popgym-ConcentrationMedium-v0"},
    MineSweeperMedium: {"id": "popgym-MineSweeperMedium-v0"},
}

GAME_HARD: Dict[gym.Env, Dict[str, Any]] = {
    HigherLowerHard: {"id": "popgym-HigherLowerHard-v0"},
    BattleshipHard: {"id": "popgym-BattleshipHard-v0"},
    ConcentrationHard: {"id": "popgym-ConcentrationHard-v0"},
    MineSweeperHard: {"id": "popgym-MineSweeperHard-v0"},
}

ALL_GAME = {**GAME_EASY, **GAME_MEDIUM, **GAME_HARD}

#
# Navigation envs
#
def has_mazelib():  # noqa: E302
    """Check if mazelib is installed"""
    return find_loader("mazelib") is not None


if has_mazelib():
    # mazelib can somtimes be a headache to install
    # mazes are also a poor test of memory
    from popgym.envs.labyrinth_escape import (
        LabyrinthEscape,
        LabyrinthEscapeEasy,
        LabyrinthEscapeHard,
        LabyrinthEscapeMedium,
    )
    from popgym.envs.labyrinth_explore import (
        LabyrinthExplore,
        LabyrinthExploreEasy,
        LabyrinthExploreHard,
        LabyrinthExploreMedium,
    )

    NAVIGATION: Dict[gym.Env, Dict[str, Any]] = {
        LabyrinthExplore: {"id": "popgym-LabyrinthExplore-v0"},
        LabyrinthEscape: {"id": "popgym-LabyrinthEscape-v0"},
    }

    NAVIGATION_EASY: Dict[gym.Env, Dict[str, Any]] = {
        LabyrinthExploreEasy: {"id": "popgym-LabyrinthExploreEasy-v0"},
        LabyrinthEscapeEasy: {"id": "popgym-LabyrinthEscapeEasy-v0"},
    }

    NAVIGATION_MEDIUM: Dict[gym.Env, Dict[str, Any]] = {
        LabyrinthExploreMedium: {"id": "popgym-LabyrinthExploreMedium-v0"},
        LabyrinthEscapeMedium: {"id": "popgym-LabyrinthEscapeMedium-v0"},
    }

    NAVIGATION_HARD: Dict[gym.Env, Dict[str, Any]] = {
        LabyrinthExploreHard: {"id": "popgym-LabyrinthExploreHard-v0"},
        LabyrinthEscapeHard: {"id": "popgym-LabyrinthEscapeHard-v0"},
    }
else:
    NAVIGATION = {}
    NAVIGATION_EASY = {}
    NAVIGATION_MEDIUM = {}
    NAVIGATION_HARD = {}

ALL_NAVIGATION = {
    **NAVIGATION_EASY,
    **NAVIGATION_MEDIUM,
    **NAVIGATION_HARD,
}

#
# Sets of envs
#
ALL_BASE: Dict[gym.Env, Dict[str, Any]] = {
    **DIAGNOSTIC,
    **CONTROL,
    **NOISY,
    **GAME,
    **NAVIGATION,
}
ALL_EASY: Dict[gym.Env, Dict[str, Any]] = {
    **DIAGNOSTIC_EASY,
    **CONTROL_EASY,
    **NOISY_EASY,
    **GAME_EASY,
    **NAVIGATION_EASY,
}
ALL_MEDIUM: Dict[gym.Env, Dict[str, Any]] = {
    **DIAGNOSTIC_MEDIUM,
    **CONTROL_MEDIUM,
    **NOISY_MEDIUM,
    **GAME_MEDIUM,
    **NAVIGATION_MEDIUM,
}
ALL_HARD: Dict[gym.Env, Dict[str, Any]] = {
    **DIAGNOSTIC_HARD,
    **CONTROL_HARD,
    **NOISY_HARD,
    **GAME_HARD,
    **NAVIGATION_HARD,
}
ALL: Dict[gym.Env, Dict[str, Any]] = {
    **ALL_EASY,
    **ALL_MEDIUM,
    **ALL_HARD,
}

# Register envs
for e, v in ALL.items():
    mod_name = inspect.getmodule(e).__name__  # type: ignore
    gym.envs.register(
        entry_point=":".join([mod_name, e.__name__]), order_enforce=False, **v
    )
# For Sphinx automodule docs
del e
