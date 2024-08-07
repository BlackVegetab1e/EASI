from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL
from .GARAT import GARAT, GARATTrainer

ALGOS = {
    'gail': GAIL,
    'airl': AIRL
}
