from src.train.off_policy_train import OffPolicyTrain
from src.train.on_policy_train import OnPolicyTrain
from src.train.policy_train import TrainerFactory


TrainerFactory.register_trainer("DQN", OffPolicyTrain)
TrainerFactory.register_trainer("PPO", OnPolicyTrain)
TrainerFactory.register_trainer("SAC", OffPolicyTrain)
