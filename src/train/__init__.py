from src.train.off_policy_train import OffPolicyTrain
from src.train.on_policy_train import OnPolicyTrain
from src.train.policy_train import TrainerFactory

TRAIN_MAP = {
    "OnPolicyTrain": OnPolicyTrain,
    "OffPolicyTrain": OffPolicyTrain,
}
