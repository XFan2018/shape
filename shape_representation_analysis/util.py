from datetime import datetime
from enum import Enum

TODAY_FORMAT = "date_%d-%m-%Y"


def today(dt_format: str) -> str:
    now = datetime.now()
    return now.strftime(dt_format)


class LRSchedulerCreator:
    """LRScheduleCreator(lr_type)
        create lambda function that will be used by the scheduler
    """

    class Type(Enum):
        # two types of lambda function that changes the learning rate (lr drop exponentially and quadratically)
        EXPONENTIAL = 1
        QUADRATIC = 2

    def __init__(self, lr_type: Type):
        self._lr_type = lr_type

    def __call__(self):
        if self._lr_type == self.Type.EXPONENTIAL:
            return lambda epoch: 0.99 ** epoch if 0.99 ** epoch > 0.005 else 0.005
        elif self._lr_type == self.Type.QUADRATIC:
            return lambda epoch: 3.98e-10 * (epoch - 50000) ** 2 + 0.005 if epoch < 50000 else 0.005
        else:
            raise Exception("scheduler type not exists")

    @property
    def lr_type(self):
        return self._lr_type

    @lr_type.setter
    def lr_type(self, value):
        self._lr_type = value
