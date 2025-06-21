"""
This module defines a modular and extensible curriculum learning system.

It provides an abstract base class, `Curriculum`, which defines the interface
for any curriculum component. A concrete implementation, `LinearCurriculum`,
is also provided, which allows for the linear interpolation of a parameter
based on a given performance metric (e.g., success rate).

This design allows different aspects of the environment's difficulty
(e.g., success thresholds, penalty weights) to be controlled by separate,
independently configurable curriculum objects.
"""
import abc
from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    """
    Configuration dataclass for a curriculum component.

    This structure holds the parameters that define how a value should be
    scheduled over the course of training based on a performance metric.

    Attributes:
        start_value: The initial value of the parameter being scheduled.
        end_value: The final value of the parameter once the curriculum is complete.
        start_metric_val: The metric value at which the curriculum begins to progress.
                          Before this point, the value remains `start_value`.
        end_metric_val: The metric value at which the curriculum is considered
                        complete. At or beyond this point, the value will be
                        `end_value`.
    """
    start_value: float
    end_value: float
    start_metric_val: float
    end_metric_val: float


class Curriculum(abc.ABC):
    """
    Abstract base class for a curriculum.

    This class defines the basic interface for a curriculum component. Subclasses
    must implement the `update` method, which defines how the curriculum's
    value changes in response to a performance metric.
    """

    def __init__(self, config: CurriculumConfig):
        """
        Initializes the curriculum.

        Args:
            config: A `CurriculumConfig` object containing the parameters
                    for the curriculum.
        """
        self.config = config
        self._current_value: float = self.config.start_value
        self._progress: float = 0.0

    @abc.abstractmethod
    def update(self, metric_value: float):
        """
        Updates the curriculum's internal state based on a metric.

        This method should be called periodically (e.g., every step or episode)
        with a relevant performance metric, such as the agent's success rate.
        It updates the internal `_progress` and `_current_value` attributes.

        Args:
            metric_value: The current value of the performance metric being used
                          to drive the curriculum.
        """
        pass

    @property
    def current_value(self) -> float:
        """Returns the current value of the curriculum parameter."""
        return self._current_value

    @property
    def progress(self) -> float:
        """Returns the current progress of the curriculum (from 0.0 to 1.0)."""
        return self._progress


class LinearCurriculum(Curriculum):
    """
    A curriculum that linearly interpolates a parameter from a start to an end value.

    The interpolation is driven by a performance metric. As the metric increases
    from a defined start point to an end point, the curriculum's value transitions
    linearly from its start value to its end value.
    """

    def update(self, metric_value: float):
        """
        Updates the curriculum value based on the metric.

        The value is interpolated linearly between `start_value` and `end_value`.
        The interpolation is driven by where the `metric_value` falls between
        `start_metric_val` and `end_metric_val`. If the metric is below the start
        threshold, the progress is 0. If it's above the end threshold, the
        progress is 1.

        Args:
            metric_value: The current value of the performance metric.
        """
        # Avoid division by zero if the metric range is invalid.
        if self.config.end_metric_val <= self.config.start_metric_val:
            self._progress = 1.0 if metric_value >= self.config.end_metric_val else 0.0
        else:
            # Calculate progress as a fraction of the way through the metric range.
            raw_progress = (metric_value - self.config.start_metric_val) / \
                (self.config.end_metric_val - self.config.start_metric_val)

            # Clamp the progress to be within the [0, 1] range.
            self._progress = max(0.0, min(1.0, raw_progress))

        # Linearly interpolate the value based on the calculated progress.
        self._current_value = self.config.start_value + self._progress * \
            (self.config.end_value - self.config.start_value)
