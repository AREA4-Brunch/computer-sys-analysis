import abc
from simulation.resources.resource_interfaces import ISimulatedResource
from simulation.resources.standard_resource import StandardResource
from simulation.resources.job_generator import JobGenerator
from simulation.metrics.resource_metrics import ISimulatedResourceMetrics
from simulation.resources.processing_time_counter import ProcessingTimeCounter
from simulation.resources.jobs_counter import JobsCounter
from simulation.utils.timer import ITimer


class IFactoryMethodConfig(abc.ABC):
    """ Configuration needed to create an instance within some
        factory method.
    """
    pass


class IResourceFactory(abc.ABC):
    @abc.abstractmethod
    def create_std_resource(
        self,
        config: IFactoryMethodConfig,
        *resource_constructor_args,
        **resource_constructor_kwargs,
    ) -> StandardResource:
        pass

    @abc.abstractmethod
    def create_job_generator(
        self,
        config: IFactoryMethodConfig,
        *resource_constructor_args,
        **resource_constructor_kwargs,
    ) -> JobGenerator:
        pass


class ResourceFactory(IResourceFactory):
    def create_std_resource(
        self,
        config: IFactoryMethodConfig,
        *resource_constructor_args,
        **resource_constructor_kwargs,
    ) -> StandardResource:
        return StandardResource(
            *resource_constructor_args,
            **resource_constructor_kwargs
        )

    def create_job_generator(
        self,
        config: IFactoryMethodConfig,
        *resource_constructor_args,
        **resource_constructor_kwargs,
    ) -> JobGenerator:
        return JobGenerator(
            *resource_constructor_args,
            **resource_constructor_kwargs
        )


class MetricsTrackingFactoryMethodConfig(IFactoryMethodConfig):
    # to_track: dict

    def __init__(
        self,
        to_track: dict,
        timer: ITimer,
        metrics_registry: ISimulatedResourceMetrics,
    ) -> None:
        super().__init__()
        self.to_track = to_track
        self.timer = timer
        self.metrics_registry = metrics_registry


class MetricsTrackingResourceFactory(IResourceFactory):
    def create_std_resource(
        self,
        config: MetricsTrackingFactoryMethodConfig,
        *resource_constructor_args,
        **resource_constructor_kwargs,
    ) -> StandardResource:
        resource = StandardResource(
            *resource_constructor_args,
            **resource_constructor_kwargs
        )
        resource = self._add_metrics_trackers(resource, config)
        return resource

    def create_job_generator(
        self,
        config: MetricsTrackingFactoryMethodConfig,
        *resource_constructor_args,
        **resource_constructor_kwargs,
    ) -> JobGenerator:
        resource = JobGenerator(
            *resource_constructor_args,
            **resource_constructor_kwargs
        )
        resource = self._add_metrics_trackers(resource, config)
        return resource

    def _add_metrics_trackers(
        resource: ISimulatedResource,
        config: MetricsTrackingFactoryMethodConfig,
    ) -> ISimulatedResource:
        if 'throughput' or 'usage' in config.to_track:
            resource = ProcessingTimeCounter(
                resource, config.timer, config.metrics_registry
            )

        if 'num_jobs' in config.to_track:
            resource = JobsCounter(
                resource, config.timer, config.metrics_registry
            )

        return resource
