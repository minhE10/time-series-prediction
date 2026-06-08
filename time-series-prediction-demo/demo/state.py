from core.inference import ForecastService
from core.registry import Registry


def create_demo_state():
    registry = Registry.load()
    service = ForecastService(registry)
    return registry, service
