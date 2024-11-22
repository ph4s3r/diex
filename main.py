import logging
from logconf import setup_logging

from dependency_injector.wiring import Provide, inject, providers
from container import Container
from components.services.vector_indexer import VectorIndexer


@inject
def main(service: VectorIndexer = Provide[Container.vector_indexer_service]) -> None:
   
    setup_logging()

    path = "files/vnets.md"

    service.process(path)

if __name__ == "__main__":

    container = Container()

    container.config.from_yaml('configs/config.yaml', required=True)

    container.wire(modules=[__name__])
    main()
