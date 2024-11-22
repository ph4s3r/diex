import logging
from logconf import setup_logging

from dependency_injector.wiring import Provide, inject
from container import Container
from components.services.vector_indexer import VectorIndexer


@inject
def main(service: VectorIndexer = Provide[Container.vector_indexer_service]) -> None:
    
    setup_logging()
    main_logger = logging.getLogger('main')
    main_logger.info("STARTING DIEX VECTOR INDEXER")

    path = "files/vnets.md"

    try:
        service.process(path)
    except Exception as e:
        main_logger.error(f"DIEX VECTOR INDEXER ENCOUNTERED AN ERROR: {e}")
    finally:
        main_logger.info("DIEX VECTOR INDEXER FINISHED")

if __name__ == "__main__":

    container = Container()
    container.wire(modules=[__name__])
    main()
