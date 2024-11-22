import os, sys
from container import Container
from dependency_injector.wiring import Provide, inject
from components.services.vector_indexer import VectorIndexer

# no bytecode please
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
  
@inject
def main(service: VectorIndexer = Provide[Container.vector_indexer_service]) -> None:
    path = "files/vnets.md"
    service.process(path)


if __name__ == "__main__":
    container = Container()
    container.wire(modules=[__name__])
    main()
