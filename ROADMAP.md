# Features and Issues

## NEXT PLANNED FEATURES

- streamline the whole pipeline with queues / factory / serialization
- look into generating vectors from the metadata added (otherwise we should do some non-vector search based on the metadata)

## ISSUES

### MAJOR ISSUES

- the api-embedder sometimes cannot retrieve vectors since sometimes it is bigger than than seq length. The real problem is that we don't delete the document related to it and then we try to upsert the doc while the vector is empty (list of Nones). Even I reduced the tiktoken seq len to 50% it does not help. there is a problem there as well :/
- the constructed azure docs link is dependent on the folder we point the loader to. this needs to be fixed or at least checked every time we run the thing

### MINOR ISSUES

- there is a bug in unstructured, maybe not a big one - see the loader python file
