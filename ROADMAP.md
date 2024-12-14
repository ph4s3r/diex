# NEXT PLANNED FEATURES

- streamline the whole pipeline with queues / factory / serialization
- look into generating vectors from the metadata added (otherwise we should do some non-vector search based on the metadata)

# PROBLEMS

- there is a bug in unstructured, maybe not a big one - see the loader python file
- is it possible that we don't need the splitter ? the mdloader has a 5k char limit and it seems like it is never more than the seq length (it was before!)
