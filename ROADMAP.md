# Features and Issues

## NEXT PLANNED FEATURES

1. create a GUI
2. streamline the process by
   1. calculating / estimating total size of the job
   2. breaking it down to chunks
   3. run one chunk through the pipeline only
   4. techniques we could use: queues / factory / serialization
   5. create a kind-of batch system where there are jobs
(why do we do that? reading files for an hour then failing on the first embedding or something is a bit sketchy.)

3. look into generating vectors from the metadata added (otherwise we should do some non-vector search based on the metadata)
