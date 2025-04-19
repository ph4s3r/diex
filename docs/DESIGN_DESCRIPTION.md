# DESIGN DECISIONS

...TODO

# ISSUES

## 1

**Description:** the constructed azure docs link is dependent on the folder we point the loader to. this needs to be fixed or at least checked every time we run the thing.

**Solution 0:** no solution yet - check the depth of the directory and craft the URL like that or something
**Solution 1:** no solution yet - we currently display one sample, so its manual now. We could stop if the URL does not resolve or something

## 2

**Description:** there is a bug in unstructured, maybe not a big one - see the loader python file

# ROADMAP

## Features and Issues

### NEXT PLANNED FEATURES

1. create a GUI
2. streamline the process by
   1. calculating / estimating total size of the job
   2. breaking it down to chunks
   3. run one chunk through the pipeline only
   4. techniques we could use: queues / factory / serialization
   5. create a kind-of batch system where there are jobs
(why do we do that? reading files for an hour then failing on the first embedding or something is a bit sketchy.)

3. look into generating vectors from the metadata added (otherwise we should do some non-vector search based on the metadata)

## TESTING
 - Run tests: **`uv run pytest`**
 - Run tests with coverage with console statistics: **`uv run pytest --cov`**
 - Run tests with coverage with HTML statistics: **`uv run pytest --cov --cov-report=html`**

Open `htmlcov/index.html` in any browser to see uncovered implementation.
