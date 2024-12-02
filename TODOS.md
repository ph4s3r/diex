# TODOs

- is there a way to install a requirement only when we use that particular module? maybe someone implements something but it should not be necessarily a part of the whole image / solution 

# PROBLEMS

- the file.read in the unstructured markdown loader already converts the md header hashtags to some other characters. maybe there is no information lost, but the split by headers is I guess already not working... we cannot build a nice metadata hierarchy based structure... the metadata is currently the header of these azure docs like title authors topic etc...
- the stella model has no onnx built-in so we need to create that manually somehow to be able to use the TEI (Huggungface inference)
- the torch model seemingly gets overwhelmed with a lot of inputs - maybe we need to split that to batches as well.