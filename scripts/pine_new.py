import sys
sys.dont_write_bytecode = True

import os
import pickle
import pprint
from pinecone.grpc import PineconeGRPC as Pinecone


class PineConeClient():

    def __init__(self
    ) -> None:

        self.max_batch_size = 100
        self.api_key = "pcsk_r1c8Y_HvcdmYqVuYkhCLtjLYBTzGbDSYpJn3Mxf47SCkmXHBGUHLF2n6r9VcswV4gHzos" #os.getenv("PINECONE_API_KEY")
        self.index_name = "universe"
        self.index_host = f"https://{self.index_name}-226a147.svc.aped-4627-b74a.pinecone.io"
        self.namespace = "tests"
        self.cached_vectors = {}

        # this value should be coming from the local cache
        self.cached_vectors_count = None
        self.local_cache = "cache/vectors.dict"
        
        # this data should be coming from the api
        self.ns_vectorcount = None

        self.pc = Pinecone(
            api_key=self.api_key
            )

        self.index = self.pc.Index(
            name=self.index_name,
            host=self.index_host
            )
        
        self.stats = None
        print("successfully connected to PineCone!")
        self._refresh_index_stats()

        self.read_units_used = 0

        if not self._cache_synced():
            print("Cache is not synced, please run the refresh_cache() function to sync it up.")
            # self.refresh_cache()


        
    def _refresh_index_stats(self):
        try:
            self.stats = self.index.describe_index_stats()
            print("successfully refreshed index stats:")
            pprint.pprint(self.stats)
            
            self.ns_vectorcount = self.stats["namespaces"][self.namespace]["vector_count"]
        except Exception as e:
            print(f"There was an error refreshing index stats :/ we must exit. Error: {e}")
            os._exit(231)

        return True


    def query(self, inputvector, top_k=3):

        results = self.index.query(
            namespace=self.namespace,
            vector=inputvector,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            show_progress=True,
        )

        return results

    def refresh_cache(self):

        ru = 0

        for ids in self.index.list(limit=self.max_batch_size, namespace=self.namespace):
            
            fetch_response = self.index.fetch(
                ids=ids,
                namespace=self.namespace
            )
            
            ru += int(fetch_response.get("usage", {}).get("read_units", 0))

            for vector_id, vector_data in fetch_response["vectors"]:
                self.cached_vectors[vector_id] = {
                    'metadata': vector_data["metadata"]
                }
            break

        print(f"{ru} read units used for this operation")
        
        self.read_units_used += ru

        print(f"Trying to serialize the vectors to cache.")
        directory = os.path.dirname(self.local_cache)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(self.local_cache, "wb") as file:
            pickle.dump(self.cached_vectors, file)

        print(f"Successfully pickled the vectors.")
        self.cached_vectors_count = len(self.cached_vectors)




    
    def _cache_synced(self):
        """
        The function checks if we have a local replica, and loads the self.all_vectors with it, otherwise does not do anything
        """

        print("checking if the replica is available...")

        cache_intact = False

        if os.path.exists(self.local_cache):
            print("Replica file exists.")
            try:
                print("trying to read vectors...")
                with open(self.local_cache, "rb") as file:
                    print("opened file, trying to load with pickle.")
                    self.cached_vectors = pickle.load(file)
                    
                    if len(self.cached_vectors) == 0:
                        print(f"vector file read successfully, but there are no vectors in it :(")
                        cache_intact = False
                    elif self.cached_vectors_count != self.ns_vectorcount:
                        print(f"we have a different amount of vectors in the cache than actually in the namespace..")
                        print(f"ns: {self.ns_vectorcount} vs cache: {self.cached_vectors_count}")
                        cache_intact = False
                    else:
                        self.cached_vectors_count = len(self.cached_vectors)
                        print(f"vector dict file read successfully, seem like we have {self.cached_vectors_count} vectors")
                        cache_intact = True
            except KeyError as ke:
                print(f"The cache file is unreadable, continuing. Error: {ke}")
                cache_intact = False     
            except Exception as e:
                print(f"An error occured during reading the vector file: {e}")
                cache_intact = False
        else: # os.path. not exists(self.local_cache)
            cache_intact = False
    
        return cache_intact
            

mpc = PineConeClient()

mpc.refresh_cache()

# to fetch all vectors, and cache them, run the refresh_cache() function
