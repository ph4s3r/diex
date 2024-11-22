#####
#   about: Script to connect to a chroma server
#  author: Peter Karacsonyi <peterkaracsonyi85@gmail.com>
#    date: 22 Nov 2024
# license: GNU General Public License, version 2
#####

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
chroma_client = chromadb.HttpClient(
    host="51.103.224.94",
    port=8000
)
print("trying to ping the server")
print(chroma_client.heartbeat())
