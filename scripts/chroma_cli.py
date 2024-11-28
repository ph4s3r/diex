#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI App: Chroma Server Manager

Author: Peter Karacsonyi <peterkaracsonyi85@gmail.com>
Date: 22 Nov 2024
License: GNU General Public License, version 2
"""

import os
import sys
import json
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import box
import questionary
from typing import List
from rich.pretty import pprint

console = Console()

# Configuration
CHROMA_HOST = "chroma.d3eyekfegubub9fz.switzerlandnorth.azurecontainer.io"
CHROMA_PORT = 8000

EMBEDDING_MODELS = [
    "dunzhang/stella_en_1.5B_v5",
    "text-embedding-3-small",
    "text-embedding-3-large"
]

class ChromaCLI:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client = self.connect_to_chroma()
        self.selected_collection = None
        self.embedding_model = None

    def connect_to_chroma(self):
        try:
            client = chromadb.HttpClient(host=self.host, port=self.port)
            console.print("[bold green]Connected to Chroma server successfully![/bold green]")
            return client
        except Exception as e:
            console.print(f"[bold red]Failed to connect to Chroma server: {e}[/bold red]")
            sys.exit(1)

    def display_server_info(self):
        console.print("\n[bold cyan]Server Information[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Info", style="dim", width=20)
        table.add_column("Value")
        
        try:
            heartbeat = self.client.heartbeat()
            table.add_row("Heartbeat", json.dumps(heartbeat))
        except Exception as e:
            table.add_row("Heartbeat", f"[red]Error: {e}[/red]")
        
        try:
            version = self.client.get_version()
            table.add_row("Version", version)
        except Exception as e:
            table.add_row("Version", f"[red]Error: {e}[/red]")
        
        try:
            max_batch_size = self.client.get_max_batch_size()
            table.add_row("Max Batch Size", str(max_batch_size))
        except Exception as e:
            table.add_row("Max Batch Size", f"[red]Error: {e}[/red]")
        
        console.print(table)

    def list_collections(self) -> List[chromadb.Collection]:
        try:
            collections = self.client.list_collections()
            if not collections:
                console.print("[yellow]No collections found on the server.[/yellow]")
                sys.exit(0)
            table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL_DOUBLE_HEAD)
            table.add_column("No.", style="dim", width=6)
            table.add_column("Collection Name", style="cyan")
            table.add_column("Number of Embeddings", style="green")
            table.add_column("Model", style="yellow")
            for idx, coll in enumerate(collections, start=1):
                count = coll.count()  # Number of embeddings in the collection
                model = coll.metadata.get("model", "Not specified") if coll.metadata else "Not specified"
                table.add_row(str(idx), coll.name, str(count), model)
            console.print("\n[bold cyan]Available Collections[/bold cyan]")
            console.print(table)
            return collections
        except Exception as e:
            console.print(f"[bold red]Error listing collections: {e}[/bold red]")
            sys.exit(1)

    def select_collection(self, collections: List[chromadb.Collection]):
        choices = [f"{idx + 1}. {coll.name} (Embeddings: {coll.count()})" for idx, coll in enumerate(collections)]
        choice = questionary.select(
            "Select a collection to work on:",
            choices=choices
        ).ask()
        if choice is None:
            console.print("[red]No collection selected. Exiting.[/red]")
            sys.exit(0)
        selected_index = int(choice.split(".")[0]) - 1
        self.selected_collection = collections[selected_index]
        console.print(f"[bold green]Selected Collection:[/bold green] {self.selected_collection.name}")
        
        # Retrieve and display metadata
        try:
            metadata = self.selected_collection.metadata
            model = metadata.get("model", "Not specified") if metadata else "Not specified"
            last_updated = metadata.get("lastupdated", "Not specified") if metadata else "Not specified"
            
            console.print("\n[bold magenta]Collection Metadata[/bold magenta]")
            table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL_DOUBLE_HEAD)
            table.add_column("Attribute", style="dim", width=20)
            table.add_column("Value")
            table.add_row("Model", model)
            table.add_row("Last Updated", last_updated)
            console.print(table)
            
            # Set embedding model from metadata
            self.embedding_model = model
            if self.embedding_model in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]:
                ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name=self.embedding_model
                )
            else:
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
            
            # https://docs.trychroma.com/guides/embeddings chroma embedding function wrappers need to be set when working with a collection
            self.selected_collection._embedding_function = ef
            console.print(f"[bold green]Embedding function set to '{self.selected_collection._embedding_function}' based on the metadata.[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]Error retrieving or setting metadata: {e}[/bold red]")
            sys.exit(1)

    def peek_collection(self):
        try:
            results = self.selected_collection.peek()
            console.print("[bold cyan]Peek into Collection[/bold cyan]")

            if not results:
                console.print("[yellow]No items to peek in the collection.[/yellow]")
                return
            
            # returns a GetResult object:

            # class GetResult(TypedDict):
            #     ids: List[ID]
            #     embeddings: Optional[
            #         Union[Embeddings, PyEmbeddings, NDArray[Union[np.int32, np.float32]]]
            #     ]
            #     documents: Optional[List[Document]]
            #     uris: Optional[URIs]
            #     data: Optional[Loadable]
            #     metadatas: Optional[List[Metadata]]
            #     included: Include
            
        
            pprint("IDs: ")
            pprint(results.get('ids'))
            pprint("Embeddings: ")
            pprint(results.get('embeddings'))
            pprint("Metadatas: ")
            pprint( results.get('metadatas'))
            pprint("Document: ")
            # for doc in results.get('documents'):
            #     for chunk in doc:
            #         print(chunk)
            pprint(results.get('documents'))
            pprint("Data: ")
            pprint(results.get('data'))
            pprint("URIs:")
            pprint( results.get('uris'))
            pprint("Included: ")
            pprint(results.get('included'))
    
        except Exception as e:
            console.print(f"[bold red]Error peeking into collection: {e}[/bold red]")

    def delete_collection(self):
        confirm = questionary.confirm(f"Are you sure you want to delete the collection '{self.selected_collection.name}'?").ask()
        if confirm:
            try:
                self.selected_collection.delete()
                console.print(f"[bold green]Collection '{self.selected_collection.name}' deleted successfully.[/bold green]")
                sys.exit(0)
            except Exception as e:
                console.print(f"[bold red]Error deleting collection: {e}[/bold red]")
        else:
            console.print("[yellow]Deletion cancelled.[/yellow]")

    def perform_query(self):
        # Only allow queries if an embedding model has been selected
        if not self.embedding_model:
            console.print("[red]No embedding model selected. Please select a collection with a valid embedding model.[/red]")
            return

        def embed_query(text: str):
            query_prompt_name = "s2p_query"

            model = SentenceTransformer(self.embedding_model, trust_remote_code=True)
            query_embeddings = model.encode(text, prompt_name=query_prompt_name)
            
            print(f"Query Embedded with {self.embedding_model} successfully. query_embeddings.shape: {query_embeddings.shape}")

            return query_embeddings

        # Predefined queries
        predefined_queries = [
            "How to set up a GitHub Actions workflow to register an API?",
            "Can virtual networks across tenants be peered?"
        ]
        query_choice = questionary.select(
            "Select a query:",
            choices=predefined_queries + ["Write a new query"]
        ).ask()
        if query_choice is None:
            console.print("[red]No query selected. Operation cancelled.[/red]")
            return
        if query_choice == "Write a new query":
            query_text = Prompt.ask("Enter your query")
        else:
            query_text = query_choice

        # Perform query
        try:
            if self.embedding_model in ["dunzhang/stella_en_1.5B_v5"]:
                query_embedding = embed_query(query_text) 
                results = self.selected_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=2
                )
            else:
                results = self.selected_collection.query(
                    query_texts=[query_text],
                    n_results=2
                )
                    
            console.print("[bold cyan]Query Results[/bold cyan]")
           
            # Print using pprint
            pprint("IDs: ")
            pprint(results.get('ids'))
            pprint("Embeddings: ")
            pprint(results.get('embeddings'))
            pprint("Metadatas: ")
            pprint( results.get('metadatas'))
            pprint("Document: ")
            for doc in results.get('documents'):
                for chunk in doc:
                    print(chunk)
            pprint("Data: ")
            pprint(results.get('data'))
            pprint("URIs:")
            pprint( results.get('uris'))
            pprint("Included: ")
            pprint(results.get('included'))

        except Exception as e:
            console.print(f"[bold red]Error performing query: {e}[/bold red]")


    def view_collection_info(self):
        try:
            count = self.selected_collection.count()
            console.print("\n[bold magenta]Collection Information[/bold magenta]")
            table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL_DOUBLE_HEAD)
            table.add_column("Attribute", style="dim", width=20)
            table.add_column("Value")
            table.add_row("Name", self.selected_collection.name)
            table.add_row("Number of Embeddings", str(count))
            # Add more attributes if needed
            console.print(table)
        except Exception as e:
            console.print(f"[bold red]Error retrieving collection info: {e}[/bold red]")

    def main_menu(self):
        while True:
            action = questionary.select(
                "Select an action:",
                choices=[
                    "1) View collection information",
                    "2) Peek into collection",
                    "3) Delete collection",
                    "4) Perform a query",
                    "5) Exit"
                ]
            ).ask()
            
            if action is None or action == "5) Exit":
                console.print("[bold yellow]Goodbye![/bold yellow]")
                sys.exit(0)
            
            if action.startswith("1"):
                self.view_collection_info()
            
            elif action.startswith("2"):
                self.peek_collection()
            
            elif action.startswith("3"):
                self.delete_collection()
            
            elif action.startswith("4"):
                self.perform_query()
            
            else:
                console.print("[red]Invalid selection. Please try again.[/red]")

    def run(self):
        console.print("[bold magenta]Welcome to the Chroma Server CLI App[/bold magenta]")
        
        # Display server info
        self.display_server_info()
        
        # List collections
        collections = self.list_collections()
        
        # Select a collection
        self.select_collection(collections)
        
        # Main menu
        self.main_menu()

def main():
    cli_app = ChromaCLI(CHROMA_HOST, CHROMA_PORT)
    cli_app.run()

if __name__ == "__main__":
    main()
